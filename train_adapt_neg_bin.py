#!/usr/bin/env python3

import os
import h5py
import time
import numpy as np
import torch as pt

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from constant import *
from preprocess import *
from model import *
from data import *

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5679'

# softmax on last dimension of p
def lossFocal(p, gt, bg, size, power=3**.5):
    shape = p.shape
    p, gt = p.reshape([-1, shape[-1]]), gt.reshape(-1)
    pp = F.softmax(p, dim=-1)[range(len(gt)), gt]
    #loss = F.cross_entropy(p, gt, reduction='none')  # non-focal
    loss = (1 - pp).pow(power) * F.cross_entropy(p, gt, reduction='none')  # focal

    with pt.no_grad():
        match = (gt.reshape([len(gt), 1]) == pt.arange(size, dtype=pt.int64, device='cuda').reshape([1, size])).type_as(p)
        count = pt.sum(match, dim=0)
        count[count < 1] = 1
        weight = 1e4 * pt.sum(match / count, dim=1)  # non-focal
        #weight = 1e4 * pt.sum(match / count, dim=1) * pp.pow(1/power)  # focal
        norm = pt.sum(weight)
        assert(norm > 0)  # debug

    return pt.sum(weight * loss) / norm


def main_function(rank, world_size):
    sched_chk, sched_cycle = 4, 32

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    #/home/wangyaoyu/shrec/data
    datafn = '/home/wangyaoyu/shrec/data/' + dataset + '.hdf5'
    srcfn = '/home/wangyaoyu/SHREC/output_adapt_neg_bin_coarser/data21.chk'
    chkfn = '/home/wangyaoyu/SHREC/output_adapt_neg_bin_coarser/%s.chk' % dataset
    # weight_load_fn = '/home/wangyaoyu/SHREC/output_adapt_neg_bin_coarser/weightdata21.hdf595'
    outfn = '/home/wangyaoyu/SHREC/output_adapt_neg_bin_coarser/weight%s.hdf5' % dataset

    random.seed(20220123)
    pt.cuda.set_device(rank)

    # print('#loading neight weight ...')
    # cache = h5py.File(weight_load_fn, 'r')
    # negweight_load = cache['weight'][()]
    # cache.close()
    # print(f'#loading neight weight success {weight_load_fn}')

    print('#loading dataset', datafn, '...')
    cache = h5py.File(datafn, 'r')
    data = cache['data'][:datasize]
    print('#data[train]:', data.shape, data.dtype)
    label = cache['label'][:datasize]
    print('#label[train]:', label.shape, label.dtype)
    dist = cache['dist'][:datasize]
    print('#dist[train]:', dist.shape, dist.dtype)
    center = cache['center'][()]
    center = center[center[:, 0] < datasize]
    print('#center[train]:', center.shape, center.dtype)

    trainsampler_neg_bin = snrSampler_neg_bin(label, dist, center, rank)
    # trainsampler_neg_bin.neg_weight = negweight_load
    neg_length = trainsampler_neg_bin.neg_length
    trainset = trainSet_neg_bin(data, label, dist, neg_length, augment=True)
    trainloader = DataLoader(trainset, batch_size=batchsize, sampler=trainsampler_neg_bin,
            num_workers=2, prefetch_factor=batchsize)

    data = cache['data'][datasize:-1]
    print('#data[valid]:', data.shape, data.dtype)
    label = cache['label'][datasize:]
    print('#label[valid]:', label.shape, label.dtype)
    dist = cache['dist'][datasize:]
    print('#dist[valid]:', dist.shape, dist.dtype)
    center = cache['center'][()]
    center = center[center[:, 0] >= datasize]
    center[:, 0] -= datasize
    print('#center[valid]:', center.shape, center.dtype)

    validsampler = snrSampler(label, dist, center, rank)
    validset = trainSet(data, label, dist, augment=False)
    validloader = DataLoader(validset, batch_size=batchsize, sampler=validsampler,
            num_workers=2, prefetch_factor=batchsize)

    #testsampler = gdrSampler(label[:len(data)-datasize].shape)
    #testset = testSet(data[datasize:])
    #testloader = DataLoader(testset, batch_size=batchsize, sampler=testsampler,
    #        num_workers=2, prefetch_factor=batchsize)
    cache.close()
    print()

    print('#building model ...')
    batchidx, best = 0, 0.4
    epochsize = (len(trainsampler_neg_bin) + batchsize - 1) // batchsize // world_size

    print(epochsize)
    # input()

    model = CentralResNet3D().cuda()
    ddp_model = DDP(model, device_ids=[rank])

    # ---------------
    pgrp0, pgrp1 = [], []
    for n, p in ddp_model.named_parameters():
        if n.startswith('head') and n.endswith('weight'):
            pgrp1.append(p)
        else:
            pgrp0.append(p)
    pgrp = [{'params': pgrp0, 'weight_decay': 0}, {'params': pgrp1, 'weight_decay': 1}]
    optimizer, sched_lr = optim.SGD(pgrp, lr=lr_init, momentum=0.9), lr_exp * 2
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 2)

    if os.path.exists(srcfn):
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            state = torch.load(srcfn, map_location=map_location)
            ddp_model.load_state_dict(state['model'], strict=False)
            batchidx = state['epoch'] * epochsize
            best = state['best']
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])

            print('#model load:', srcfn)

        except Exception as e:
            print('#model transfer:', srcfn, e)


    print('#model size:', np.sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print('#model dist:', world_size)
    print()

    print('#training model ...')
    summary = []
    tepoch = tcheck = time.perf_counter()

    for batchtrain, batchvalid in zip(trainloader, validloader):
        # schedule
        with pt.no_grad():
            if batchidx % epochsize == 0 and batchidx >= epochsize * 2:
                epoch = batchidx // epochsize
                if epoch % sched_chk == 0:
                    scheduler.base_lrs = [max(lr/2, sched_lr) for lr in scheduler.base_lrs]
                    if epoch >= sched_cycle: scheduler.step(epoch % sched_cycle + sched_cycle - 1)
                    else: scheduler.step(epoch - 1)
                    sched_chk = min(sched_chk * 2, sched_cycle)
                else:
                    if epoch >= sched_cycle: scheduler.step(epoch % sched_cycle + sched_cycle - 1)
                    else: scheduler.step(epoch - 1)

        # train
        ddp_model.train()
        optimizer.zero_grad()
        x = batchtrain['data'].clone().detach().cuda()
        y = batchtrain['label'].clone().detach().cuda()
        z = batchtrain['dist'].clone().detach().cuda()
        _, yy, zz = ddp_model(x)
        lossy = lossFocal(yy, y, 0, labelsize)
        lossz = lossFocal(zz, z, gridpow, gridpow2)
        loss = lossy + lossz/2
        loss.backward()
        nn.utils.clip_grad_value_(ddp_model.parameters(), 1)
        optimizer.step()
        ddp_model.eval()

        with pt.no_grad():

            index = batchtrain['index'].cuda()
            generate_index = batchtrain['generate'].cuda()
            yy_argmax = pt.argmax(yy, dim=-1).cuda()
            zz_argmax = pt.argmax(zz, dim=-1).cuda()
            neg_index = batchtrain['neg_index'].cuda()

            index_output = [torch.zeros_like(index).cuda() for i in range(world_size)]
            generate_index_output = [torch.zeros_like(generate_index).cuda() for i in range(world_size)]
            yy_argmax_output = [torch.zeros_like(yy_argmax).cuda() for i in range(world_size)]
            zz_argmax_output = [torch.zeros_like(zz_argmax).cuda() for i in range(world_size)]
            y_output = [torch.zeros_like(y).cuda() for i in range(world_size)]
            z_output = [torch.zeros_like(z).cuda() for i in range(world_size)]
            neg_index_output = [torch.zeros_like(neg_index).cuda() for i in range(world_size)]

            # allgather
            torch.distributed.all_gather(index_output ,index)
            torch.distributed.all_gather(generate_index_output, generate_index)
            torch.distributed.all_gather(yy_argmax_output ,yy_argmax)
            torch.distributed.all_gather(zz_argmax_output, zz_argmax)
            torch.distributed.all_gather(y_output, y)
            torch.distributed.all_gather(z_output, z)
            torch.distributed.all_gather(neg_index_output, neg_index)

            # update
            trainsampler_neg_bin.update(index_output, yy_argmax_output, zz_argmax_output, y_output,
                                        z_output, neg_index_output, generate_index_output)

            batchidx += 1
            if batchidx > epochsize * epochlast: break

            # valid
            # if batchidx % 8 == 0:
            x = batchvalid['data'].clone().detach().cuda()
            _, yy, zz = ddp_model(x)
            yy, zz = yy.cpu(), zz.cpu()
            accy = pt.mean((pt.argmax(yy, dim=-1) == batchvalid['label']).type_as(x)).item() * 100
            accz = pt.mean((pt.argmax(zz, dim=-1) == batchvalid['dist']).type_as(x)).item() * 100

            t = pt.tensor([lossy.item(), lossz.item(), accy, accz]).cuda()
            torch.distributed.all_reduce(t)
            summary.append(t.cpu().numpy() / world_size)

            epoch = batchidx / epochsize
            msg = np.mean(np.array(summary), axis=0)
            if rank == 0:
                print('#check[%.3f]: %.3f %.3f %.1f%% %.1f%%' % (epoch, *msg))
            assert (not np.isnan(msg[0]))  # debug

            # checkpoint
            if rank > 0: continue

            tcurr, lr = time.perf_counter(), optimizer.param_groups[0]['lr']
            if batchidx % epochsize == 0:

                epoch = batchidx // epochsize
                tdiff = (tcurr - tepoch) / 60
                msg, summary = np.mean(np.array(summary), axis=0), []

                # if(epoch % 50 == 0):
                pt.save({'epoch': epoch, 'best': best,
                         'model': ddp_model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()}, chkfn + str(epoch))
                with h5py.File(outfn + str(epoch), 'w') as f:
                    f.create_dataset('weight', data=trainsampler_neg_bin.neg_weight)

                if msg[2] + msg[3] / 2 > best:
                    best = msg[2] + msg[3] / 2
                    pt.save({'epoch': epoch, 'best': best,
                             'model': ddp_model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict()}, chkfn)
                    with h5py.File(outfn, 'w') as f:
                        f.create_dataset('weight', data=trainsampler_neg_bin.neg_weight)

                    print('#epoch[%.3f]: %.3f %.3f %.1f%% %.1f%% %.1e %.1fm *' % (epoch, *msg, lr, tdiff))
                else:
                    print('#epoch[%.3f]: %.3f %.3f %.1f%% %.1f%% %.1e %.1fm' % (epoch, *msg, lr, tdiff))
                print()
                assert (not np.isnan(msg[0]))  # debug
                tepoch = tcheck = tcurr

            # elif tcurr - tcheck > 60:
            #     try:
            #         epoch = batchidx / epochsize
            #         msg = np.mean(np.array(summary), axis=0)
            #         print('#check[%.3f]: %.3f %.3f %.1f%% %.1f%%' % (epoch, *msg))
            #         assert(not np.isnan(msg[0]))  # debug
            #         tcheck = tcurr
            #     except:
            #         continue


    print()
    print('#done!!!')


if __name__ == '__main__':


    mp.spawn(main_function,
        args=(world_size,),
        nprocs=world_size,
        join=True)