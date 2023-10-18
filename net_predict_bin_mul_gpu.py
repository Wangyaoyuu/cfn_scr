#!/usr/bin/env python3
# nohup python -Bu src_distributed/train_adapt_neg.py &> docker_adapt_neg.log &
import os
import h5py
import time
import numpy as np
import torch as pt

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from constant import *
from model import *
from data import *

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5681'


def main_function(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    #/home/wangyaoyu/shrec/data
    # datafn = '/home/wangyaoyu/shrec/data/' + dataset + '.hdf5'
    predictfn = '/home/wangyaoyu/shrec/data/' + dataset + '_predict.hdf5'
    srcfn = '/home/wangyaoyu/SHREC/output_adapt_neg_bin_coarser/data21.chk95'
    outfn = '/home/wangyaoyu/SHREC/output_adapt_neg_bin_coarser/%s_prob_95.h5py' % dataset

    random.seed(20220123)
    pt.cuda.set_device(rank)

    print('#loading predict dataset', predictfn, '...')
    cache = h5py.File(predictfn, 'r')
    p_data = cache['data'][()]
    print('#data:', p_data.shape, p_data.dtype)
    p_label = cache['label'][()]
    print('#label:', p_label.shape, p_label.dtype)
    p_dist = cache['dist'][()]
    print('#dist:', p_dist.shape, p_dist.dtype)
    p_center = cache['center'][()]
    print('#center:', p_center.shape, p_center.dtype)

    print(p_data.shape, p_label.shape, p_dist.shape)

    testset = testSet_origin(p_data, p_label, p_dist)
    testsampler = gdrSampler_origin(p_label.shape, rank)
    testloader = DataLoader(testset, batch_size=eval_batchsize, sampler=testsampler,
           num_workers=2, prefetch_factor=eval_batchsize)

    print()
    print('#building model ...')
    batchidx, best = 0, 0.4
    epochsize = (len(testsampler) + eval_batchsize - 1) // eval_batchsize // world_size // 2
    print(epochsize)

    model = CentralResNet3D().cuda()
    ddp_model = DDP(model, device_ids=[rank])

    if os.path.exists(srcfn):
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            state = torch.load(srcfn, map_location=map_location)
            ddp_model.load_state_dict(state['model'], strict=False)

            print('#model load:', srcfn)

        except Exception as e:
            print('#model transfer:', srcfn, e)


    print('#model size:', np.sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print('#model dist:', world_size)
    print()

    print('#testing model ...')
    summary = []
    testlabel = np.ones([1, *p_label.shape[1:], labelsize], dtype=np.float16) * -1
    testdist = np.ones([1, *p_dist.shape[1:], gridpow2], dtype=np.float16) * -1
    tepoch = tcheck = time.perf_counter()

    for batchtest in testloader:

        # eval
        ddp_model.eval()
        with pt.no_grad():
            x = batchtest['data'].clone().detach().cuda()
            _, yy, zz = ddp_model(x)

            index = batchtest['index'].cuda()
            yy_argmax = pt.argmax(yy, dim=-1).cuda()
            zz_argmax = pt.argmax(zz, dim=-1).cuda()

            index_output = [torch.zeros_like(index).cuda() for i in range(world_size)]
            yy_argmax_output = [torch.zeros_like(yy_argmax).cuda() for i in range(world_size)]
            zz_argmax_output = [torch.zeros_like(zz_argmax).cuda() for i in range(world_size)]
            yy_output = [torch.zeros_like(yy).cuda() for i in range(world_size)]
            zz_output = [torch.zeros_like(zz).cuda() for i in range(world_size)]

            # allgather
            torch.distributed.all_gather(index_output ,index)
            torch.distributed.all_gather(yy_argmax_output ,yy_argmax)
            torch.distributed.all_gather(zz_argmax_output, zz_argmax)
            torch.distributed.all_gather(yy_output, yy)
            torch.distributed.all_gather(zz_output, zz)

            # update
            for first, second, third in zip(index_output, yy_output, zz_output):
                for i, j, k in zip(first.tolist(), F.softmax(second, dim=-1).tolist(), F.softmax(third, dim=-1).tolist()):
                    testlabel[i[0], i[1], i[2], i[3]], testdist[i[0], i[1], i[2], i[3]] = j, k

            testsampler.update(index_output, yy_argmax_output, zz_argmax_output)

            batchidx += 1
            if batchidx > epochsize * epochlast: break

            yy_argmax, zz_argmax = yy_argmax.cpu(), zz_argmax.cpu()
            accy = pt.mean((yy_argmax == batchtest['label']).type_as(x)).item() * 100
            accz = pt.mean((zz_argmax == batchtest['dist']).type_as(x)).item() * 100

            t = pt.tensor([accy, accz]).cuda()
            torch.distributed.all_reduce(t)
            summary.append(t.cpu().numpy() / world_size)

            epoch = batchidx / epochsize
            msg = np.mean(np.array(summary), axis=0)
            if rank == 0:
                print('#check[%.3f]: %.1f%% %.1f%%' % (epoch, *msg))
            assert (not np.isnan(msg[0]))  # debug

            # checkpoint
            if rank > 0: continue
            # print(yy_argmax, zz_argmax)
            # print()

            tcurr = time.perf_counter()
            if batchidx % epochsize == 0:
                lr = 0
                epoch = batchidx // epochsize
                tdiff = (tcurr - tepoch) / 60
                msg, summary = np.mean(np.array(summary), axis=0), []

                with h5py.File(outfn, 'w') as f:
                    f.create_dataset('testlabel', data=testlabel)
                    f.create_dataset('testdist', data=testdist)
                    f.create_dataset('weight', data=testsampler.weight)
                    f.create_dataset('state', data=testsampler.state)

                print('#epoch[%.3f]: %.1f%% %.1f%% %.1e %.1fm' % (epoch, *msg, lr, tdiff))
                print()
                assert (not np.isnan(msg[0]))  # debug
                tepoch = tcheck = tcurr


    print()
    print('#done!!!')


if __name__ == '__main__':


    mp.spawn(main_function,
        args=(world_size,),
        nprocs=world_size,
        join=True)

