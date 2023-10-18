#!/usr/bin/env python3

import random
import numpy as np
import torch as pt

from torch.utils.data import Dataset, Sampler, DataLoader

from constant import *
from preprocess import *

# signal-to-noise sampler
# called by the main thread in case of map-style datasets
class snrSampler(Sampler):
    def __init__(self, label, dist, center, rank, snr=1.0, exp=0.0):
        self.rank = rank
        print('#initializing data sampler ...')
        self.index, self.weight = [], []
        d0 = np.all(dist == 0, axis=-1)
        for i in range(labelsize):
            li = label == i
            self.index.append(np.where(li.reshape(-1))[0])
            w, c = len(self.index[-1]), np.sum(li & d0)
            if i == 0:
                print('#label[%d]: %d' % (i, w))
                self.weight.append(0)
            elif i == 14:
                self.weight.append(1 ** exp)
            elif i == 15:
                self.weight.append(1 ** exp)
            else:
                print('#label[%d]: %d %.2f' % (i, c, w/c))
                self.weight.append((w / c) ** exp)
        print('#pos/particle: %.2f' % (sum([len(i) for i in self.index[1:]]) / len(center)))
        print('#neg/pos: %.2f' % (len(self.index[0]) / sum([len(i) for i in self.index[1:]])))
        self.weight = np.array(self.weight, dtype=np.float32)
        self.weight[0] = np.sum(self.weight) / snr
        print('#weight:', *self.weight)
        self.weight = np.cumsum(self.weight)
        self.weight /= self.weight[-1]
        # number of samples per epoch
        self.epochsize = int(len(center) / (self.weight[-1] - self.weight[0]) + 0.5)
        self.datasize = label.size

    def __iter__(self):
        lstidx, lstaug = list(range(labelsize)), list(range(4))
        while True:
            bobsize = batchsize * world_size * 64
            batchidx = random.choices(lstidx, cum_weights=self.weight, k=bobsize)
            batchaug = random.choices(lstaug, k=bobsize)
            for k in range(self.rank, bobsize, world_size):
                i, j = batchidx[k], batchaug[k]
                yield self.index[i][random.randrange(len(self.index[i]))] + j * self.datasize

    def __len__(self):
        return self.epochsize

samplestep = 4
# gradient descent with restart sampler
# called by the main thread in case of map-style datasets
class gdrSampler(Sampler):
    def __init__(self, shape, rank, label, restart=0.2):
        self.rank = rank
        self.aug_size = label.size
        print('#initializing gdr sampler ...')
        self.shape = list(shape)
        for i in range(1, len(shape)): self.shape[i] //= samplestep
        print('#shape:', *self.shape)
        self.restart = restart
        print('#restart:', self.restart)

        self.weight = np.zeros(self.shape, dtype=np.float32)
        self.initial_value = 2.
        print('#initial value: ', self.initial_value)
        self.weight[:, 1:-1, 1:-1, 1:-1] = self.initial_value
        self.weightmax = 14.14  # neg/pos
        self.state = np.zeros(self.shape, dtype=np.int16) - 1
        self.size = self.weight.size
        self.index = list(range(self.size))
        self.step = [0, 1, 2, 4, 8, 16, 0, -16, -8, -5, -2, -1]  # debug for step[gridpow]

    def __iter__(self):
        lstaug = list(range(4))
        while True:
            bobsize = batchsize * world_size * 64
            batch = random.choices(self.index, weights=self.weight.reshape(-1), k=bobsize)
            batchaug = random.choices(lstaug, k=bobsize)

            for k in range(self.rank, bobsize, world_size):
                i = batch[k]
                aug = batchaug[k]
                if self.state.reshape(-1)[i] < 0 or random.random() < self.restart:
                    j = random.randrange(samplestep ** 3)
                else:
                    j = self.state.reshape(-1)[i]
                self.state.reshape(-1)[i] = -1

                x, i, j = i % self.shape[3] * samplestep + j % samplestep, i // self.shape[3], j // samplestep
                y, i, j = i % self.shape[2] * samplestep + j % samplestep, i // self.shape[2], j // samplestep
                z, i, j = i % self.shape[1] * samplestep + j % samplestep, i // self.shape[1], j // samplestep

                yield (((i * self.shape[1]*samplestep + z) * self.shape[2]*samplestep + y) * self.shape[3]*samplestep + x) + aug * self.aug_size

    def __len__(self):
        return self.size

    def update(self, index, label, center, label_true, center_true):
        for first, second, third, fourth, fifth in zip(index, label, center, label_true, center_true):
            for (i, z, y, x), l, c, l_t, c_t in zip(first.tolist(), second.tolist(), third.tolist(), fourth.tolist(), fifth.tolist()):
                zz, yy, xx = z // samplestep, y // samplestep, x // samplestep
                if l == 0 or gridpow in c:
                    # weight tends to be 1 if the grid contains more backgrounds
                    self.weight[i, zz, yy, xx] = max(self.weight[i, zz, yy, xx] - 1, 1)
                else:
                    # weight tends to be max if the grid contains more signals
                    self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] + 1, self.weightmax)
                    if(l != l_t):
                        self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] * 2, self.weightmax)

                    z0 = min(max(z + self.step[c[0]], 0), self.shape[1]*samplestep - 1)
                    y0 = min(max(y + self.step[c[1]], 0), self.shape[2]*samplestep - 1)
                    x0 = min(max(x + self.step[c[2]], 0), self.shape[3]*samplestep - 1)
                    zz0, yy0, xx0 = z0 // samplestep, y0 // samplestep, x0 // samplestep

                    # weight tends to be max if the grid contains a center
                    self.weight[i, zz0, yy0, xx0] = min(self.weight[i, zz0, yy0, xx0] + 2, self.weightmax)

                    if z == z0 and y == y0 and x == x0: continue  # converge, then restart
                    if zz == zz0 and yy == yy0 and xx == xx0:  # same grid, then gradient descent
                        self.state[i, zz, yy, xx] = ((z0-zz*samplestep) * samplestep + y0-yy*samplestep) * samplestep + x0-xx*samplestep

    def update_improve(self, index, label, center, label_true, center_true):
        for first, second, third, fourth, fifth in zip(index, label, center, label_true, center_true):
            for (i, z, y, x), l, c, l_t, c_t in zip(first.tolist(), second.tolist(), third.tolist(), fourth.tolist(), fifth.tolist()):
                zz, yy, xx = z // samplestep, y // samplestep, x // samplestep
                if l == 0 or gridpow in c:
                    if(l != l_t):
                        self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] + 1, self.weightmax)
                    else:
                        # weight tends to be 1 if the grid contains more backgrounds
                        self.weight[i, zz, yy, xx] = max(self.weight[i, zz, yy, xx] - 1, 1)
                else:
                    # weight tends to be max if the grid contains more signals
                    self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] + 1, self.weightmax)
                    if(l != l_t):
                        self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] * 2, self.weightmax)

                    z0 = min(max(z + self.step[c[0]], 0), self.shape[1]*samplestep - 1)
                    y0 = min(max(y + self.step[c[1]], 0), self.shape[2]*samplestep - 1)
                    x0 = min(max(x + self.step[c[2]], 0), self.shape[3]*samplestep - 1)
                    zz0, yy0, xx0 = z0 // samplestep, y0 // samplestep, x0 // samplestep

                    # weight tends to be max if the grid contains a center
                    self.weight[i, zz0, yy0, xx0] = min(self.weight[i, zz0, yy0, xx0] + 2, self.weightmax)

                    if z == z0 and y == y0 and x == x0: continue  # converge, then restart
                    if zz == zz0 and yy == yy0 and xx == xx0:  # same grid, then gradient descent
                        self.state[i, zz, yy, xx] = ((z0-zz*samplestep) * samplestep + y0-yy*samplestep) * samplestep + x0-xx*samplestep

    def update_half(self, index, label, center, label_true, center_true):
        for first, second, third, fourth, fifth in zip(index, label, center, label_true, center_true):
            for (i, z, y, x), l, c, l_t, c_t in zip(first.tolist(), second.tolist(), third.tolist(), fourth.tolist(), fifth.tolist()):
                zz, yy, xx = z // samplestep, y // samplestep, x // samplestep
                if l == 0 or gridpow in c:
                    # weight tends to be 1 if the grid contains more backgrounds
                    self.weight[i, zz, yy, xx] = max(self.weight[i, zz, yy, xx] - 1, 1)
                else:
                    # weight tends to be max if the grid contains more signals
                    self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] + 1, self.weightmax)
                    if(l != l_t):
                        self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] * 2, self.weightmax)

                    z0 = min(max(z + self.step[c[0]], 0), self.shape[1]*samplestep - 1)
                    y0 = min(max(y + self.step[c[1]], 0), self.shape[2]*samplestep - 1)
                    x0 = min(max(x + self.step[c[2]], 0), self.shape[3]*samplestep - 1)
                    zz0, yy0, xx0 = z0 // samplestep, y0 // samplestep, x0 // samplestep

                    # weight tends to be max if the grid contains a center
                    self.weight[i, zz0, yy0, xx0] = min(self.weight[i, zz0, yy0, xx0] + 2, self.weightmax)

                    if z == z0 and y == y0 and x == x0: continue  # converge, then restart
                    if zz == zz0 and yy == yy0 and xx == xx0:  # same grid, then gradient descent
                        self.state[i, zz, yy, xx] = ((z0-zz*samplestep) * samplestep + y0-yy*samplestep) * samplestep + x0-xx*samplestep

class trainSet(Dataset):
    def __init__(self, data, label, dist, augment=False):
        self.data, self.label, self.dist, self.augment = data, label, dist, augment
        self.shape, self.size = self.label.shape, self.label.size

    def __getitem__(self, idx):
        aug, idx = idx // self.size, idx % self.size
        x, idx = idx % self.shape[3], idx // self.shape[3]
        y, idx = idx % self.shape[2], idx // self.shape[2]
        z, idx = idx % self.shape[1], idx // self.shape[1]

        data = self.data[idx, z:z+gridsize, y:y+gridsize, x:x+gridsize][None, :, :, :].astype(np.float32)
        label = self.label[idx, z, y, x].astype(np.int64)
        dist = self.dist[idx, z, y, x].astype(np.int64)
        index = np.array([idx, z, y, x], dtype=np.int32)

        if not self.augment or aug == 0: pass
        elif aug == 1:
            data = np.rot90(data, k=2, axes=(2, 3))
            dist[0], dist[1], dist[2] = dist[0], (gridpow2-dist[1])%gridpow2, (gridpow2-dist[2])%gridpow2
        elif aug == 2:
            data = np.rot90(data, k=2, axes=(1, 2))
            dist[0], dist[1], dist[2] = (gridpow2-dist[0])%gridpow2, (gridpow2-dist[1])%gridpow2, dist[2]
        elif aug == 3:
            data = np.rot90(data, k=2, axes=(1, 2))
            dist[0], dist[1], dist[2] = (gridpow2-dist[0])%gridpow2, (gridpow2-dist[1])%gridpow2, dist[2]
            data = np.rot90(data, k=2, axes=(2, 3))
            dist[0], dist[1], dist[2] = dist[0], (gridpow2-dist[1])%gridpow2, (gridpow2-dist[2])%gridpow2

        index[0] += aug * datasize

        return {'data':data.copy(), 'label':label, 'dist':dist, 'index':index}

    def __len__(self):
        return self.size

class testSet(Dataset):
    def __init__(self, data, label, dist, augment=False):
        self.aug_size = label.size
        self.data = data
        self.label, self.dist = label, dist
        self.shape = list(self.data.shape)
        self.size = self.shape[0]
        for i in range(1, len(self.shape)):
            self.shape[i] -= 2 * gridhalf
            self.size *= self.shape[i]
        print('#shape:', *self.shape)
        print('#size:', self.size)

        self.augment = augment
        print('#augment:', self.augment)

    def __getitem__(self, idx):
        aug, idx = idx // self.aug_size, idx % self.aug_size
        x, idx = idx % self.shape[3], idx // self.shape[3]
        y, idx = idx % self.shape[2], idx // self.shape[2]
        z, idx = idx % self.shape[1], idx // self.shape[1]

        data = self.data[idx, None, z:z+gridsize, y:y+gridsize, x:x+gridsize].astype(np.float32)
        label = self.label[idx, z, y, x].astype(np.int64)
        dist = self.dist[idx, z, y, x].astype(np.int64)
        index = np.array([idx, z, y, x], dtype=np.int32)

        if not self.augment or aug == 0: pass
        elif aug == 1:
            data = np.rot90(data, k=2, axes=(2, 3))
            dist[0], dist[1], dist[2] = dist[0], (gridpow2-dist[1])%gridpow2, (gridpow2-dist[2])%gridpow2
        elif aug == 2:
            data = np.rot90(data, k=2, axes=(1, 2))
            dist[0], dist[1], dist[2] = (gridpow2-dist[0])%gridpow2, (gridpow2-dist[1])%gridpow2, dist[2]
        elif aug == 3:
            data = np.rot90(data, k=2, axes=(1, 2))
            dist[0], dist[1], dist[2] = (gridpow2-dist[0])%gridpow2, (gridpow2-dist[1])%gridpow2, dist[2]
            data = np.rot90(data, k=2, axes=(2, 3))
            dist[0], dist[1], dist[2] = dist[0], (gridpow2-dist[1])%gridpow2, (gridpow2-dist[2])%gridpow2

        return {'data':data.copy(), 'label':label, 'dist':dist, 'index':index}

    def __len__(self):
        return self.size

class snrSampler_neg_bin(Sampler):
    def __init__(self, label, dist, center, rank, snr=1.0, exp=0.0):
        self.rank = rank
        self.shape, self.size = label.shape, label.size
        print('#shape: ', self.shape)
        print('#initializing data sampler ...')
        #减少 label == 0 预测对的backgroud的采样率
        #增加 label == 0 预测错的backgroud的采样率
        #确保最有效的学习到假阳性的背景，背景中的假阳性作为重点学习甄别的对象

        self.index, self.weight = [], []
        d0 = np.all(dist == 0, axis=-1)
        for i in range(labelsize):
            li = label == i
            self.index.append(np.where(li.reshape(-1))[0])
            w, c = len(self.index[-1]), np.sum(li & d0)
            if i == 0:
                print('#label[%d]: %d' % (i, w))
                self.weight.append(0)
            else:
                print('#label[%d]: %d %.2f' % (i, c, w/c))
                self.weight.append((w / c) ** exp)

        print('#pos/particle: %.2f' % (sum([len(i) for i in self.index[1:]]) / len(center)))
        print('#neg/pos: %.2f' % (len(self.index[0]) / sum([len(i) for i in self.index[1:]])))
        self.weight = np.array(self.weight, dtype=np.float32)
        self.weight[0] = np.sum(self.weight) / snr
        print('#weight:', *self.weight)
        self.weight = np.cumsum(self.weight)
        self.weight /= self.weight[-1]
        # number of samples per epoch
        self.epochsize = int(len(center) / (self.weight[-1] - self.weight[0]) + 0.5)
        self.datasize = label.size

        # false weight ------------------------------------------------
        self.neg_length = 8 * (180//10) * (512//16) * (512//16)
        print('#neg_length:', self.neg_length)
        self.neg_weight = np.ones(self.neg_length, dtype=np.float128) * 6
        self.neg_weight_max = 12


    def __iter__(self):
        lstidx, lstaug, lstneg = list(range(labelsize)), list(range(4)), list(range(self.neg_length))
        while True:
            bobsize = batchsize * world_size * 64
            batchidx = random.choices(lstidx, cum_weights=self.weight, k=bobsize)
            batchaug = random.choices(lstaug, k=bobsize)
            batchidx_neg = random.choices(lstneg, weights=self.neg_weight, k=bobsize)

            for seq, k in enumerate(range(self.rank, bobsize, world_size)):
                i, j = batchidx[k], batchaug[k]
                neg_weight_idx = batchidx_neg[k]
                if i == 0:
                    sample_index = list(np.unravel_index(neg_weight_idx, (8, 18, 32, 32)))
                    tmp = random.randrange(10 * 16 * 16)
                    inter_index = list(np.unravel_index(tmp, (10, 16, 16)))

                    inter_shape = (10, 16, 16)
                    for q in range(1,4):
                        sample_index[q] = sample_index[q] * inter_shape[q-1] + inter_index[q-1]

                    final_index = sample_index[0]
                    for q in range(1,4):
                        final_index = final_index * self.shape[q] + sample_index[q]

                    yield ((final_index + j * self.datasize) * self.neg_length + neg_weight_idx) * 2 + 1
                    # yield (self.index[i][neg_weight_idx] + j * self.datasize) * self.neg_length + neg_weight_idx
                else:
                    yield ((self.index[i][random.randrange(len(self.index[i]))] + j * self.datasize) * self.neg_length + neg_weight_idx) * 2


    def __len__(self):
        return self.epochsize

    def update(self, index, label, center, true_label, true_center, neg_index, generate):
        for a, b, c, d, e, f, g in zip(index, label, center, true_label, true_center, neg_index, generate):
            for (i, z, y, x), l, c, tl, tc, neg_idx, generate_idx in zip(a.tolist(), b.tolist(), c.tolist(), d.tolist(), e.tolist(), f.tolist(), g.tolist()):
                # 判断生成方式，奇数为格子生成，偶数为正负样本比生成
                if generate_idx == 0:
                    continue
                if (l == tl):
                    # weight tends to be 1 if the predicted background is true
                    self.neg_weight[neg_idx] = max(self.neg_weight[neg_idx] - 1, 1)
                else:
                    # weight tends to be max if the predicted background is wrong
                    self.neg_weight[neg_idx] = min(self.neg_weight[neg_idx] + 1, self.neg_weight_max)

class trainSet_neg_bin(Dataset):
    def __init__(self, data, label, dist, neg_length, augment=False):
        self.data, self.label, self.dist, self.augment = data, label, dist, augment
        self.shape, self.size = self.label.shape, self.label.size
        self.neg_length = neg_length

    def __getitem__(self, idx):
        generate, idx = idx % 2, idx // 2
        neg_index, idx = idx % self.neg_length, idx // self.neg_length
        aug, idx = idx // self.size, idx % self.size
        x, idx = idx % self.shape[3], idx // self.shape[3]
        y, idx = idx % self.shape[2], idx // self.shape[2]
        z, idx = idx % self.shape[1], idx // self.shape[1]

        data = self.data[idx, z:z+gridsize, y:y+gridsize, x:x+gridsize][None, :, :, :].astype(np.float32)
        label = self.label[idx, z, y, x].astype(np.int64)
        dist = self.dist[idx, z, y, x].astype(np.int64)
        index = np.array([idx, z, y, x], dtype=np.int32)


        if not self.augment or aug == 0: pass

        elif aug == 1:
            data = np.rot90(data, k=2, axes=(2, 3))
            dist[0], dist[1], dist[2] = dist[0], (gridpow2-dist[1])%gridpow2, (gridpow2-dist[2])%gridpow2
        elif aug == 2:
            data = np.rot90(data, k=2, axes=(1, 2))
            dist[0], dist[1], dist[2] = (gridpow2-dist[0])%gridpow2, (gridpow2-dist[1])%gridpow2, dist[2]
        elif aug == 3:
            data = np.rot90(data, k=2, axes=(1, 2))
            dist[0], dist[1], dist[2] = (gridpow2-dist[0])%gridpow2, (gridpow2-dist[1])%gridpow2, dist[2]
            data = np.rot90(data, k=2, axes=(2, 3))
            dist[0], dist[1], dist[2] = dist[0], (gridpow2-dist[1])%gridpow2, (gridpow2-dist[2])%gridpow2

        index[0] += aug * datasize

        return {'data':data.copy(), 'label':label, 'dist':dist, 'index':index, 'neg_index': neg_index, 'generate': generate}

    def __len__(self):
        return self.size

class gdrSampler_origin(Sampler):
    def __init__(self, shape, rank, restart=0.2):
        self.rank = rank
        print('#initializing gdr sampler ...')
        self.shape = list(shape)
        for i in range(1, len(shape)): self.shape[i] //= samplestep
        print('#shape:', *self.shape)
        self.restart = restart
        print('#restart:', self.restart)
        self.weight = np.zeros(self.shape, dtype=np.float32)
        self.weight[:, 1:-1, 1:-1, 1:-1] = 1
        self.weightmax = 14.14  # neg/pos
        self.state = np.zeros(self.shape, dtype=np.int16) - 1
        self.size = self.weight.size
        self.index = list(range(self.size))
        self.step = [0, 1, 2, 4, 8, 16, 0, -16, -8, -5, -2, -1]  # debug for step[gridpow]

    def __iter__(self):
        while True:
            bobsize = batchsize * world_size * 64
            batch = random.choices(self.index, weights=self.weight.reshape(-1), k=bobsize)
            for k in range(self.rank, bobsize, world_size):
                i = batch[k]
                if self.state.reshape(-1)[i] < 0 or random.random() < self.restart:
                    j = random.randrange(samplestep ** 3)
                else:
                    j = self.state.reshape(-1)[i]
                self.state.reshape(-1)[i] = -1

                x, i, j = i % self.shape[3] * samplestep + j % samplestep, i // self.shape[3], j // samplestep
                y, i, j = i % self.shape[2] * samplestep + j % samplestep, i // self.shape[2], j // samplestep
                z, i, j = i % self.shape[1] * samplestep + j % samplestep, i // self.shape[1], j // samplestep

                yield ((i * self.shape[1]*samplestep + z) * self.shape[2]*samplestep + y) * self.shape[3]*samplestep + x

    def __len__(self):
        return self.size

    def update(self, index, label, center):
        for first, second, third in zip(index, label, center):
            for (i, z, y, x), l, c in zip(first.tolist(), second.tolist(), third.tolist()):
                zz, yy, xx = z // samplestep, y // samplestep, x // samplestep
                if l == 0 or gridpow in c:
                    # weight tends to be 1 if the grid contains more backgrounds
                    self.weight[i, zz, yy, xx] = max(self.weight[i, zz, yy, xx] - 1, 1)
                else:
                    # weight tends to be max if the grid contains more signals
                    self.weight[i, zz, yy, xx] = min(self.weight[i, zz, yy, xx] + 1, self.weightmax)

                    z0 = min(max(z + self.step[c[0]], 0), self.shape[1]*samplestep - 1)
                    y0 = min(max(y + self.step[c[1]], 0), self.shape[2]*samplestep - 1)
                    x0 = min(max(x + self.step[c[2]], 0), self.shape[3]*samplestep - 1)
                    zz0, yy0, xx0 = z0 // samplestep, y0 // samplestep, x0 // samplestep

                    # weight tends to be max if the grid contains a center
                    self.weight[i, zz0, yy0, xx0] = min(self.weight[i, zz0, yy0, xx0] * 2, self.weightmax)

                    if z == z0 and y == y0 and x == x0: continue  # converge, then restart
                    if zz == zz0 and yy == yy0 and xx == xx0:  # same grid, then gradient descent
                        self.state[i, zz, yy, xx] = ((z0-zz*samplestep) * samplestep + y0-yy*samplestep) * samplestep + x0-xx*samplestep

class testSet_origin(Dataset):
    def __init__(self, data, label, dist):
        self.data = data
        self.label, self.dist = label, dist
        self.shape = list(self.data.shape)
        self.size = self.shape[0]
        for i in range(1, len(self.shape)):
            self.shape[i] -= 2 * gridhalf
            self.size *= self.shape[i]
        print('#shape:', *self.shape)
        print('#size:', self.size)

    def __getitem__(self, idx):
        x, idx = idx % self.shape[3], idx // self.shape[3]
        y, idx = idx % self.shape[2], idx // self.shape[2]
        z, idx = idx % self.shape[1], idx // self.shape[1]

        data = self.data[idx, None, z:z+gridsize, y:y+gridsize, x:x+gridsize].astype(np.float32)
        label = self.label[idx, z, y, x].astype(np.int64)
        dist = self.dist[idx, z, y, x].astype(np.int64)
        index = np.array([idx, z, y, x], dtype=np.int32)

        return {'data':data, 'label':label, 'dist':dist, 'index':index}

    def __len__(self):
        return self.size
