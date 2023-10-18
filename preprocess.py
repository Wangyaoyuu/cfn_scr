#!/usr/bin/env -S python3 -B

import h5py
import mrcfile
import numpy as np
from glob import glob
from itertools import product

from constant import *


def pad(matrix):
    shape = list(matrix.shape)
    for i in range(1, len(shape)): shape[i] += 2 * gridhalf
    result = np.zeros(shape, dtype=matrix.dtype)
    result[:, gridhalf:-gridhalf, gridhalf:-gridhalf, gridhalf:-gridhalf] = matrix
    return result


if __name__ == '__main__':
    pdb2idx = {}
    with open('/home/wangyaoyu/shrec/data/' + dataset + '.txt', 'r') as f:
        for l in f.readlines():
            k, v = l.split()
            pdb2idx[k] = int(v)

    data = []
    for fn in sorted(glob('/home/wangyaoyu/shrec/' + dataset + '/model_*/reconstruction.mrc')):
        print('#working on', fn, '...')
        with mrcfile.open(fn, permissive=True) as mrc:
            if len(mrc.data) == 512: data.append(mrc.data[zshift:-zshift, :, :])
            else: data.append(mrc.data)
    data = np.array(data, dtype=np.float16)

    label = []
    for fn in sorted(glob('/home/wangyaoyu/shrec/' + dataset + '/model_*/class_mask.mrc')):
        print('#working on', fn, '...')
        with mrcfile.open(fn, permissive=True) as mrc:
            if len(mrc.data) == 512: label.append(mrc.data[zshift:-zshift, :, :])
            else: label.append(mrc.data)
    label = np.array(label, dtype=np.int8)
    label[label >= labelsize] = 0

    mask = []
    for fn in sorted(glob('/home/wangyaoyu/shrec/' + dataset + '/model_*/occupancy_mask.mrc')):
        print('#working on', fn, '...')
        with mrcfile.open(fn, permissive=True) as mrc:
            if len(mrc.data) == 512: mask.append(mrc.data[zshift:-zshift, :, :])
            else: mask.append(mrc.data)
    mask = np.array(mask, dtype=np.int16)

    dist, center = np.ones([*label.shape, 3], dtype=np.int8) * gridpow, []
    r1, r2 = 1, 3
    dist0 = np.empty([r2, r2, r2, 3], dtype=np.int8)
    for i, j, k in product(range(r2), repeat=3): dist0[i, j, k] = [i-r1, j-r1, k-r1]
    for i, fn in enumerate(sorted(glob('/home/wangyaoyu/shrec/' + dataset + '/model_*/particle_locations.txt'))):
        print('#working on', fn, '...')
        with open(fn, 'r') as f:
            for j, l in enumerate(f.readlines()):
                pdb, x, y, z, z0, x0, z1 = l.split()
                if pdb2idx[pdb] == 0: continue

                idx0, idx1, idx2 = np.where(mask[i] == j+1)
                c1 = np.array(list(zip(idx0, idx1, idx2)), dtype=np.int16)

                x, y, z = int(x), int(y), int(z)
                if dataset == 'data21': z -= zshift
                c0 = np.array([z, y, x], dtype=np.int16)
                assert(np.max(label[i, z-r1:z+r1+1, y-r1:y+r1+1, x-r1:x+r1+1]) == pdb2idx[pdb])
                center.append([i, c0[0], c0[1], c0[2]])

                c10 = c1 - c0
                dist[i, idx0, idx1, idx2] = np.ceil(np.log2(np.abs(c10) + 1)) * ((c10 >= 0) * 2 - 1)

                # force center labels and distances
                mask[i, z-r1:z+r1+1, y-r1:y+r1+1, x-r1:x+r1+1] = j + 1
                label[i, z-r1:z+r1+1, y-r1:y+r1+1, x-r1:x+r1+1] = pdb2idx[pdb]
                dist[i, z-r1:z+r1+1, y-r1:y+r1+1, x-r1:x+r1+1] = dist0
    dist[dist<0] += gridpow2
    center = np.array(center, dtype=np.int16)


    data, mask = pad(data), pad(mask)
    print(data.dtype, label.dtype, dist.dtype, center.dtype)
    print(data.shape, label.shape, dist.shape. center.shape)
    with h5py.File('/home/wangyaoyu/shrec/data/' + dataset + '.hdf5', 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('mask', data=mask)
        f.create_dataset('label', data=label)
        f.create_dataset('dist', data=dist)
        f.create_dataset('center', data=center)

