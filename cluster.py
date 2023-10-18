#!/opt/anaconda3/bin/python3

import sys
import h5py
import numpy as np

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering


posprob = 0.5
clustdist = 9

idx2pdb = ['fiducial', '4V94', '4CR2', '1QVR', '1BXN', '3CF3', '1U6G', '3D2F', '2CG9', '3H84', '3GL1', '3QM1', '1S3X', '5MRC']


with h5py.File('/home/wangyaoyu/SHREC/output_adapt_neg_bin_coarser/data21_prob_95.h5py', 'r') as f:
    label = f['testlabel'][1:]
    dist = f['testdist'][1:]
hit = np.sum(label[0, :, :, :, 0] >= 0)
total = label[0, :, :, :, 0].size
print(hit)
print(total)
print('#hit: %.2f' % (64 * hit / total))
# input()

refavoid = []
with open('/home/wangyaoyu/shrec/output.submit/predict.avoid', 'r') as f:
    for l in f.readlines():
        _, x, y, z = l.split(' ')
        refavoid.append([int(z)-166, int(y), int(x)])
refavoid = np.array(refavoid, dtype=np.int32)


# not background by label
signal0 = (label[:, :, :, :, 0] >= 0) & (label[:, :, :, :, 0] < posprob)
# near center by dist
signal1 = np.all(np.sum(dist[:, :, :, :, :, [0, 1, 2, -2, -1]], axis=-1) > posprob, axis=-1)  # 153929
# signal1 = np.all(np.sum(dist[:, :, :, :, :, [0, 1, -1]], axis=-1) > posprob, axis=-1)  # 24274
# not background and near center
signal01 = signal0 & signal1
coord0 = np.array(np.where(signal01[0])).T
print(len(coord0))

clust = AgglomerativeClustering(distance_threshold=clustdist, n_clusters=None,
        affinity='euclidean', linkage='average').fit_predict(coord0)
print('#signal:', np.sum(signal0), np.sum(signal1), np.sum(signal01), np.max(clust)+1)

coord1 = []
for i in tqdm(range(np.max(clust)+1)):
    coord1.append(np.mean(coord0[clust == i], axis=0))
coord1 = np.array(coord1, dtype=np.int32)


# outfn = '/home/wangyaoyu/shrec/output.submit/submit_all.h5py'
# with h5py.File(outfn, 'w') as f:
#     f.create_dataset('particle', data=coord1)
#
# print('#loading dataset', outfn, '...')
# cache = h5py.File(outfn, 'r')
#
# coord1 = cache['particle'][()]
# cache.close()

print('#center[train]:', coord1.shape, coord1.dtype)

pair = np.sqrt(np.sum(np.square(coord1[:, None, :] - refavoid[None, :, :]), axis=-1))
coord2 = coord1[np.min(pair, axis=1) > clustdist]

print('fiducials ',len(coord2))

idx = 0
result, grid = [[] for i in range(len(idx2pdb))], clustdist // 2
for z, y, x in tqdm(coord2):
    z = min(max(z, grid), label.shape[1]-grid-1)
    y = min(max(y, grid), label.shape[2]-grid-1)
    x = min(max(x, grid), label.shape[3]-grid-1)

    l = label[0, z-grid:z+grid+1, y-grid:y+grid+1, x-grid:x+grid+1]
    l = np.mean(l[l[:, :, :, 0] >= 0], axis=0)

    i, s = np.argmax(l), np.max(l)
    if i > 0:
        result[i].append([idx2pdb[i], x, y, z+166, s])
    else:
        idx += 1
print(idx)



for i in range(1, len(idx2pdb)):
    result[i] = sorted(result[i], key=lambda k: -k[-1])
    # normal
    result[i] = [k for k in result[i]]

    ## print all particles
    result[i] = [k for k in result[i]]

for z, y, x in refavoid:
    result[0].append([idx2pdb[0], x, y, z+166, 0])


with open('/home/wangyaoyu/shrec/output.submit/predict_radius_probability.txt', 'w') as f:
    for i in result:
        print(i[0][0], len(i), i[-1][-1])
        for k in i:
            f.write('%s %d %d %d %f\n' % (k[0], k[1], k[2], k[3], k[4]))

