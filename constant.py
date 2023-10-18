#!/usr/bin/env -S python3 -B

import torch as pt


devname, devcount = pt.cuda.get_device_name(0), pt.cuda.device_count()
if pt.cuda.get_device_name().endswith('RTX 2080 Ti'): batchsize = 1
elif pt.cuda.get_device_name().endswith('RTX 3090'): batchsize = 4
elif pt.cuda.get_device_name().endswith('V100-SXM2-32GB'): batchsize = 12
eval_batchsize = 64
print('#device:', devname)
print('#batchsize:', batchsize)
print('#eval_batchsize:', eval_batchsize)

dataset = 'data21'
# datasize, labelsize, zshift = 8, 16, 166
datasize, labelsize, zshift = 8, 14, 166

print('#dataset:', dataset)
print('#labelsize:', labelsize)

gridpow, gridpow2 = 6, 12
gridsize, gridhalf = 2 ** gridpow + 1, 2 ** (gridpow - 1)
print('#gridsize:', gridsize)

lr_init, lr_exp = 5e-3, 2e-4  # note
sched_chk, sched_cycle = 4, 32
epochlast = sched_cycle * 32
print('#scheduler:', sched_cycle, epochlast)
print(pt.cuda.get_device_name())

world_size = 4
print('world number:', world_size)
print()
