#!/usr/bin/env -S python3 -B

import h5py
import numpy as np
import torch as pt

from torch import nn
from torch.autograd import Variable

from constant import *


dimgroup, dimwrap, dimhead = 16, 32, 64


class ReZero(nn.Module):
    def __init__(self, cio, dim=1):
        super(ReZero, self).__init__()
        self.dim = dim
        self.weight0 = nn.parameter.Parameter(pt.zeros([cio//dimgroup, 1]), requires_grad=True)

    def forward(self, x):
        return (x.reshape(*x.shape[:self.dim], x.shape[self.dim]//dimgroup, -1) * self.weight0).reshape(x.shape)

class FeatureBlock(nn.Module):
    def __init__(self, cout, size):
        super(FeatureBlock, self).__init__()
        chid = min(max(cout // 4, dimwrap), dimhead)
        print('#Feature:', size, 1, chid, cout)

        layers = [nn.LayerNorm([size, size, size], elementwise_affine=True),
                  nn.Conv3d(1, chid, kernel_size=3, padding=1),
                  nn.GroupNorm(chid//dimgroup, chid), nn.GELU(),
                  nn.Conv3d(chid, cout, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class PoolBlock(nn.Module):
    def __init__(self, cin, cout, size):
        super(PoolBlock, self).__init__()
        chid = min(max(cin // 4, dimwrap), dimhead)
        nhead = cin // chid
        print('#Pool:', size//2+1, cin, chid, nhead, cout)

        layers = [nn.GroupNorm(cin//dimgroup, cin), nn.GELU(),
                  nn.Conv3d(cin, cout, kernel_size=3, padding=1, stride=2, groups=nhead),
                  nn.LayerNorm([size//2+1, size//2+1, size//2+1], elementwise_affine=True), nn.GELU(),
                  nn.Conv3d(cout, cout, kernel_size=1, padding=0, stride=1, groups=1)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResConvBlock(nn.Module):
    def __init__(self, cio, size):
        super(ResConvBlock, self).__init__()
        chid = min(max(cio // 4, dimwrap), dimhead)
        nhead = cio // chid
        print('#ResConv:', size, cio, chid, nhead)

        layers = [nn.GroupNorm(cio//dimgroup, cio), nn.GELU(),
                  nn.Conv3d(cio, cio, kernel_size=3, padding=1, groups=nhead),
                  nn.GroupNorm(cio//dimgroup, cio), nn.GELU(),
                  nn.Conv3d(cio, cio, kernel_size=1, padding=0, groups=1),
                  ReZero(cio)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

class ResMixerBlock(nn.Module): #(dhead*31, 5**3)
    def __init__(self, cio1, cio2, dhead=dimhead, nhead=16, depth=64):
        super(ResMixerBlock, self).__init__()
        print('#ResMixer:', cio1//dimgroup, dimgroup, cio2, dhead, nhead, depth)
        self.depth = depth

        layers = [nn.GroupNorm(dimgroup, cio2*dimgroup), nn.GELU(),
                  nn.Conv1d(cio2*dimgroup, dhead*nhead, kernel_size=1, groups=1),
                  nn.GroupNorm(dhead*nhead//dimgroup, dhead*nhead), nn.GELU(),
                  nn.Conv1d(dhead*nhead, dhead*nhead, kernel_size=1, groups=nhead),
                  nn.GroupNorm(dhead*nhead//dimgroup, dhead*nhead), nn.GELU(),
                  nn.Conv1d(dhead*nhead, dhead*nhead, kernel_size=1, groups=nhead),
                  nn.GroupNorm(dhead*nhead//dimgroup, dhead*nhead), nn.GELU(),
                  nn.Conv1d(dhead*nhead, cio2, kernel_size=1, groups=1),
                  nn.LayerNorm([cio2, 1], elementwise_affine=True)]
        self.gate = nn.Sequential(*layers)

        layers = [nn.GroupNorm(cio1//dimgroup, cio1), nn.GELU(),
                  nn.Conv1d(cio1, dhead*nhead, kernel_size=1, groups=1),
                  nn.GroupNorm(dhead*nhead//dimgroup, dhead*nhead), nn.GELU(),
                  nn.Conv1d(dhead*nhead, dhead*nhead, kernel_size=1, groups=nhead),
                  nn.GroupNorm(dhead*nhead//dimgroup, dhead*nhead), nn.GELU(),
                  nn.Conv1d(dhead*nhead, dhead*nhead, kernel_size=1, groups=nhead),
                  nn.GroupNorm(dhead*nhead//dimgroup, dhead*nhead), nn.GELU(),
                  nn.Conv1d(dhead*nhead, cio1, kernel_size=1, groups=1),
                  ReZero(cio1)]
        self.value = nn.Sequential(*layers)

    def forward(self, x):
        xx = x
        for i in range(self.depth):
            g = pt.exp( self.gate(xx.reshape(-1, dimgroup*x.shape[2], 1)) ).reshape(x.shape[0], -1, 1, x.shape[2])
            v = self.value(xx).reshape(x.shape[0], -1, dimgroup, x.shape[2])
            xx = xx + (g * v).reshape(x.shape)
        return xx

class CentralResNet3D(nn.Module):
    def __init__(self, dhead=dimhead):
        super(CentralResNet3D, self).__init__()
        self.dhead = dhead

        # high-resolution reconstructions are highly noisy with weak signals
        # central feature network addresses different particle sizes
        self.conv65 = nn.Sequential(FeatureBlock(dhead*1, 65),
                                    ResConvBlock(dhead*1, 65))  # 65x64
        self.conv33 = nn.Sequential(PoolBlock(dhead*1, dhead*2, 65),
                                    ResConvBlock(dhead*2, 33))  # 33x128
        self.conv17 = nn.Sequential(PoolBlock(dhead*2, dhead*4, 33),
                                    ResConvBlock(dhead*4, 17))  # 17x256
        self.conv9  = nn.Sequential(PoolBlock(dhead*4, dhead*8, 17),
                                    ResConvBlock(dhead*8, 9))  # 9x512
        self.conv5  = nn.Sequential(PoolBlock(dhead*8, dhead*16, 9),
                                    ResConvBlock(dhead*16, 5))  # 5x1024

        self.mixer  = ResMixerBlock(dhead*31, 5**3)  # 5x1984

        self.embed0 = nn.Sequential(nn.GroupNorm(dhead*31//dimgroup, dhead*31), nn.GELU(),
                                    nn.Linear(dhead*31, dhead*15),
                                    nn.LayerNorm(dhead*15, elementwise_affine=False))  # norm
        self.embed1 = nn.Sequential(nn.GroupNorm(dhead*31//dimgroup, dhead*31), nn.GELU(),
                                    nn.Linear(dhead*31, dhead*15),
                                    nn.LayerNorm(dhead*15, elementwise_affine=False))  # norm

        self.head0  = nn.Linear(dhead*15, labelsize)
        self.head1  = nn.Conv1d(dhead*5, gridpow2, kernel_size=1, groups=1)

    def forward(self, x):
        xx65 = self.conv65(x)
        xx33 = self.conv33(xx65)
        xx17 = self.conv17(xx33)
        xx9  = self.conv9(xx17)
        xx5  = self.conv5(xx9)

        xx65 = xx65[:, :, 30:35, 30:35, 30:35].contiguous().reshape(xx65.shape[0], xx65.shape[1], -1)
        xx33 = xx33[:, :, 14:19, 14:19, 14:19].contiguous().reshape(xx33.shape[0], xx33.shape[1], -1)
        xx17 = xx17[:, :, 6:11, 6:11, 6:11].contiguous().reshape(xx17.shape[0], xx17.shape[1], -1)
        xx9  = xx9[:, :, 2:7, 2:7, 2:7].contiguous().reshape(xx9.shape[0], xx9.shape[1], -1)
        xx5  = xx5.reshape(xx5.shape[0], xx5.shape[1], -1)

        xx   = pt.cat([xx5, xx9, xx17, xx33, xx65], dim=1)
        xx   = self.mixer(xx)[:, :, 5**3//2]

        x0   = self.head0(self.embed0(xx))
        x1   = self.head1(self.embed1(xx).reshape(-1, self.dhead*5, 3)).permute(0, 2, 1)

        return xx, x0, x1

