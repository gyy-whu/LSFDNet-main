import sys
import os
import torch
from torch import nn
from typing import List
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.io import imsave
import inspect
from scripts.guided_diffusion.script_util import create_model_and_diffusion
from basicsr.utils.registry import ARCH_REGISTRY



from timm.layers import trunc_normal_, DropPath
from scripts.cv2util import LayerNorm, GRN

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        outs.pop()
        return outs

Mdoel_size = {  
        "atto": {"depth": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},  
        "femto": {"depth": [2, 2, 6, 2], "dims": [48, 96, 192, 384]},  
        "pico": {"depth": [2, 2, 6, 2], "dims": [64, 128, 256, 512]},  
        "nano": {"depth": [2, 2, 8, 2], "dims": [80, 160, 320, 640]},  
        "tiny": {"depth": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},  
        "base": {"depth": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},  
        "large": {"depth": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},  
        "huge": {"depth": [3, 3, 27, 3], "dims": [352, 704, 1408, 2816]}, 

        "f_pico": {"depth": [2, 2, 6, 2], "dims": [64, 128, 256, 512]}
    } 

@ARCH_REGISTRY.register()
class FeatureExtractorConvNeXtV2(nn.Module):
    def __init__(self,size='f_pico', **kwargs):
        super().__init__()
        c2_size = Mdoel_size.get(size, "wrong c2 model")
        self.depths = c2_size.get("depth", "wrong depth input")
        self.dims = c2_size.get("dims", "wrong dims input")
        self.model = ConvNeXtV2(in_chans=2, depths=self.depths, dims=self.dims, **kwargs)

    def forward(self, x):
        x = self.model.forward(x)
        return x