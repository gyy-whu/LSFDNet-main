import torch
from torch import nn
import math
from typing import List
import cv2
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY
from timm.layers import trunc_normal_, DropPath
from scripts.cv2util import LayerNorm, GRN

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

class FeatureExtractorBase(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureExtractorBase, self).__init__()
        self.Base = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.Base(x)
        return out
    
class FeatureExtractorMulNet(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureExtractorMulNet, self).__init__()
        self.Mul_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # self.det_ft = ConvNeXtV2(in_chans=1, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)

    def forward(self, x):
        out_1 = self.Mul_1(x)
        return out_1
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        device = x.device
        x = x + self.pe[:, :x.size(1)].to(device)
        return self.dropout(x)
    
def self_attention(input_s, MHA):
    embeding_dim = input_s.size()[2]
    if MHA.training:
        PE = PositionalEncoding(embeding_dim, 0.1).train()
    else:
        PE = PositionalEncoding(embeding_dim, 0.1).eval()
    input_pe = PE(input_s)

    q = input_pe
    k = input_pe
    v = input_s
    a = MHA(q, k, v)[0]
    enhance = a + input_s
    return enhance

def cross_attention(query, key_value, MHA):
    embeding_dim = query.size()[2]
    if MHA.training:
        PE = PositionalEncoding(embeding_dim, 0.1).train()
    else:
        PE = PositionalEncoding(embeding_dim, 0.1).eval()

    q_pe = PE(query)
    kv_pe = PE(key_value)

    q = q_pe
    k = kv_pe
    v = key_value
    a = MHA(q, k, v)[0]
    enhance = a + query
    return enhance
    
class crossFusion(nn.Module):
    def __init__(self):
        super(crossFusion, self).__init__()
        self.size = 4
        self.dropout = 0.1
        self.channel = 4
        self.dim = self.size * self.size * self.channel
        self.SWIREncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.LWIREncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.LWIREncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.mlp_SA_SWIR = nn.Sequential(  
            nn.Linear(self.dim,self.dim*2),  
            nn.ReLU(),  
            nn.Linear(self.dim*2, self.dim)  
        )  
        self.mlp_SA_LWIR = nn.Sequential(  
            nn.Linear(self.dim, self.dim*2),  
            nn.ReLU(),  
            nn.Linear(self.dim*2, self.dim)  
        )  
        self.mlp_CA_SWIR = nn.Sequential(  
            nn.Linear(self.dim, self.dim*2),  
            nn.ReLU(),  
            nn.Linear(self.dim*2, self.dim)  
        )  
        self.mlp_CA_LWIR = nn.Sequential(  
            nn.Linear(self.dim, self.dim*2),  
            nn.ReLU(),  
            nn.Linear(self.dim*2, self.dim)  
        )  
        self.Decoder_mul = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU()
        )
        self.SA_SWIR, self.SA_LWIR, self.CA_SWIR, self.CA_LWIR = [ nn.MultiheadAttention(self.size * self.size * self.channel, 1, self.dropout) for _ in range(4)]  

    def forward(self, feats_SW, feats_LW):
        HW_size = feats_SW.size()[3]
        flod_win_s = nn.Fold(output_size=(HW_size, HW_size), kernel_size=(self.size, self.size),
                             stride=self.size)
        flod_win_l = nn.Fold(output_size=(HW_size, HW_size), kernel_size=(self.size, self.size),
                             stride=self.size)
        
        unflod_win_s = nn.Unfold(kernel_size=(self.size, self.size), stride=self.size)
        unflod_win_l = nn.Unfold(kernel_size=(self.size, self.size), stride=self.size)

        SWIR_scale_f = self.SWIREncoder(feats_SW)
        LWIR_scale_f = self.LWIREncoder(feats_LW)

        SWIR_scale_f_w = unflod_win_s(SWIR_scale_f).permute(2, 0, 1)
        LWIR_scale_f_w = unflod_win_l(LWIR_scale_f).permute(2, 0, 1)

        SWIR_scale_f_w_s = self.mlp_SA_SWIR(self_attention(SWIR_scale_f_w, self.SA_SWIR))
        LWIR_scale_f_w_s = self.mlp_SA_LWIR(self_attention(LWIR_scale_f_w, self.SA_LWIR))

        SWIR_scale_f_w_s_c = self.mlp_CA_SWIR(cross_attention(SWIR_scale_f_w_s, LWIR_scale_f_w_s, self.CA_SWIR))
        LWIR_scale_f_w_s_c = self.mlp_CA_LWIR(cross_attention(LWIR_scale_f_w_s, SWIR_scale_f_w_s, self.CA_LWIR))

        SWIR_scale_f_s_c = flod_win_l(SWIR_scale_f_w_s_c.permute(1, 2, 0))
        LWIR_scale_f_s_c = flod_win_s(LWIR_scale_f_w_s_c.permute(1, 2, 0))

        enhance_f = torch.cat([SWIR_scale_f_s_c, LWIR_scale_f_s_c], dim=1)
        enhance_f = self.Decoder_mul(enhance_f)

        return enhance_f

class MulLayerFusion(nn.Module):
    def __init__(self):
        super(MulLayerFusion, self).__init__()
        self.CMAB_H = crossFusion()
        self.CMAB_L = crossFusion()
        self.CMAB_F = crossFusion()
        self.downsample_layer_SW = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        self.downsample_layer_LW = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )     
        self.upsample_layer = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )  
        self.Decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=1)
        )

    def forward(self, feats_SW, feats_LW):
        fuse_h = self.CMAB_H(feats_SW, feats_LW)
        down_fS = self.downsample_layer_SW(feats_SW)
        down_fL = self.downsample_layer_LW(feats_LW)
        fuse_l = self.CMAB_L(down_fS, down_fL)
        fuse_l_up = self.upsample_layer(fuse_l)
        feats_mul = self.CMAB_F(fuse_h,fuse_l_up)
        out = self.Decoder(feats_mul)
        
        return out
    
@ARCH_REGISTRY.register()
class MMFusion(nn.Module):
    def __init__(self):
        super(MMFusion, self).__init__()
        self.netfe_base = FeatureExtractorBase()
        self.netfe_SW =  FeatureExtractorMulNet()
        self.netfe_LW =  FeatureExtractorMulNet()
        self.netMF_mulLayer =  MulLayerFusion()
    
    def forward(self, data):
        feats_SW_base = self.netfe_base(data['SW'])
        feats_LW_base = self.netfe_base(data['LW'])
        feats_SW = self.netfe_SW(feats_SW_base)
        feats_LW = self.netfe_LW(feats_LW_base)
        pred_img = self.netMF_mulLayer(feats_SW, feats_LW)

        return pred_img
