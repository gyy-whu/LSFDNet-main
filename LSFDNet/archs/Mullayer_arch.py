import torch
from torch import nn
import math
from typing import List
import cv2
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY

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
        self.size = 8
        self.dropout = 0.1
        self.channel = 4
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

        SWIR_scale_f_w_s = self_attention(SWIR_scale_f_w, self.SA_SWIR)
        LWIR_scale_f_w_s = self_attention(LWIR_scale_f_w, self.SA_LWIR)

        SWIR_scale_f_w_s_c = cross_attention(SWIR_scale_f_w_s, LWIR_scale_f_w_s, self.CA_SWIR)
        LWIR_scale_f_w_s_c = cross_attention(LWIR_scale_f_w_s, SWIR_scale_f_w_s, self.CA_LWIR)

        SWIR_scale_f_s_c = flod_win_l(SWIR_scale_f_w_s_c.permute(1, 2, 0))
        LWIR_scale_f_s_c = flod_win_s(LWIR_scale_f_w_s_c.permute(1, 2, 0))

        enhance_f = torch.cat([SWIR_scale_f_s_c, LWIR_scale_f_s_c], dim=1)
        enhance_f = self.Decoder_mul(enhance_f)

        return enhance_f
    
@ARCH_REGISTRY.register()
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