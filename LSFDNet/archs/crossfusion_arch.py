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

@ARCH_REGISTRY.register()
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
        self.size = 4
        self.dropout = 0.1
        self.channel = 4
        self.SA_SWIR, self.SA_LWIR, self.CA_SWIR, self.CA_LWIR = [ nn.MultiheadAttention(self.size * self.size * self.channel, 1, self.dropout) for _ in range(4)]  

    def self_attention(self, input_s, MHA):
        """
        :param MHA:
        :param input_s: [batch_size, patch_nums, patch_size * patch_size * channel]
        :return:
        """
        embeding_dim = input_s.size()[2]
        if MHA.training:
            PE = PositionalEncoding(embeding_dim, self.dropout).train()
        else:
            PE = PositionalEncoding(embeding_dim, self.dropout).eval()
        input_pe = PE(input_s)

        q = input_pe
        k = input_pe
        v = input_s
        a = MHA(q, k, v)[0]
        enhance = a + input_s
        return enhance

    def cross_attention(self, query, key_value, MHA):
        """
        :param MHA:
        :param query:
        :param key_value:
        :return:
        """
        embeding_dim = query.size()[2]
        if MHA.training:
            PE = PositionalEncoding(embeding_dim, self.dropout).train()
        else:
            PE = PositionalEncoding(embeding_dim, self.dropout).eval()

        q_pe = PE(query)
        kv_pe = PE(key_value)

        q = q_pe
        k = kv_pe
        v = key_value
        a = MHA(q, k, v)[0]
        enhance = a + query
        return enhance

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

        SWIR_scale_f_w_s = self.self_attention(SWIR_scale_f_w, self.SA_SWIR)
        LWIR_scale_f_w_s = self.self_attention(LWIR_scale_f_w, self.SA_LWIR)

        SWIR_scale_f_w_s_c = self.cross_attention(SWIR_scale_f_w_s, LWIR_scale_f_w_s, self.CA_SWIR)
        LWIR_scale_f_w_s_c = self.cross_attention(LWIR_scale_f_w_s, SWIR_scale_f_w_s, self.CA_LWIR)

        SWIR_scale_f_s_c = flod_win_l(SWIR_scale_f_w_s_c.permute(1, 2, 0))
        LWIR_scale_f_s_c = flod_win_s(LWIR_scale_f_w_s_c.permute(1, 2, 0))

        enhance_f = torch.cat([SWIR_scale_f_s_c, LWIR_scale_f_s_c], dim=1)
        enhance_f = self.Decoder(enhance_f)

        return enhance_f