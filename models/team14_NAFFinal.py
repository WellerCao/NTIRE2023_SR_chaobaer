# -- coding: utf-8 --
# @Time : 2023/3/21 16:31
# @Author : cj
# @Email : weller1212@163.com
# @File : team14_NAFFinal.py
# @Function: xxx
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., dialation=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=dialation, stride=1, groups=dw_channel,
                               bias=True, dilation = dialation)
        # self.conv2 = GCLBlock(dw_channel, dw_channel)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0., dialation = 1):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate, dialation = dialation)

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        return feats

class FeatExtractBlk(nn.Module):
    def __init__(self,n_feat):
        super(FeatExtractBlk, self).__init__()

        self.body1 = NAFBlockSR(n_feat)
        self.body2 = NAFBlockSR(n_feat)
        self.body3 = NAFBlockSR(n_feat)

        self.combine = default_conv(n_feat * 3, n_feat*2, 1, )
        self.sg = SimpleGate()

    def forward(self, x):
        res1 = self.body1(x)[0]
        res2 = self.body2(x)[0]
        res3 = self.body3(x)[0]
        feats = torch.cat([res1, res2, res3], dim=1)
        res = x + self.sg(self.combine(feats))

        return res

class Embedding(nn.Module):
    def __init__(self,in_feat, out_feat):
        super(Embedding, self).__init__()

        self.conv3x3 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=2, dilation=2)
        self.conv = nn.Conv2d(out_feat*2+in_feat, out_feat*2, kernel_size=1)
        self.sg = SimpleGate()

    def forward(self, x):
        res = self.sg(
            self.conv(torch.cat([
                self.conv3x3(x), self.conv5x5(x), x
            ], dim=1))
        )

        return res

class UpScale(nn.Module):

    def __init__(self, up_scale=4, n_feats=64, n_outfeats=64, kernel_size=3):
        super().__init__()

        # define pixelshuffle module
        m_pixelshuffle = [
            default_conv(n_feats, int(n_feats * up_scale * up_scale/4), kernel_size),
            torch.nn.PixelShuffle(int(up_scale / 2)),
            default_conv(n_feats, int(n_feats * up_scale * up_scale / 4), kernel_size),
            torch.nn.PixelShuffle(int(up_scale / 2)),
        ]
        m_pixelshuffle.append(default_conv(n_feats, n_outfeats, kernel_size))
        self.pixelshuffle = nn.Sequential(*m_pixelshuffle)

    def forward(self, x):
        feats = self.pixelshuffle(x)
        return feats


class Final(nn.Module):
    def __init__(self):
        super(Final, self).__init__()

        self.upscale_x = 4  # 上采样倍率
        self.layers = 11
        assert self.layers%2 == 1
        n_feats = 64  # 主conv的宽度
        kernel_size = 3

        # define head module
        self.head = Embedding(6, n_feats)

        # define encoder module, 3层的下采样模型
        self.body = nn.ModuleList([
            FeatExtractBlk(n_feats) for _ in range(self.layers)
        ])

        self.tail = UpScale(self.upscale_x, n_feats, 3, kernel_size)

        # sobelx sobely lap
        bias = torch.randn(1) * 1e-3
        bias = torch.reshape(bias, (1,))

        self.sobelx_mask = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        self.sobelx_bias = nn.Parameter(torch.FloatTensor(bias))
        for i in range(3):
            self.sobelx_mask[0, i, 0, 0] = 1.0
            self.sobelx_mask[0, i, 1, 0] = 2.0
            self.sobelx_mask[0, i, 2, 0] = 1.0
            self.sobelx_mask[0, i, 0, 2] = -1.0
            self.sobelx_mask[0, i, 1, 2] = -2.0
            self.sobelx_mask[0, i, 2, 2] = -1.0
        self.sobelx_mask = nn.Parameter(data=self.sobelx_mask, requires_grad=False)

        self.sobely_mask = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        self.sobely_bias = nn.Parameter(torch.FloatTensor(bias))
        for i in range(3):
            self.sobely_mask[0, i, 0, 0] = 1.0
            self.sobely_mask[0, i, 0, 1] = 2.0
            self.sobely_mask[0, i, 0, 2] = 1.0
            self.sobely_mask[0, i, 2, 0] = -1.0
            self.sobely_mask[0, i, 2, 1] = -2.0
            self.sobely_mask[0, i, 2, 2] = -1.0
        self.sobely_mask = nn.Parameter(data=self.sobely_mask, requires_grad=False)

        self.lap_mask = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        self.lap_bias = nn.Parameter(torch.FloatTensor(bias))
        for i in range(3):
            self.lap_mask[0, i, 0, 1] = 1.0
            self.lap_mask[0, i, 1, 0] = 1.0
            self.lap_mask[0, i, 1, 2] = 1.0
            self.lap_mask[0, i, 2, 1] = 1.0
            self.lap_mask[0, i, 1, 1] = -4.0
        self.lap_mask = nn.Parameter(data=self.lap_mask, requires_grad=False)

    def forward(self, x):
        B, C, H, W = x.shape

        x_hr = F.interpolate(x[:, :3, :, :], scale_factor=self.upscale_x, mode='bilinear')
        x = self.head(x)

        inter_rets = []
        inter_idx = int(self.layers//2)
        for i in range(self.layers):
            if i<=inter_idx:
                x = self.body[i](x)
                inter_rets.append(x)
            else:
                x = self.body[i](x+inter_rets[2*inter_idx-i])

        residual3 = self.tail(x)

        ret = residual3 + x_hr

        return torch.clamp(ret, 0., 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = math.ceil(h / 2 ** self.layers) * 2 ** self.layers - h
        mod_pad_w = math.ceil(w / 2 ** self.layers) * 2 ** self.layers - w
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

if __name__ == '__main__':
    batch_size = 1
    img_height = 64
    img_width = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    input = torch.rand(batch_size, 6, img_height, img_width).to(device)
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    model = Final()

    # path = '../models/Final_2_epoch_340.pth'
    # state_dict = model.state_dict()
    # for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
    #     if n in state_dict.keys():
    #         state_dict[n].copy_(p)
    #     else:
    #         raise KeyError(n)

    model = model.to(device)
    output = model(input)
    print(f"output shapes: {[t.shape for t in output]}")

    # from ptflops import get_model_complexity_info
    # inp_shape = (6,256,256)
    # macs, params = get_model_complexity_info(Final(), inp_shape, verbose=False, print_per_layer_stat=False)
    #
    # params = float(params[:-3])
    # macs = float(macs[:-4])
    #
    # print(macs, params)



