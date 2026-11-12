"""
任务 2: 简单 encoder-decoder baseline
任务 3: U-Net要画结构图并核对每层shape

两个模型输入 [B,3,H,W]，输出 logits [B,1,H,W] (不加sigmoid, 交给loss
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# 一定要理解好什么是上/下采样，才能知道为什么需要其他网络。并且理解encoder/decoder的关系。
class EncoderDecoder(nn.Module):
    """
    Baseline: 几层conv + 下采样，再几层上采样回到原尺寸。没有 skip connection。
    """

    # think about why the input and output channels are different, and why the input and output sizes are the same.
    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()

        # encoder   
        self.enc_conv0 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # H -> H/2
        self.enc_conv1 = nn.Conv2d(base, base, 3, padding= 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(base, base, 3, padding=1) 
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Conv2d(base, base, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(base, base, 3, padding=1)

        # decoder (upsampling)
        # scale_factor=2: 放大 2 倍，任何能被 16 整除的输入尺寸都能还原
        # (课上写的 nn.Upsample(16) 是放大"到" 16x16，只适用于 128 的输入)
        self.upsample0 = nn.Upsample(scale_factor=2)  # H/16 -> H/8
        self.dec_conv0 = nn.Conv2d(base, base, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)  # H/8 -> H/4
        self.dec_conv1 = nn.Conv2d(base, base, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)  # H/4 -> H/2
        self.dec_conv2 = nn.Conv2d(base, base, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)  # H/2 -> H
        self.dec_conv3 = nn.Conv2d(base, out_ch, 3, padding=1)
        
    def forward(self, x):
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3



class UNet(nn.Module):
    """
    Encoder 每层: 两次 conv3x3 -> maxpool；decoder: 上采样 -> 和对应 encoder 特征 concat -> 两次 conv。
    depth 次下采样 -> H, W 必须能被 2**depth 整除。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 64, depth: int = 4):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_model(name: str) -> nn.Module:
    """name: 'encdec' | 'unet'"""
    if name == "encdec":
        return EncoderDecoder()
    if name == "unet":
        return UNet()
    raise ValueError(f"unknown model: {name}")


if __name__ == "__main__":
    # 自检 + 任务 3 要求的 "核对每层 shape": 用 forward hook 打印每个模块的输出形状
    x = torch.randn(2, 3, 256, 256)
    for name in ["encdec", "unet"]:
        m = build_model(name)
        print(name, m(x).shape)  # 应为 [2, 1, 256, 256]
