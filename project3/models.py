"""
任务 2: 简单 encoder-decoder baseline。
任务 3: U-Net (要画结构图并核对每层 shape)。

两个模型输入 [B,3,H,W]，输出 logits [B,1,H,W] (不加 sigmoid，交给 loss)。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    Baseline: 几层 conv + 下采样，再几层上采样回到原尺寸。没有 skip connection。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


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
