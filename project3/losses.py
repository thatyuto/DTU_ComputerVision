"""
任务 4: loss ablation。三种 loss 都接收 (logits [B,1,H,W], target [B,1,H,W] 0/1)。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    gamma=0 时退化成普通 BCE，可以用这个来检查实现对不对。
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        # 提示: 用 F.binary_cross_entropy_with_logits(..., reduction="none") 得到逐像素 -log(p_t)
        raise NotImplementedError


def compute_pos_weight(train_dataset) -> float:
    """背景像素数 / 前景像素数，只在训练集上统计。给 weighted BCE 用。"""
    # TODO
    raise NotImplementedError


def build_loss(name: str, pos_weight: float | None = None) -> nn.Module:
    """name: 'bce' | 'focal' | 'weighted_bce'"""
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return FocalLoss()
    if name == "weighted_bce":
        assert pos_weight is not None, "weighted_bce 需要 pos_weight"
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    raise ValueError(f"unknown loss: {name}")
