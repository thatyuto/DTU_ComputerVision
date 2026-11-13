"""
任务 4: loss ablation。三种 loss 都接收 (logits [B,1,H,W], target [B,1,H,W] 0/1)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    gamma=0 时退化成普通 BCE, 可以用这个来检查实现对不对。
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        # 提示: 用 F.binary_cross_entropy_with_logits(..., reduction="none") 得到逐像素 -log(p_t)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # [B,1,H,W]

        # 从bce算出p_t: we know that bce = -log(p_t) -> p_t = exp(-bce)
        p_t = torch.exp(-bce)  # [B,1,H,W]，p_t = p if y=1 else 1-p

        # mutiply
        loss = (1 - p_t) ** self.gamma * bce  # [B,1,H,W]

        return loss.mean()

    


def compute_pos_weight(train_dataset) -> float:
    """背景像素数 / 前景像素数(在这里就是血管/病变像素数)，只在训练集上统计。给 weighted BCE 用。

    dataset每个样本: PH2是(image, mask), DRIVE是(image, mask, fov)。
    DRIVE只数FOV圆内的像素, 否则黑边会让背景数虚高、pos_weight 偏大。
    """
    n_fg = 0.0  # 前景 (血管/病变) 像素数
    n_bg = 0.0  # 背景像素数

    # Fov是Field of View(视野,这里只识别有血管的区域)
    # Mask是血管的二值图
    # sample = (image, mask, fov)
    for sample in train_dataset:
        mask = sample[1].float()
        # 有 FOV 就用它，没有就全部像素都算(和 metrics.confusion 的 valid 一样)
        if len(sample) > 2:
            valid =  sample[2].float()
        else:
            valid = torch.ones_like(mask)

        n_fg += (mask * valid).sum().item()
        n_bg += ((1 - mask) * valid).sum().item()
    return n_bg / max(n_fg, 1.0)  # max 防止除以 0


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
