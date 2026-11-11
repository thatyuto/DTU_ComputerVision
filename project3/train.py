#!/usr/bin/env python3
"""
训练一个分割模型，按验证集 Dice 保存 best.pt。

Debug (少量 batch):
  python train.py --dataset ph2 --model unet --epochs 1 --max-batches 2 --out-dir /tmp/smoke
Full run:
  python train.py --dataset drive --model unet --loss focal --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from losses import build_loss, compute_pos_weight
from metrics import all_metrics
from models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ph2", "drive"], required=True)
    p.add_argument("--model", choices=["encdec", "unet"], default="unet")
    p.add_argument("--loss", choices=["bce", "focal", "weighted_bce"], default="bce")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)  # DRIVE 大图时要调小
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="debug 用")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None) -> dict:
    """
    optimizer 为 None -> 评估模式 (no_grad)。
    返回 {"loss": ..., "dice": ..., ...}。
    提示: 指标要在整个 split 上累加 TP/FP/FN/TN 再算，不要对每个 batch 的 Dice 取平均。
    DRIVE 的 batch 是 (image, mask, fov)，PH2 是 (image, mask)。
    """
    # TODO
    raise NotImplementedError


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or f"runs/{args.dataset}_{args.model}_{args.loss}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # TODO:
    # 1. build_dataset(train/val) -> DataLoader (train shuffle=True)
    # 2. build_model -> to(device)
    # 3. pos_weight = compute_pos_weight(train_ds) if args.loss == "weighted_bce"
    #    criterion = build_loss(args.loss, pos_weight).to(device)
    # 4. optimizer = Adam
    # 5. 每个 epoch: run_epoch(train) -> run_epoch(val) -> 记录 history
    #    val dice 最好时 torch.save({"model": ..., "args": vars(args)}, out_dir / "best.pt")
    # 6. 写 history.json
    raise NotImplementedError


if __name__ == "__main__":
    main()
