#!/usr/bin/env python3
"""
加载 best.pt，在 train/val/test 三个 split 上报告五个指标 (任务 3 要求三个都报)，
并保存几张 "原图 | 真实 mask | 预测 mask" 对比图给 report 用。

  python eval.py --ckpt runs/drive_unet_bce/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from losses import build_loss
from models import build_model
from train import run_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--n-vis", type=int, default=4, help="保存几张对比图")
    return p.parse_args()


def save_predictions(model, dataset, device, out_path: Path, n: int):
    """matplotlib 画 n 行: image | ground truth | prediction。"""
    # TODO
    raise NotImplementedError


def main():
    args = parse_args()
    # TODO:
    # 1. ckpt = torch.load(args.ckpt)；用 ckpt["args"] 里的 dataset / model / loss 重建
    # 2. 对每个 split: run_epoch(..., optimizer=None) -> 指标
    # 3. 打印成表格，写 metrics.json 到 ckpt 同目录
    # 4. save_predictions(test split)
    raise NotImplementedError


if __name__ == "__main__":
    main()
