#!/usr/bin/env python3
"""Evaluate a saved checkpoint on val or test (video-level accuracy)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets import DEFAULT_DATA_ROOT, DualVideoDataset, VideoFrameDataset, default_transform
from models import build_model, needs_flow
from train import run_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--ckpt", required=True, help="Path to runs/<model>/best.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    name = ckpt.get("model", Path(args.ckpt).parent.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    use_flow = needs_flow(name)
    if use_flow:
        data = DualVideoDataset(args.data_root, args.split, image_size=args.image_size)
    else:
        data = VideoFrameDataset(
            args.data_root,
            args.split,
            transform=default_transform(False, args.image_size),
            stack_frames=True,
        )
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    loss, acc = run_epoch(
        model,
        loader,
        nn.CrossEntropyLoss(),
        None,
        device,
        use_flow,
        args.max_batches,
        train=False,
    )
    print(f"{args.split}  loss={loss:.4f}  acc={acc:.3f}  n={len(data)}  model={name}")


if __name__ == "__main__":
    main()
