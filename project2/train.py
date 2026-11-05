#!/usr/bin/env python3
"""

Debug on an interactive node (few batches):
  python train.py --data-root /path/to/ucf --model per_frame --epochs 1 --max-batches 2

Full run on HPC batch:
  python train.py --data-root /path/to/ucf --model late --epochs 20
"""

from __future__ import annotations


"""
argparse 负责读命令行参数, json 负责把每轮的训练记录写成文件
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    DEFAULT_DATA_ROOT,
    DualVideoDataset,
    FrameImageDataset,
    VideoFrameDataset,
    default_transform,
)
from models import build_model, needs_flow, needs_video_clip


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_batch(batch, flow: bool):
    if flow:
        rgb, fl, y = batch
        return rgb, fl, y
    x, y = batch
    return x, None, y


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_flow: bool,
    max_batches: int,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for i, batch in enumerate(tqdm(loader, disable=not train, leave=False)):
            if max_batches and i >= max_batches:
                break
            x, fl, y = unpack_batch(batch, use_flow)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if fl is not None:
                fl = fl.to(device, non_blocking=True)
                logits = model(x, fl)
            else:
                logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n += bs
    if n == 0:
        raise RuntimeError("Loader was empty. Check --data-root and CSV paths.")
    return total_loss / n, total_acc / n


def build_datasets(args):
    if needs_flow(args.model):
        return (
            DualVideoDataset(args.data_root, "train", image_size=args.image_size),
            DualVideoDataset(args.data_root, "val", image_size=args.image_size),
        )
    train_tf = default_transform(train=True, image_size=args.image_size)
    eval_tf = default_transform(train=False, image_size=args.image_size)
    if needs_video_clip(args.model):
        train_set = VideoFrameDataset(
            args.data_root, "train", transform=train_tf, stack_frames=True
        )
    else:
        train_set = FrameImageDataset(args.data_root, "train", transform=train_tf)
    val_set = VideoFrameDataset(
        args.data_root, "val", transform=eval_tf, stack_frames=True
    )
    return train_set, val_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 2 video classification")
    p.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="ufc10 root (metadata/ + frames/). Default: course_cv/data/ufc10",
    )
    p.add_argument(
        "--model",
        default="per_frame",
        choices=["per_frame", "late", "early", "3d", "dual"],
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--max-batches", type=int, default=0, help="0 = full split; use 2 to debug")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="", help="Default: runs/<model>")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={args.model}  data={args.data_root}")

    out_dir = Path(args.out_dir or f"runs/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_datasets(args)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"train samples={len(train_set)}  val videos={len(val_set)}")

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    best_acc = -1.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            needs_flow(args.model),
            args.max_batches,
            train=True,
        )
        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            needs_flow(args.model),
            args.max_batches,
            train=False,
        )
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}  train {tr_loss:.4f}/{tr_acc:.3f}  "
            f"val {va_loss:.4f}/{va_acc:.3f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "val_acc": best_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
            print(f"  saved {out_dir / 'best.pt'}  val_acc={best_acc:.3f}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"done in {time.time() - t0:.1f}s  best val_acc={best_acc:.3f}")


if __name__ == "__main__":
    main()
