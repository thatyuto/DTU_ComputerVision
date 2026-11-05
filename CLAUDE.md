# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Coursework for DTU's Introduction to Deep Learning in Computer Vision (IDLCV, course 02516), in PyTorch. The course's exercise notebooks and slides sit at the root; the graded work is in two projects:

- `project1/`: hotdog vs. not-hotdog image classification, in notebooks. This is the only git repo (`thatyuto/Deep_Learning_CV_project`). The root and `project2/` are not under version control, despite `project2/.gitignore`.
- `project2/`: UCF-10 video action classification, in Python scripts. This is the only part with reusable code.

There is no test suite, linter or build step. Use a short `--max-batches` training run as the check that code works.

## Environment

- Local: run with `~/mambaforge/envs/dl_env/bin/python`. It has torch and torchvision; the base mambaforge `python` does not.
- `project2/train.py` and `eval.py` choose only CUDA or CPU, so they run on CPU on this Mac. The project 1 notebooks also fall back to MPS.
- HPC: DTU LSF cluster, queue `c02516`, with `module load python3/3.10.16` and a venv at `/zhome/13/2/232841/idlcv_hpc/.venv`. Files are copied over with scp. Submit jobs with `bsub < jobscript.sh`.

## Project 2 commands

Run from `project2/`. Outputs go to `runs/<model>` relative to the current directory.

```bash
# Quick check. Use --out-dir so it doesn't overwrite a real run's best.pt
python train.py --model per_frame --epochs 1 --max-batches 2 --no-pretrained --out-dir /tmp/smoke

# Full run: --model is per_frame | late | early | 3d | dual
python train.py --data-root /path/to/data --model late --epochs 20

# Video-level accuracy from a checkpoint (the model type is read from the checkpoint)
python eval.py --ckpt runs/late/best.pt --split test
```

`--max-batches` also limits validation, and the validation loader is unshuffled, so accuracy from a debug run means nothing. `eval.py` does not read `--image-size` from the checkpoint; pass it again if training used a non-default value.

## Project 2 architecture

`train.py` picks a dataset and model using the name helpers in `models.py` (`build_model`, `needs_video_clip`, `needs_flow`). `eval.py` reuses `train.run_epoch`. A new model has to be added to the `build_model` table and to the `--model` choices in `train.py`, and to the helpers if its input differs.

Each model's training input differs, but validation and test are always per video:

| `--model` | Class | Dataset used for training | Training input |
|---|---|---|---|
| `per_frame` | `PerFrameNet` | `FrameImageDataset` (5000 samples) | single frames `[B,C,H,W]` |
| `late` | `LateFusionNet` | `VideoFrameDataset` | clips `[B,C,T,H,W]` |
| `early` | `EarlyFusionNet` | `VideoFrameDataset` | clips, flattened to `[B,C*T,H,W]` |
| `3d` | `Conv3DNet` (R3D-18) | `VideoFrameDataset` | clips `[B,C,T,H,W]` |
| `dual` | `DualStreamNet` | `DualVideoDataset` | RGB clip plus flow clip |

- `per_frame` trains on single frames but averages per-frame logits over the clip at evaluation, so `PerFrameNet.forward` accepts both 4D and 5D input.
- `dual` batches are `(rgb, flow, label)` rather than `(x, label)`; `unpack_batch` and `run_epoch` handle this. Its logits are the average of two `LateFusionNet` streams.
- `DualVideoDataset` uses `_shared_clip_transform` so every RGB and flow image in a clip gets the same crop and flip. Don't replace it with per-image random transforms, which would misalign the streams.
- Clips use a fixed 10 frames (`N_FRAMES`, plus hardcoded `range(1, 11)` in `datasets.py`) and 10 classes. Inputs are normalized with ImageNet statistics, including flow images.
- `train.py` saves `best.pt` (highest validation accuracy) and `history.json` to the output directory.

## Data

`data/` holds the datasets; `project2` defaults to `data/ufc10`. Expected layout under a data root:

```
metadata/{train,val,test}.csv        # video_name, video_path, action, label
frames/{split}/{action}/{video_name}/frame_{1..10}.jpg
flows_png/{split}/{action}/{video_name}/flow_{i}_{i+1}.png   # i = 1..9; dual only
```

- `flows_png` is not available locally, so `--model dual` only runs on the cluster, against the no-leakage dataset set in `project2/jobscript.sh`.
- The local `data/ufc10` has leakage: UCF video groups (`_gNN`) overlap across its splits. The no-leakage copy is on the cluster under `/dtu/datasets1/02516/`. Results from the two aren't comparable.
- Project 1 notebooks look for `/dtu/datasets1/02516/hotdog_nothotdog`, then `../data/hotdog_nothotdog`.

## Project 1 notebooks

- `IDLCV_Project_1_hotdog_v1.ipynb` is the group version: the full sequence from baseline through optimizers, BatchNorm, augmentation, transfer learning, saliency maps, and ConvNeXt with TTA. It holds out 80/20 of the training set for validation and expects `hotdog_nothotdog` relative to the notebook.
- `IDLCV_Project_1_hotdog.ipynb` is a separate version that reruns comparisons against a wider baseline. Its markdown refers to the "group v1 baseline".

## Conventions

- Assignment briefs are the PDFs in each project (`Project_1_hotdog.pdf`, `Project_2_VideoClassification.pdf`).
- `project2/sec/*.tex` holds report sections. Numbers in them should come from actual runs.
- Comments mix English and Chinese, and many are explanatory study notes. Keep them when editing nearby code.
