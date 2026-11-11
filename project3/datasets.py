"""
任务 1: PH2 和 DRIVE 的 Dataset 和 train/val/test 划分。

两个数据集都在集群的 /dtu/datasets1/02516 下，本地没有。
先在集群上 ls 看清目录结构，再把下面的路径和文件名规则填进去。
以课上 segmentation exercise 的 data loader 为起点。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

DATA_ROOT = Path("/dtu/datasets1/02516")
PH2_ROOT = DATA_ROOT / "PH2_Dataset_images"   # TODO: 在集群上确认真实文件夹名
DRIVE_ROOT = DATA_ROOT / "DRIVE"              # TODO: 在集群上确认真实文件夹名

IMAGE_SIZE = 256  # TODO: 任务 3 要求选好 resize 尺寸。DRIVE 血管很细，可能要更大


def split_ids(ids: list[str], val_frac: float, test_frac: float, seed: int = 0):
    """把样本 id 随机打乱后切成 (train, val, test) 三份。固定 seed 保证每次划分一样。"""
    # TODO
    raise NotImplementedError


class PH2Dataset(Dataset):
    """皮肤病变分割。返回 (image [3,H,W] float, mask [1,H,W] 取值 0/1)。"""

    def __init__(self, split: str, transform=None, seed: int = 0):
        # TODO: 列出所有病例 -> split_ids -> 只保留本 split 的样本
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        # TODO: 读图像和 lesion mask，resize，转 tensor，mask 二值化
        # 注意: 图像和 mask 的随机增强(翻转等)必须同步
        raise NotImplementedError


class DRIVEDataset(Dataset):
    """
    视网膜血管分割。返回 (image, vessel_mask, fov_mask)。

    NB: label 要用血管 mask (DRIVE 原版里是 1st_manual/*.gif)，不是 FOV mask
    (mask/*.gif，一个白色实心圆)。FOV mask 只用来在视野内计算指标。
    DRIVE 官方 test 集可能没有血管标注 -> 只能从有标注的图里自己切 train/val/test。
    """

    def __init__(self, split: str, transform=None, seed: int = 0):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        # 插值: image 用 bilinear，mask 用 nearest (否则 mask 出现 0~1 之间的值)
        raise NotImplementedError


def build_dataset(name: str, split: str, **kwargs) -> Dataset:
    """name: 'ph2' | 'drive'"""
    if name == "ph2":
        return PH2Dataset(split, **kwargs)
    if name == "drive":
        return DRIVEDataset(split, **kwargs)
    raise ValueError(f"unknown dataset: {name}")


if __name__ == "__main__":
    # 自检: 打印每个 split 的大小和一个样本的 shape / mask 取值
    for name in ["ph2", "drive"]:
        for split in ["train", "val", "test"]:
            ds = build_dataset(name, split)
            sample = ds[0]
            print(name, split, len(ds), [t.shape for t in sample], torch.unique(sample[1]))
