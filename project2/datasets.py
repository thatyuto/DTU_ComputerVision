"""
In this file, we define two datasets: FrameImageDataset and VideoFrameDataset.

bc we should explore 4 different models:
1. pre-frame + aggregate - usually the sample we train is a single frame
2. Late fusion - usually the sample we train is a video
3. Early fusion - usually the sample we train is a video
4. 3d CNN - usually the sample we train is a video

Hence,
we need to define two datasets: FrameImageDataset and VideoFrameDataset.
FrameImageDataset: one sample = one frame. Used to train a per-frame 2D CNN.
VideoFrameDataset: one sample = 10 frames from the same video. Used to train a 3D CNN.
"""



import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

# this file is in project2/, data is in course_cv/data/ufc10/
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "ufc10"

# ImageNet mean/std: required because we use a pretrained ResNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_transform(train=False, image_size=112):
    """Resize a PIL image to a tensor the CNN can use."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FrameImageDataset(Dataset):
    """One sample = one frame. Used to train a per-frame 2D CNN."""

    def __init__(self, data_root=DEFAULT_DATA_ROOT, split="train", transform=None):
        """
        if you don't provide a transform, use the default transform
        note: transform is a function that takes in a PIL image and returns a tensor
        and train=(split == "train") is a boolean that tells you if this is a training set
        """
        self.transform = transform or default_transform(train=(split == "train"))
        self.samples = []  # (image_path, label)

        csv_path = Path(data_root) / "metadata" / f"{split}.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                folder = Path(data_root) / "frames" / split / row["action"] / row["video_name"]
                label = int(row["label"])  # each video has a label
                for i in range(1, 11):  # frame_1.jpg ... frame_10.jpg
                    self.samples.append((folder / f"frame_{i}.jpg", label))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        return x, label

    """
    1. Take an idx
    2. Retrieve the path and label from self.samples[idx]
    3. Open the image and convert it to RGB
    4. Apply transform to convert it into a tensor
    5. Return (tensor, label), where label is the class ID of this action (0-9)
    """



class VideoFrameDataset(Dataset):
    """One sample = 10 frames from the same video.

    stack_frames=True  -> tensor shape [C, T, H, W]
    stack_frames=False -> a list of 10 tensors, each [C, H, W]
    """

    def __init__(self, data_root=DEFAULT_DATA_ROOT, split="train", transform=None, stack_frames=True):
        self.transform = transform or default_transform(train=(split == "train"))
        self.stack_frames = stack_frames
        self.videos = []  # (frame_folder, label)

        csv_path = Path(data_root) / "metadata" / f"{split}.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                folder = Path(data_root) / "frames" / split / row["action"] / row["video_name"]
                self.videos.append((folder, int(row["label"])))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        folder, label = self.videos[idx]
        frames = []
        for i in range(1, 11):
            image = Image.open(folder / f"frame_{i}.jpg").convert("RGB")
            frames.append(self.transform(image))
        if self.stack_frames:
            x = torch.stack(frames, dim=1)  # [C, 10, H, W]
        else:
            x = frames
        return x, label


def _shared_clip_transform(images, train, image_size):
    """Resize, crop, and flip every frame in one clip with the same geometry.

    RGB and flow must stay aligned. Independent RandomCrop per image would
    show the two streams different parts of the person.
    """
    resize_to = int(image_size * 1.15) if train else image_size
    resized = [TF.resize(image, resize_to) for image in images]
    if train:
        top, left, height, width = transforms.RandomCrop.get_params(
            resized[0], (image_size, image_size)
        )
        cropped = [TF.crop(image, top, left, height, width) for image in resized]
        if random.random() < 0.5:
            cropped = [TF.hflip(image) for image in cropped]
    else:
        cropped = [TF.center_crop(image, image_size) for image in resized]
    return [
        TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD) for image in cropped
    ]


class DualVideoDataset(Dataset):
    """One sample = 10 RGB frames and 9 RAFT flow images from the same video.

    Flow files on the cluster are:
      flows_png/{split}/{action}/{video_name}/flow_1_2.png ... flow_9_10.png
    Returns rgb [C, 10, H, W], flow [C, 9, H, W], label.
    """

    def __init__(self, data_root=DEFAULT_DATA_ROOT, split="train", image_size=112):
        self.train = split == "train"
        self.image_size = image_size
        self.videos = []  # (frame_folder, flow_folder, label)
        root = Path(data_root)
        csv_path = root / "metadata" / f"{split}.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel = Path(split) / row["action"] / row["video_name"]
                self.videos.append(
                    (root / "frames" / rel, root / "flows_png" / rel, int(row["label"]))
                )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frame_dir, flow_dir, label = self.videos[idx]
        images = [
            Image.open(frame_dir / f"frame_{i}.jpg").convert("RGB") for i in range(1, 11)
        ]
        images += [
            Image.open(flow_dir / f"flow_{i}_{i + 1}.png").convert("RGB") for i in range(1, 10)
        ]
        tensors = _shared_clip_transform(images, self.train, self.image_size)
        rgb = torch.stack(tensors[:10], dim=1)
        flow = torch.stack(tensors[10:], dim=1)
        return rgb, flow, label
