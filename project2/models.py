"""Video classifiers for Project 4.1 / 4.2.

we should explore 4 different models:
1. pre-frame + aggregate - usually the sample we train is a single frame
2. Late fusion - usually the sample we train is a video
3. Early fusion - usually the sample we train is a video
4. 3d CNN - usually the sample we train is a video

Hence,
we need to define four models: PerFrameNet, LateFusionNet, EarlyFusionNet, and Conv3DNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights

NUM_CLASSES = 10
N_FRAMES = 10


"""
ResNet-18 本来连在一起做两件事:
前面很多卷积层看图，最后挤成 512 个数，这 512 个数就是特征，比如「人倒立、手臂伸直」。
"""

def _resnet18_backbone(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    net = resnet18(weights=weights)
    net.fc = nn.Identity()   # Identity()是恒等变换，不改变输入，直接输出
    """
    因为resnet18最后一层就有一个fc分类头, 使它们分成1000类, 但对于我们来说并不需要, 所以我们把原本的fc层改了, 改成Identity层, 意思就是不改变512, 宏观上来看就是把原本的fc层删掉了
    """
    return net


class PerFrameNet(nn.Module):
    """2D CNN on each frame; train on single frames, aggregate at video test.

    Video input [B, C, T, H, W] -> mean of per-frame logits (aggregation).
    Frame input [B, C, H, W] -> logits.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = _resnet18_backbone(pretrained)
        self.fc = nn.Linear(512, num_classes)

    # 这里的forwar_frame函数是用来处理单帧输入的
    def forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))

    # 通俗来讲，forward函数是深度学习中定义数据从输入到输出的过程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:   # check if the input is single frame, if dim is 4, then it is single frame
            return self.forward_frame(x)
        # [B, C, T, H, W]
        """
        b：这一批的视频数
        c：颜色通道，这里是 3
        t：帧数，这里是 10
        h、w：每一帧的高和宽
        """
        b, c, t, h, w = x.shape

        """
        permute(0, 2, 1, 3, 4)：这一步是把时间维度（t）移动到最前面，把 [B, C, T, H, W] 变成 [B, T, C, H, W]。
        reshape(b * t, c, h, w)：这一步是把时间维度（t）和颜色通道（c）合并，把 [B, T, C, H, W] 变成 [B*T, C, H, W]。


        permute 是重新排列张量的维度顺序，数据本身不变，只是换轴的位置。
        参数 (0, 2, 1, 3, 4) 表示新张量的第 0、1、2、3、4 维分别来自原来的第 0、2、1、3、4 维。原来的形状是 [b, 3, 10, h, w]，对应轴编号 0、1、2、3、4。换完之后是 [b, 10, 3, h, w]：帧数到了颜色通道前面。
        """
        logits = self.forward_frame(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        return logits.view(b, t, -1).mean(dim=1)


class LateFusionNet(nn.Module):
    """Shared 2D backbone per frame, then fuse features (mean + concat-linear)."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = _resnet18_backbone(pretrained)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        b, c, t, h, w = x.shape
        feats = self.backbone(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        feats = feats.view(b, t, -1).mean(dim=1)
        return self.fc(feats)


class EarlyFusionNet(nn.Module):
    """Stack time into channels: [B, C*T, H, W] into a 2D CNN."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        n_frames: int = N_FRAMES,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        old = net.conv1  
        net.conv1 = nn.Conv2d(
            3 * n_frames,  # because this is early fusion, we need to make the input channel to be 3 * n_frames
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            # repeat RGB filters across time so ImageNet init is not wasted
            # repeat(),repeat 的四个数，按顺序告诉每一维要复制几遍。
            net.conv1.weight.copy_(old.weight.repeat(1, n_frames, 1, 1) / n_frames)
        net.fc = nn.Linear(512, num_classes)
        self.net = net
        self.n_frames = n_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b, c * t, h, w)
        return self.net(x)


class Conv3DNet(nn.Module):
    """3D CNN (R3D-18). Input [B, C, T, H, W]."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.net = r3d_18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        return self.net(x)


class DualStreamNet(nn.Module):
    """RGB + optical-flow streams (4.2.2). Average logits.

    Flow PNGs are loaded as 3-channel images, same as RGB.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.rgb = LateFusionNet(num_classes=num_classes, pretrained=pretrained)
        self.flow = LateFusionNet(num_classes=num_classes, pretrained=pretrained)

    # 这里的forward函数是用来处理RGB和光流输入的
    """
    rgb 是彩色画面，flow是光流图。两路都是张量，形状和普通视频片段一样，都是 [B, 3, T, H, W]
    这里的光流图是事先算好的，每个像素往哪动的图。相邻两frame之间，一个点向右动，图上就有一个向右的标记
    向上动就有一个向上的标记。这个practice的光流图已经用RAFT算好了，放在HPC里面。
    """
    def forward(self, rgb: torch.Tensor, flow: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.rgb(rgb)
        if flow is None:
            return logits
        return 0.5 * (logits + self.flow(flow))  ## 两路平均一下


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    key = name.lower().replace("-", "_")
    table = {
        "per_frame": PerFrameNet,
        "late": LateFusionNet,
        "late_fusion": LateFusionNet,
        "early": EarlyFusionNet,
        "early_fusion": EarlyFusionNet,
        "3d": Conv3DNet,
        "conv3d": Conv3DNet,
        "dual": DualStreamNet,
        "dual_stream": DualStreamNet,
    }
    if key not in table:
        raise ValueError(f"Unknown model {name!r}. Choose from {sorted(table)}")
    return table[key](num_classes=num_classes, pretrained=pretrained)


def needs_video_clip(name: str) -> bool:
    """Per-frame trains on single images; others need [C, T, H, W]."""
    return name.lower().replace("-", "_") not in {"per_frame"}


def needs_flow(name: str) -> bool:
    return name.lower().replace("-", "_") in {"dual", "dual_stream"}
