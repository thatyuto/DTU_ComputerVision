"""
任务 2: 五个分割指标。

输入都是0/1的 tensor, 形状相同 (pred 已经做过 sigmoid + 阈值 0.5)。
可选 valid: 只在 valid==1 的像素上计算 (DRIVE用FOV mask)。
先从 pred / target 得到 TP, FP, FN, TN, 五个指标都由这四个数算出来。
"""

from __future__ import annotations

import torch

EPS = 1e-7  # 防止分母为 0


def confusion(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None):
    """返回 (tp, fp, fn, tn)，都是标量 tensor。"""
    pred = pred.float()
    target = target.float()
    # valid 为 None 时所有像素都算 -> 用全 1 代替
    valid = torch.ones_like(target) if valid is None else valid.float()


    # 0/1 相乘 = "且": 只有两个条件都满足时结果才是 1；再乘 valid 去掉 FOV 外的像素
    tp = (pred * target * valid).sum()              # 说是，真是
    fp = (pred * (1 - target) * valid).sum()        # 说是，其实不是 (误报)
    fn = ((1 - pred) * target * valid).sum()        # 说不是，其实是 (漏报)
    tn = ((1 - pred) * (1 - target) * valid).sum()  # 说不是，真不是
    return tp, fp, fn, tn

    # 注意上面这个1-x，可以把x原来的格子翻转。如果只是普通乘法只能捕抓到1，但1-x，就能将整件事翻转，可以捕抓到0（此时捕抓不到1）
    """
	我要的 pred 是几？	我要的 target 是几？	公式
    TP 说是、真是	1 → pred	1 → target	pred * target
    FP 说是、真不是	1 → pred	0 → 1-target	pred * (1-target)
    FN 说不是、真是	0 → 1-pred	1 → target	(1-pred) * target
    TN 说不是、真不是	0 → 1-pred	0 → 1-target	(1-pred) * (1-target)
    """



# 输入 confusion()数出来的4个数字，返回一个0~1之间的分数，表示预测区域和真实区域重合得有多好
def dice(tp, fp, fn, tn):
    # 2*TP / (2TP + FP + FN)
    return 2 * tp / (2 * tp + fp + fn + EPS)  # eps is a small number to avoid division by zero)


def iou(tp, fp, fn, tn):
    # TP / (TP + FP + FN)
    return tp / (tp + fp + fn + EPS)


# calculate how many samples were correctly identified
def accuracy(tp, fp, fn, tn):
    # (TP + TN) / 全部
    return (tp + tn) / (tp + tn + fn + fp + EPS)


# Recall , calculate how many positive samples were correctly identified
def sensitivity(tp, fp, fn, tn):
    # 召回率 TP / (TP + FN)
    return tp / (tp + fn + EPS)


# calculate how many background samples were correctly identified
def specificity(tp, fp, fn, tn):
    # TN / (TN + FP)
    return tn / (tn + fp + EPS)


def all_metrics(pred, target, valid=None) -> dict[str, float]:
    """一次返回五个指标, train.py 和 eval.py 都用它。"""
    c = confusion(pred, target, valid)
    return {
        "dice": float(dice(*c)),
        "iou": float(iou(*c)),
        "accuracy": float(accuracy(*c)),
        "sensitivity": float(sensitivity(*c)),
        "specificity": float(specificity(*c)),
    }


if __name__ == "__main__":
    # 自检: 完美预测 -> 全部为 1；全预测成背景 -> dice=0，但 accuracy 依然很高
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:4, 2:4] = 1
    print(all_metrics(t, t))
    print(all_metrics(torch.zeros_like(t), t))
