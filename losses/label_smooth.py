"""
损失函数集合：Label Smoothing CE / Focal Loss / 标准 CE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.size_average = size_average

    def forward(self, preds, targets):
        lb_pos = 1.0 - self.smooth
        lb_neg = self.smooth / (preds.shape[1] - 1)
        smoothed_lb = torch.zeros_like(preds).fill_(lb_neg).scatter_(
            1, targets[:, None], lb_pos
        )
        log_soft = F.log_softmax(preds, dim=1)

        if self.class_weights is not None:
            loss = -log_soft * smoothed_lb * self.class_weights[None, :]
        else:
            loss = -log_soft * smoothed_lb

        loss = loss.sum(1)
        if self.size_average == 'mean':
            return loss.mean()
        elif self.size_average == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError


class FocalLoss(nn.Module):
    """
    Focal Loss — 解决类别不平衡问题

    对于容易分类的样本降低权重，让模型更专注于困难样本。
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def build_loss(cfg):
    """
    根据配置构建损失函数

    Args:
        cfg: Config 实例
    Returns:
        nn.Module
    """
    if cfg.loss_type == "label_smooth":
        return LabelSmoothEntropy(smooth=cfg.smooth)
    elif cfg.loss_type == "focal":
        return FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    elif cfg.loss_type == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {cfg.loss_type}")
