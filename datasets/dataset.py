"""
街道字符识别数据集 — 支持多级数据增强、Cutout、MixUp
"""

import json
import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Cutout:
    """随机遮挡正方形区域"""

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """img: Tensor (C, H, W)"""
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).unsqueeze(0).expand_as(img)
        return img * mask


def build_transforms(cfg, mode='train'):
    """
    根据 aug_level 构建变换管线

    Args:
        cfg: Config 实例
        mode: 'train' | 'val' | 'test'
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if mode != 'train':
        return transforms.Compose([
            transforms.Resize(cfg.img_h),
            transforms.CenterCrop((cfg.img_h, cfg.img_w)),
            transforms.ToTensor(),
            normalize,
        ])

    aug_list = [
        transforms.Resize(cfg.img_h),
        transforms.CenterCrop((cfg.img_h, cfg.img_w)),
    ]

    level = cfg.aug_level

    if level == "light":
        aug_list.extend([
            transforms.ColorJitter(0.05, 0.05, 0.05),
        ])
    elif level == "medium":
        aug_list.extend([
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.RandomGrayscale(0.1),
            transforms.RandomAffine(15, translate=(0.05, 0.1), shear=5),
        ])
    elif level == "strong":
        aug_list.extend([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomGrayscale(0.15),
            transforms.RandomAffine(20, translate=(0.08, 0.15), shear=8),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        ])
    elif level == "extreme":
        aug_list.extend([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            transforms.RandomGrayscale(0.5),
            transforms.RandomAffine(
                45, translate=(0.2, 0.3), shear=20, scale=(0.6, 1.4)
            ),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.6),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0)),
            transforms.RandomAutocontrast(p=0.5),
        ])

    aug_list.extend([transforms.ToTensor(), normalize])

    if cfg.use_cutout and level != "light":
        aug_list.append(Cutout(n_holes=1, length=20))

    return transforms.Compose(aug_list)


class DigitsDataset(Dataset):
    """
    街道字符识别数据集

    Args:
        cfg: Config 实例
        data_dir: 数据路径字典
        mode: 'train' | 'val' | 'test'
    """

    def __init__(self, cfg, data_dir, mode='train'):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        self.width = cfg.img_w
        self.batch_count = 0
        self.transform = build_transforms(cfg, mode)

        if mode == 'test':
            self.imgs = sorted(glob(data_dir['test_data'] + '*.png'))
            self.labels = None
        else:
            labels = json.load(open(data_dir['%s_label' % mode], 'r'))
            imgs = sorted(glob(data_dir['%s_data' % mode] + '*.png'))
            self.imgs = [
                (img, labels[os.path.split(img)[-1]])
                for img in imgs
                if os.path.split(img)[-1] in labels
            ]

    def __getitem__(self, idx):
        if self.mode != 'test':
            img_path, label = self.imgs[idx]
        else:
            img_path = self.imgs[idx]
            label = None

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.mode != 'test':
            padded = label['label'][:4] + (4 - len(label['label'])) * [10]
            return img, torch.tensor(padded).long()
        else:
            return img, img_path

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        """自定义 collate 函数"""
        imgs, labels = zip(*batch)
        return torch.stack(imgs).float(), torch.stack(labels)


def mixup_data(x, y, alpha=0.2):
    """
    MixUp 数据增强

    Args:
        x: 输入图像 (B, C, H, W)
        y: 标签 (B, 4)
        alpha: Beta 分布参数

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
