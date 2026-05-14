"""
基于 ResNet50 的多位数字识别模型
包含三个变体：标准 / SE 增强 / 去除 BN（消融实验）
"""

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class DigitsResnet50(nn.Module):
    """
    标准 ResNet50 + 4 个分类头

    Args:
        class_num: 每个位置的类别数（0-9 + 空 = 11）
    """

    def __init__(self, class_num=11):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        feat = self.dropout(feat)
        return self.fc1(feat), self.fc2(feat), self.fc3(feat), self.fc4(feat)


class SEDigitsResnet50(nn.Module):
    """
    改进方案 1：ResNet50 + SE 注意力 + Dropout

    在 ResNet50 的 layer2/3/4 输出后加入 SE 模块，
    增强通道级别的特征选择能力。
    """

    def __init__(self, class_num=11):
        super().__init__()
        backbone = resnet50(pretrained=True)
        children = list(backbone.children())

        self.layer0 = nn.Sequential(*children[:4])   # conv1 + bn + relu + maxpool
        self.layer1 = children[4]                     # 256 channels
        self.layer2 = children[5]                     # 512 channels
        self.layer3 = children[6]                     # 1024 channels
        self.layer4 = children[7]                     # 2048 channels
        self.avgpool = children[8]                    # AdaptiveAvgPool2d

        # SE 模块
        self.se2 = SEBlock(512, reduction=16)
        self.se3 = SEBlock(1024, reduction=16)
        self.se4 = SEBlock(2048, reduction=16)

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        x = self.layer0(img)
        x = self.layer1(x)
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        feat = self.avgpool(x).view(x.size(0), -1)
        feat = self.dropout(feat)
        return self.fc1(feat), self.fc2(feat), self.fc3(feat), self.fc4(feat)


class DigitsResnet50NoBN(nn.Module):
    """
    消融实验：去除 ResNet50 中所有 BatchNorm 层

    理论预期：去除 BN 后模型训练不稳定，
    容易出现梯度爆炸/消失，准确率显著下降。
    """

    def __init__(self, class_num=11):
        super().__init__()
        backbone = resnet50(pretrained=True)
        # 去除最后的全连接层
        modules = list(backbone.children())[:-1]
        # 递归移除所有 BatchNorm 层
        self.cnn = nn.Sequential(*modules)
        self._remove_bn(self.cnn)

        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def _remove_bn(self, module):
        """递归将所有 BN 层替换为 Identity"""
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                setattr(module, name, nn.Identity())
            else:
                self._remove_bn(child)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        return self.fc1(feat), self.fc2(feat), self.fc3(feat), self.fc4(feat)


def build_model(cfg):
    """
    根据配置构建模型

    Args:
        cfg: Config 实例
    Returns:
        nn.Module
    """
    model_map = {
        "resnet50": DigitsResnet50,
        "se_resnet50": SEDigitsResnet50,
        "resnet50_no_bn": DigitsResnet50NoBN,
    }

    if cfg.model_name not in model_map:
        raise ValueError(f"Unknown model: {cfg.model_name}. "
                         f"Available: {list(model_map.keys())}")

    return model_map[cfg.model_name](class_num=cfg.class_num)
