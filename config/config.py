"""
超参数配置与数据路径定义
支持通过 argparse 覆盖，或通过预设实验方案快速切换
"""

import os
import argparse


class Config:
    """全局配置（默认值 = Baseline 方案）"""

    # ── 数据集 ──
    dataset_path = "./data"

    # ── 训练参数 ──
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    class_num = 11
    num_workers = 8

    # ── 训练调度 ──
    start_epoch = 0
    epochs = 50
    eval_interval = 1
    print_interval = 50

    # ── 模型 ──
    model_name = "resnet50"          # resnet50 | se_resnet50 | resnet50_no_bn
    pretrained_backbone = True
    pretrained = None                # 载入训练好的权重路径
    checkpoints = "./checkpoints"

    # ── 损失函数 ──
    loss_type = "label_smooth"       # label_smooth | focal | ce
    smooth = 0.1
    focal_alpha = 0.25
    focal_gamma = 2.0

    # ── 数据增强 ──
    aug_level = "medium"             # light | medium | strong | extreme
    img_h = 128
    img_w = 224
    use_cutout = False
    use_mixup = False
    mixup_alpha = 0.2

    # ── 学习率策略 ──
    scheduler = "cosine_warm"        # cosine_warm | step | one_cycle
    T_0 = 10
    T_mult = 2
    milestones = [20, 35, 45]
    gamma = 0.1

    # ── 日志 ──
    use_wandb = False
    wandb_project = "svhn-recognition"
    experiment_name = "baseline"

    # ── TTA ──
    use_tta = False


def get_data_dir(cfg):
    """根据 config 构建数据路径索引"""
    p = cfg.dataset_path
    return {
        'train_data': os.path.join(p, 'mchar_train', 'mchar_train') + os.sep,
        'val_data': os.path.join(p, 'mchar_val', 'mchar_val') + os.sep,
        'test_data': os.path.join(p, 'mchar_test_a', 'mchar_test_a') + os.sep,
        'train_label': os.path.join(p, 'mchar_train.json'),
        'val_label': os.path.join(p, 'mchar_val.json'),
        'submit_file': os.path.join(p, 'mchar_sample_submit_A.csv'),
    }


# ── 预设实验方案 ──

EXPERIMENT_PRESETS = {
    "baseline": {},

    "improved_v1": {
        "model_name": "se_resnet50",
        "aug_level": "strong",
        "use_cutout": True,
        "lr": 5e-4,
        "epochs": 60,
        "scheduler": "cosine_warm",
        "experiment_name": "improved_v1_se_resnet50",
    },

    "improved_v2": {
        "model_name": "resnet50",
        "aug_level": "strong",
        "use_cutout": True,
        "use_mixup": True,
        "loss_type": "focal",
        "lr": 3e-4,
        "epochs": 60,
        "scheduler": "one_cycle",
        "experiment_name": "improved_v2_focal_mixup",
    },

    "ablation_no_bn": {
        "model_name": "resnet50_no_bn",
        "aug_level": "medium",
        "lr": 1e-4,
        "epochs": 30,
        "experiment_name": "ablation_no_bn",
    },

    "ablation_extreme_aug": {
        "model_name": "resnet50",
        "aug_level": "extreme",
        "use_cutout": True,
        "epochs": 30,
        "experiment_name": "ablation_extreme_aug",
    },
}


def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description="SVHN Street Character Recognition")

    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENT_PRESETS.keys()),
                        help="使用预设实验方案")
    parser.add_argument("--model", type=str, default=None, dest="model_name")
    parser.add_argument("--loss", type=str, default=None, dest="loss_type")
    parser.add_argument("--aug", type=str, default=None, dest="aug_level")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--use_cutout", action="store_true", default=False)
    parser.add_argument("--use_mixup", action="store_true", default=False)
    parser.add_argument("--use_tta", action="store_true", default=False)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None, dest="dataset_path")
    parser.add_argument("--name", type=str, default=None, dest="experiment_name")
    parser.add_argument("--num_workers", type=int, default=None)

    return parser.parse_args()


def build_config(args=None):
    """
    构建最终配置：默认值 → 预设方案覆盖 → 命令行参数覆盖

    Args:
        args: parse_args() 的返回值，为 None 时使用默认配置
    Returns:
        Config 实例
    """
    cfg = Config()

    # 1) 预设方案覆盖
    if args is not None and args.experiment is not None:
        preset = EXPERIMENT_PRESETS[args.experiment]
        for k, v in preset.items():
            setattr(cfg, k, v)

    # 2) 命令行参数覆盖（仅覆盖显式传入的参数）
    if args is not None:
        for k, v in vars(args).items():
            if v is not None and k != "experiment":
                setattr(cfg, k, v)

    return cfg


# 默认全局配置（兼容旧代码直接 import）
config = Config()
data_dir = get_data_dir(config)
