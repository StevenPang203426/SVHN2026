"""
训练入口脚本

用法:
    # Baseline
    python train.py

    # 使用预设实验方案
    python train.py --experiment improved_v1
    python train.py --experiment improved_v2
    python train.py --experiment ablation_no_bn
    python train.py --experiment ablation_extreme_aug

    # 自定义参数
    python train.py --model se_resnet50 --loss focal --aug strong --lr 5e-4 --epochs 60

    # 启用 wandb 追踪
    python train.py --experiment improved_v1 --use_wandb
"""

import sys
import os

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import parse_args, build_config, get_data_dir
from datasets import download_and_extract
from engine import Trainer


def main():
    args = parse_args()
    cfg = build_config(args)
    data_dir = get_data_dir(cfg)

    print("=" * 60)
    print(f"  Experiment: {cfg.experiment_name}")
    print(f"  Model:      {cfg.model_name}")
    print(f"  Loss:       {cfg.loss_type} (smooth={cfg.smooth})")
    print(f"  Aug:        {cfg.aug_level} | Cutout={cfg.use_cutout} | MixUp={cfg.use_mixup}")
    print(f"  LR:         {cfg.lr} | Scheduler: {cfg.scheduler}")
    print(f"  Epochs:     {cfg.epochs} | Batch: {cfg.batch_size}")
    print(f"  wandb:      {cfg.use_wandb}")
    print("=" * 60)

    # 1. 检查数据（不再自动下载，避免网络问题）
    label_file = data_dir['train_label']
    if not os.path.exists(label_file):
        print(f"\n[ERROR] 标签文件不存在: {label_file}")
        print("请先运行: python data/download.py")
        print("或手动下载 mchar_train.json / mchar_val.json 到 data/ 目录")
        return

    # 2. 训练
    trainer = Trainer(cfg, data_dir, val=True)
    best_acc = trainer.train()

    print(f"\n{'=' * 60}")
    print(f"  训练完成！")
    print(f"  最优验证准确率: {best_acc * 100:.2f}%")
    print(f"  最优模型路径:   {trainer.best_checkpoint_path}")
    print(f"{'=' * 60}")

    return trainer


if __name__ == '__main__':
    main()
