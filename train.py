"""
训练入口脚本

用法:
    # Baseline (default.yaml)
    python train.py

    # 使用实验配置
    python train.py --config baseline
    python train.py --config improved_v1
    python train.py --config improved_v2
    python train.py --config ablation_no_bn
    python train.py --config ablation_extreme_aug

    # 自定义参数 (覆盖 yaml)
    python train.py --config baseline --lr 5e-4 --epochs 60

    # 启用 wandb 追踪
    python train.py --config improved_v1 --use_wandb
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import parse_args, build_config, get_data_dir
from engine import Trainer


def main():
    args = parse_args()
    cfg = build_config(args)
    data_dir = get_data_dir(cfg)

    print("=" * 60)
    print("  Experiment: %s" % cfg.experiment_name)
    print("  Model:      %s" % cfg.model_name)
    print("  Loss:       %s (smooth=%s)" % (cfg.loss_type, cfg.smooth))
    print("  Aug:        %s | Cutout=%s | MixUp=%s" % (cfg.aug_level, cfg.use_cutout, cfg.use_mixup))
    print("  LR:         %s | Scheduler: %s" % (cfg.lr, cfg.scheduler))
    print("  Epochs:     %s | Batch: %s" % (cfg.epochs, cfg.batch_size))
    print("  wandb:      %s" % cfg.use_wandb)
    print("=" * 60)

    # 检查数据
    label_file = data_dir['train_label']
    if not os.path.exists(label_file):
        print("\n[ERROR] 标签文件不存在: %s" % label_file)
        print("请先运行: python scripts/fix_labels.py")
        return

    # 训练
    trainer = Trainer(cfg, data_dir, val=True)
    best_acc = trainer.train()

    print("\n" + "=" * 60)
    print("  训练完成！最优验证准确率: %.2f%%" % (best_acc * 100))
    print("  最优模型路径: %s" % trainer.best_checkpoint_path)
    print("=" * 60)

    return trainer


if __name__ == '__main__':
    main()
