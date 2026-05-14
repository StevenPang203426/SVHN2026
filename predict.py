"""
预测入口脚本

用法:
    python predict.py --model checkpoints/baseline-epoch50-acc85.00.pth --output result.csv

    # 使用 TTA (Test-Time Augmentation)
    python predict.py --model checkpoints/best.pth --output result_tta.csv --use_tta

    # 指定模型架构（需与训练时一致）
    python predict.py --model checkpoints/improved_v1.pth --output result.csv \
                      --experiment improved_v1
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from config import build_config, get_data_dir, EXPERIMENT_PRESETS
from utils.inference import predict


def main():
    parser = argparse.ArgumentParser(description="SVHN 街道字符识别 — 推理")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--output", type=str, default="result.csv", help="输出 CSV 路径")
    parser.add_argument("--use_tta", action="store_true", default=False, help="启用 TTA")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENT_PRESETS.keys()),
                        help="使用预设方案加载对应模型架构")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default=None, dest="dataset_path")
    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()

    # 构建配置
    cfg_args = argparse.Namespace(
        experiment=args.experiment,
        model_name=None, loss_type=None, aug_level=None,
        lr=None, batch_size=args.batch_size, epochs=None,
        scheduler=None, use_wandb=False, use_cutout=False,
        use_mixup=False, use_tta=args.use_tta,
        pretrained=None, dataset_path=args.dataset_path,
        experiment_name=None, num_workers=args.num_workers,
    )
    cfg = build_config(cfg_args)
    data_dir = get_data_dir(cfg)

    print(f"[Predict] Model: {args.model}")
    print(f"[Predict] Architecture: {cfg.model_name}")
    print(f"[Predict] TTA: {cfg.use_tta}")

    predict(cfg, data_dir, args.model, args.output)


if __name__ == '__main__':
    main()
