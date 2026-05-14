"""
预测入口脚本

用法:
    python predict.py --model checkpoints/baseline-best.pth --output result.csv

    # 使用 TTA
    python predict.py --model checkpoints/best.pth --output result_tta.csv --use_tta

    # 指定实验配置（加载对应模型架构）
    python predict.py --model checkpoints/improved_v1.pth --output result.csv \
                      --config improved_v1
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from config import load_config, get_data_dir
from utils.inference import predict


def main():
    parser = argparse.ArgumentParser(description="SVHN 街道字符识别 — 推理")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--output", type=str, default="result.csv", help="输出 CSV 路径")
    parser.add_argument("--use_tta", action="store_true", default=False, help="启用 TTA")
    parser.add_argument("--config", type=str, default=None,
                        help="实验配置名 (如 improved_v1), 用于加载对应模型架构")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default=None, dest="dataset_path")
    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 命令行覆盖
    if args.dataset_path:
        cfg.dataset_path = args.dataset_path
    cfg.batch_size = args.batch_size
    cfg.use_tta = args.use_tta
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    data_dir = get_data_dir(cfg)

    print("[Predict] Model: %s" % args.model)
    print("[Predict] Architecture: %s" % cfg.model_name)
    print("[Predict] TTA: %s" % cfg.use_tta)

    predict(cfg, data_dir, args.model, args.output)


if __name__ == '__main__':
    main()
