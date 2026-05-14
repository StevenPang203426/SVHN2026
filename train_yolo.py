"""
YOLO 目标检测方案 - 训练脚本

用法:
    # 使用 YAML 配置
    python train_yolo.py --config yolo

    # 覆盖参数
    python train_yolo.py --config yolo --epochs 100 --batch_size 32

    # 先转换数据格式
    python datasets/convert_to_yolo.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config


def main():
    parser = argparse.ArgumentParser(description="YOLO SVHN Training")
    parser.add_argument("--config", default="yolo", help="配置文件名")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--yolo_model", type=str, default=None)
    parser.add_argument("--yolo_imgsz", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=None)
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)
    # 命令行覆盖
    for k in ['epochs', 'batch_size', 'yolo_model', 'yolo_imgsz', 'use_wandb']:
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)

    # 检查数据
    data_yaml = getattr(cfg, 'yolo_data', './data/yolo/svhn.yaml')
    if not os.path.exists(data_yaml):
        print("[ERROR] YOLO dataset not found: %s" % data_yaml)
        print("Please run: python datasets/convert_to_yolo.py")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    # checkpoint 按实验名存子目录
    project_dir = os.path.join(cfg.checkpoints, cfg.experiment_name)

    print("=" * 60)
    print("  YOLO SVHN Training")
    print("  Model:      %s" % cfg.yolo_model)
    print("  Data:       %s" % data_yaml)
    print("  Epochs:     %d" % cfg.epochs)
    print("  ImgSz:      %d" % cfg.yolo_imgsz)
    print("  Batch:      %d" % cfg.batch_size)
    print("  Save to:    %s" % project_dir)
    print("=" * 60)

    model = YOLO(cfg.yolo_model)
    results = model.train(
        data=data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.yolo_imgsz,
        batch=cfg.batch_size,
        project=project_dir,
        name="train",
        patience=getattr(cfg, 'yolo_patience', 10),
        save=True,
        plots=True,
        exist_ok=True,
    )

    best_path = os.path.join(project_dir, "train", "weights", "best.pt")
    print("\nTraining complete!")
    print("Best model: %s" % best_path)


if __name__ == "__main__":
    main()
