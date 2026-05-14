"""
YOLO 目标检测方案 - 训练脚本

思路: 将每个数字视为独立的检测目标 (10 个类别: 0-9)
检测结果按 bbox 的 x 坐标从左到右排列, 拼接得到最终预测

依赖: pip install ultralytics

用法:
    # 1. 转换数据格式
    python datasets/convert_to_yolo.py

    # 2. 训练 YOLO
    python train_yolo.py --epochs 50 --model yolov8s.pt --imgsz 320

    # 3. 推理
    python predict_yolo.py --model runs/detect/train/weights/best.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="YOLO SVHN Training")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLO pretrained model")
    parser.add_argument("--data", default="./data/yolo/svhn.yaml", help="Dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--name", default="svhn_yolo")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    # 检查数据
    if not os.path.exists(args.data):
        print("[ERROR] YOLO dataset not found: %s" % args.data)
        print("Please run: python datasets/convert_to_yolo.py")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        print("Run: pip install ultralytics")
        return

    print("=" * 50)
    print("  YOLO SVHN Training")
    print("  Model:  %s" % args.model)
    print("  Data:   %s" % args.data)
    print("  Epochs: %d" % args.epochs)
    print("  ImgSz:  %d" % args.imgsz)
    print("=" * 50)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=10,
        save=True,
        plots=True,
    )

    print("\nTraining complete!")
    print("Best model: runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
