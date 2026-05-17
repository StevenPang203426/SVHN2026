"""
统一评估脚本 — 在验证集上计算整串准确率 (Full-String Accuracy)

支持两种模型:
  1. 分类模型 (ResNet50 / SE-ResNet50): 4 个分类头 argmax 拼接
  2. YOLO 检测模型: 检测框按 x 坐标排序拼接

"整串准确率" = 所有数字位全部正确的样本比例 (最严格的指标)

用法:
    # 评估分类模型
    python evaluate.py --model checkpoints/baseline/ep50-acc86.00.pth --config baseline

    # 评估 SE-ResNet50
    python evaluate.py --model checkpoints/improved_v1/best.pth --config improved_v1
    python evaluate.py --model checkpoints/improved_v2_focal_mixup-ep59-acc80.63.pth --config improved_v2

    # 评估 YOLO 模型
    python evaluate.py --model checkpoints/yolo_detect/train/weights/best.pt --config yolo
    python evaluate.py --model runs/detect/svhn_yolo/weights/best.pt  --config yolo
    python evaluate.py --model runs/detect/checkpoints/causal_yolo/train/weights/best.pt --config causal_yolo
    python evaluate.py --model checkpoints/best.pt --config causal_yolo
    # 同时输出逐位准确率
    python evaluate.py --model checkpoints/baseline/best.pth --config baseline --per_digit
"""
import argparse
import json
import os
import sys
from glob import glob

import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get_data_dir


def eval_classification(model_path, cfg, data_dir, per_digit=False):
    """评估分类模型的整串准确率"""
    from torch.utils.data import DataLoader
    from datasets.dataset import DigitsDataset
    from models import build_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = build_model(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 加载验证集
    val_set = DigitsDataset(cfg, data_dir, mode='val')
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False)

    total, correct = 0, 0
    digit_correct = [0, 0, 0, 0]
    digit_total = [0, 0, 0, 0]

    with torch.no_grad():
        for img, label in tqdm(val_loader, desc="Evaluating"):
            img, label = img.to(device), label.to(device)
            pred = model(img)

            for j in range(4):
                pred_j = pred[j].argmax(1)
                digit_correct[j] += (pred_j == label[:, j]).sum().item()
                digit_total[j] += label.size(0)

            # 整串: 4 位全对才算正确
            match = torch.stack(
                [pred[j].argmax(1) == label[:, j] for j in range(4)], dim=1)
            correct += torch.all(match, dim=1).sum().item()
            total += img.size(0)

    acc = correct / total
    print("\n" + "=" * 50)
    print("  Model:           %s" % cfg.model_name)
    print("  Checkpoint:      %s" % model_path)
    print("  Val Samples:     %d" % total)
    print("  Full-String Acc: %.2f%% (%d/%d)" % (acc * 100, correct, total))

    if per_digit:
        print("  Per-Digit Acc:")
        for j in range(4):
            da = digit_correct[j] / digit_total[j] * 100
            print("    Digit %d: %.2f%%" % (j + 1, da))

    print("=" * 50)
    return acc


def eval_yolo(model_path, cfg, data_dir, per_digit=False):
    """评估 YOLO 模型的整串准确率"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        return 0.0

    # 加载标签
    label_path = data_dir['val_label']
    with open(label_path, 'r') as f:
        labels = json.load(f)

    # 加载验证图片
    val_dir = data_dir['val_data']
    val_imgs = sorted(glob(val_dir + '*.png'))
    if len(val_imgs) == 0:
        print("[ERROR] No validation images in %s" % val_dir)
        return 0.0

    model = YOLO(model_path)
    imgsz = getattr(cfg, 'yolo_imgsz', 320)
    conf = getattr(cfg, 'yolo_conf', 0.25)

    total, correct = 0, 0
    digit_match = [0, 0, 0, 0]
    digit_count = [0, 0, 0, 0]
    batch_size = cfg.batch_size

    for start in tqdm(range(0, len(val_imgs), batch_size), desc="YOLO Eval"):
        batch = val_imgs[start:start + batch_size]
        preds = model.predict(batch, imgsz=imgsz, conf=conf, verbose=False)

        for img_path, result in zip(batch, preds):
            fname = os.path.basename(img_path)
            if fname not in labels:
                continue

            # 真实标签: 转为字符串
            gt_digits = labels[fname]['label']
            gt_str = ''.join(str(d) for d in gt_digits)

            # YOLO 预测: 按 x 坐标排序
            boxes = result.boxes
            if len(boxes) == 0:
                pred_str = ""
            else:
                detections = []
                for box in boxes:
                    cls = int(box.cls[0].item())
                    x1 = box.xyxy[0][0].item()
                    detections.append((x1, cls))
                detections.sort(key=lambda d: d[0])
                pred_str = ''.join(str(d[1]) for d in detections)

            # 整串准确率
            if pred_str == gt_str:
                correct += 1
            total += 1

            # 逐位准确率
            if per_digit:
                for j in range(min(4, len(gt_str))):
                    digit_count[j] += 1
                    if j < len(pred_str) and str(pred_str[j]) == str(gt_str[j]):
                        digit_match[j] += 1

    acc = correct / total if total > 0 else 0.0
    print("\n" + "=" * 50)
    print("  Model:           YOLO (%s)" % os.path.basename(model_path))
    print("  Checkpoint:      %s" % model_path)
    print("  Val Samples:     %d" % total)
    print("  Full-String Acc: %.2f%% (%d/%d)" % (acc * 100, correct, total))

    if per_digit:
        print("  Per-Digit Acc:")
        for j in range(4):
            if digit_count[j] > 0:
                da = digit_match[j] / digit_count[j] * 100
                print("    Digit %d: %.2f%%" % (j + 1, da))

    print("=" * 50)
    return acc


def main():
    parser = argparse.ArgumentParser(description="统一验证集评估")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--config", type=str, default=None, help="实验配置名")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--per_digit", action="store_true", help="输出逐位准确率")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path:
        cfg.dataset_path = args.data_path
    if args.batch_size:
        cfg.batch_size = args.batch_size
    data_dir = get_data_dir(cfg)

    # 自动判断模型类型
    is_yolo = (args.model.endswith('.pt') and 'yolo' in args.model.lower()) \
              or getattr(cfg, 'model_name', '') == 'yolo'

    if is_yolo:
        eval_yolo(args.model, cfg, data_dir, args.per_digit)
    else:
        eval_classification(args.model, cfg, data_dir, args.per_digit)


if __name__ == '__main__':
    main()
