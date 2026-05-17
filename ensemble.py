"""
模型集成方案 - 支持分类模型 + YOLO 混合投票

集成策略:
  1. vote (推荐): 每个模型独立预测, 对整串结果做多数投票
  2. logits: 仅限分类模型, 对 logits 加权平均后 argmax

YOLO 模型只能参与投票 (vote), 因为 YOLO 的输出是检测框而非 logits

用法:
    # 纯分类模型集成 (logits 平均)
    python ensemble.py \
        --models checkpoints/improved_v2/improved_v2_focal_mixup-ep59-acc80.63.pth checkpoints/improved_v1/best.pth \
        --archs resnet50 se_resnet50 \
        --weights 0.4 0.6 \
        --strategy logits

    # 分类 + YOLO 混合投票 (等权)
    python ensemble.py \
        --models checkpoints/improved_v2/improved_v2_focal_mixup-ep59-acc80.63.pth \
                 runs/detect/checkpoints/causal_yolo/train/weights/last.pt \
        --archs resnet50 yolo \
        --strategy vote

    # YOLO 为主, ResNet 为辅 (权重 1:5, YOLO 投票权重是 ResNet 的 5 倍)
    python ensemble.py \
        --models checkpoints/improved_v2/improved_v2_focal_mixup-ep59-acc80.63.pth \
                 runs/detect/checkpoints/causal_yolo/train/weights/last.pt \
        --archs resnet50 yolo \
        --weights 1 5 \
        --strategy vote

    # 在验证集上评估集成效果
    python ensemble.py \
        --models ckpt1.pth ckpt2.pth ckpt3.pth \
        --archs resnet50 se_resnet50 yolo \
        --strategy vote --eval_val


    python ensemble.py \
        --models checkpoints/improved_v2/improved_v2_focal_mixup-ep59-acc80.63.pth \
                 runs/detect/checkpoints/causal_yolo/train/weights/last.pt \
        --archs resnet50 yolo \
        --strategy vote --eval_val
"""
import argparse
import json
import os
import sys
from collections import Counter
from glob import glob

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get_data_dir
from datasets.dataset import DigitsDataset
from models import build_model


# ─────────────────────────────────────────────
#  分类模型预测
# ─────────────────────────────────────────────

def load_cls_model(path, arch, device):
    """加载分类模型"""
    cfg = load_config()
    cfg.model_name = arch
    model = build_model(cfg).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("  [CLS] Loaded %s (%s)" % (path, arch))
    return model


def cls_predict_batch(model, img):
    """分类模型预测一个 batch, 返回字符串列表"""
    pred = model(img)
    char_list = [str(i) for i in range(10)] + ['']
    results = []
    indices = [pred[j].argmax(1) for j in range(4)]
    for b in range(indices[0].size(0)):
        s = ''.join(char_list[indices[j][b].item()] for j in range(4))
        results.append(s)
    return results


def cls_logits_batch(model, img):
    """分类模型预测一个 batch, 返回原始 logits"""
    return model(img)


# ─────────────────────────────────────────────
#  YOLO 模型预测
# ─────────────────────────────────────────────

def load_yolo_model(path):
    """加载 YOLO 模型"""
    from ultralytics import YOLO
    model = YOLO(path)
    print("  [YOLO] Loaded %s" % path)
    return model


def yolo_predict_paths(yolo_model, img_paths, imgsz=320, conf=0.25, batch_size=32):
    """YOLO 预测一组图片路径, 返回 {filename: pred_str}

    分批推理 + stream=True 避免显存溢出
    """
    results = {}
    for start in tqdm(range(0, len(img_paths), batch_size), desc="  YOLO batch"):
        batch = img_paths[start:start + batch_size]
        preds = yolo_model.predict(batch, imgsz=imgsz, conf=conf,
                                   verbose=False, stream=True)
        for img_path, result in zip(batch, preds):
            fname = os.path.basename(img_path)
            boxes = result.boxes
            if len(boxes) == 0:
                results[fname] = ""
            else:
                detections = []
                for box in boxes:
                    cls = int(box.cls[0].item())
                    x1 = box.xyxy[0][0].item()
                    detections.append((x1, cls))
                detections.sort(key=lambda d: d[0])
                results[fname] = ''.join(str(d[1]) for d in detections)
    return results


# ─────────────────────────────────────────────
#  集成逻辑
# ─────────────────────────────────────────────

def ensemble_vote_strings(all_predictions, vote_weights=None):
    """
    对多个模型的字符串预测做加权投票

    Args:
        all_predictions: list[str], 每个模型对同一张图的预测
        vote_weights:    list[float] 或 None, 每个模型的投票权重
                         None 时每个模型权重为 1 (等权投票)
    Returns:
        str: 投票结果
    """
    if len(all_predictions) == 1:
        return all_predictions[0]
    if vote_weights is None:
        vote_weights = [1.0] * len(all_predictions)
    counter = {}
    for pred, w in zip(all_predictions, vote_weights):
        counter[pred] = counter.get(pred, 0.0) + w
    return max(counter, key=counter.get)


def ensemble_logits_avg(logits_list, weights):
    """对多个分类模型的 logits 做加权平均"""
    avg = []
    for j in range(4):
        weighted = sum(logits[j] * w for logits, w in zip(logits_list, weights))
        avg.append(weighted)
    return avg


def main():
    parser = argparse.ArgumentParser(description="Model Ensemble (CLS + YOLO)")
    parser.add_argument("--models", nargs='+', required=True,
                        help="模型权重路径列表")
    parser.add_argument("--archs", nargs='+', required=True,
                        help="对应架构 (resnet50 / se_resnet50 / yolo)")
    parser.add_argument("--weights", nargs='+', type=float, default=None,
                        help="每个模型的权重, 与 --models 一一对应. "
                             "vote 策略下为投票权重, logits 策略下为 logits 加权系数. "
                             "例: --weights 1 5 表示第一个模型权重1, 第二个权重5")
    parser.add_argument("--strategy", choices=["logits", "vote"], default="vote")
    parser.add_argument("--output", default="result_ensemble.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--config", type=str, default=None,
                        help="实验配置名 (用于读取 yolo_conf 等参数)")
    parser.add_argument("--yolo_conf", type=float, default=None,
                        help="YOLO 推理置信度阈值 (覆盖配置文件)")
    parser.add_argument("--eval_val", action="store_true",
                        help="在验证集上评估集成 Acc")
    args = parser.parse_args()

    assert len(args.models) == len(args.archs), "models and archs must match"

    # 分离分类模型和 YOLO 模型
    cls_models_info = []
    yolo_models_info = []
    for path, arch in zip(args.models, args.archs):
        if arch == 'yolo':
            yolo_models_info.append(path)
        else:
            cls_models_info.append((path, arch))

    has_yolo = len(yolo_models_info) > 0

    if args.strategy == "logits" and has_yolo:
        print("[WARN] YOLO 模型不支持 logits 策略, 自动切换为 vote")
        args.strategy = "vote"

    # 权重处理
    # vote 策略: weights 与 --models 一一对应, 作为投票权重
    # logits 策略: weights 仅对分类模型有效, 作为 logits 加权系数
    if args.weights is None:
        args.weights = [1.0] * len(args.models)
    assert len(args.weights) == len(args.models), \
        "--weights 数量 (%d) 必须与 --models 数量 (%d) 一致" % (
            len(args.weights), len(args.models))

    # 分离分类和 YOLO 的权重
    cls_weights = []
    yolo_weights = []
    for (path, arch), w in zip(zip(args.models, args.archs), args.weights):
        if arch == 'yolo':
            yolo_weights.append(w)
        else:
            cls_weights.append(w)

    if args.strategy == "logits":
        total_w = sum(cls_weights)
        cls_weights = [w / total_w for w in cls_weights]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("  Ensemble Prediction")
    print("  Strategy:   %s" % args.strategy)
    print("  CLS models: %d" % len(cls_models_info))
    print("  YOLO models: %d" % len(yolo_models_info))
    print("  CLS weights: %s" % cls_weights)
    print("  YOLO weights: %s" % yolo_weights)
    print("  Eval val:   %s" % args.eval_val)
    print("=" * 60)

    # 加载模型
    cls_models = [load_cls_model(p, a, device) for p, a in cls_models_info]
    yolo_models = []
    if has_yolo:
        from ultralytics import YOLO
        yolo_models = [load_yolo_model(p) for p in yolo_models_info]

    # 加载数据
    cfg = load_config(args.config)
    cfg.dataset_path = args.data_path
    if args.yolo_conf is not None:
        cfg.yolo_conf = args.yolo_conf
    data_dir = get_data_dir(cfg)

    mode = 'val' if args.eval_val else 'test'
    dataset = DigitsDataset(cfg, data_dir, mode=mode)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True)

    # 如果有 YOLO, 需要收集图片路径做批量预测
    yolo_preds = {}
    yolo_ordered_fnames = []  # val 模式下按 index 对应
    if has_yolo:
        if mode == 'test':
            all_paths = sorted(glob(data_dir['test_data'] + '*.png'))
        else:
            all_paths = sorted(glob(data_dir['val_data'] + '*.png'))

        yolo_ordered_fnames = [os.path.basename(p) for p in all_paths]

        print("\n[YOLO] Predicting %d images..." % len(all_paths))
        imgsz = getattr(cfg, 'yolo_imgsz', 320)
        conf = getattr(cfg, 'yolo_conf', 0.25)
        for ym in yolo_models:
            yp = yolo_predict_paths(ym, all_paths, imgsz, conf)
            for fname, pred_str in yp.items():
                yolo_preds.setdefault(fname, []).append(pred_str)

    # 主循环
    results = []
    total, correct = 0, 0
    char_list = [str(i) for i in range(10)] + ['']

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Ensemble"):
            if mode == 'val':
                img, label = batch_data
                # 构造文件名（val 模式下没有路径, 用 index）
                img_names = None
            else:
                img, img_names = batch_data

            img = img.to(device)
            B = img.size(0)

            if args.strategy == "logits":
                # 仅分类模型, logits 加权平均
                all_logits = [cls_logits_batch(m, img) for m in cls_models]
                avg = ensemble_logits_avg(all_logits, cls_weights)
                for b in range(B):
                    s = ''.join(char_list[avg[j][b].argmax().item()] for j in range(4))
                    if mode == 'val':
                        gt = ''.join(char_list[label[b, j].item()] for j in range(4))
                        if s == gt:
                            correct += 1
                        total += 1
                    else:
                        fname = os.path.basename(img_names[b])
                        results.append([fname, s])

            else:
                # vote 策略: 收集所有模型的字符串预测 + 权重
                # 1) 分类模型预测
                cls_preds_batch = []
                for m in cls_models:
                    cls_preds_batch.append(cls_predict_batch(m, img))

                for b in range(B):
                    votes = [cp[b] for cp in cls_preds_batch]
                    weights = list(cls_weights)

                    # 2) YOLO 预测 (如果有)
                    if has_yolo:
                        if mode == 'val':
                            # val 模式下用全局 index 对应排序后的文件名
                            idx = total + b
                            if idx < len(yolo_ordered_fnames):
                                fname = yolo_ordered_fnames[idx]
                            else:
                                fname = None
                        else:
                            fname = os.path.basename(img_names[b]) if img_names else None

                        if fname is not None:
                            yp_list = yolo_preds.get(fname, [])
                            for i, yp in enumerate(yp_list):
                                votes.append(yp)
                                w = yolo_weights[i] if i < len(yolo_weights) else 1.0
                                weights.append(w)

                    winner = ensemble_vote_strings(votes, weights)

                    if mode == 'val':
                        gt = ''.join(char_list[label[b, j].item()] for j in range(4))
                        if winner == gt:
                            correct += 1
                        total += 1
                    else:
                        fname = os.path.basename(img_names[b])
                        results.append([fname, winner])

    if mode == 'val':
        acc = correct / total if total > 0 else 0
        print("\n" + "=" * 60)
        print("  Ensemble Val Accuracy: %.2f%% (%d/%d)" % (acc * 100, correct, total))
        print("=" * 60)
    else:
        results.sort(key=lambda x: x[0])
        df = pd.DataFrame(results, columns=["file_name", "file_code"])
        df.to_csv(args.output, index=False)
        print("\nEnsemble results saved to %s" % args.output)


if __name__ == "__main__":
    main()
