"""
模型集成方案 - 多模型加权投票

支持两种集成策略:
  1. Logits 平均: 对多个模型的 logits 做加权平均后 argmax
  2. 投票法: 对每个模型的预测结果做多数投票

用法:
    # logits 平均 (推荐)
    python ensemble.py \
        --models checkpoints/baseline.pth checkpoints/improved_v1.pth \
        --archs resnet50 se_resnet50 \
        --weights 0.4 0.6 \
        --strategy logits

    # 多数投票
    python ensemble.py \
        --models ckpt1.pth ckpt2.pth ckpt3.pth \
        --archs resnet50 se_resnet50 resnet50 \
        --strategy vote
"""
import argparse
import os
import sys
from collections import Counter

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get_data_dir
from datasets.dataset import DigitsDataset
from models import build_model


def load_models(model_paths, arch_names, device):
    """Load multiple models."""
    models = []
    for path, arch in zip(model_paths, arch_names):
        cfg = load_config()
        cfg.model_name = arch
        model = build_model(cfg).to(device)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print("  Loaded %s (%s)" % (path, arch))
        models.append(model)
    return models


def ensemble_logits(models, img, weights):
    """Weighted average of logits."""
    all_preds = [[] for _ in range(4)]
    for m, w in zip(models, weights):
        pred = m(img)
        for j in range(4):
            all_preds[j].append(pred[j] * w)
    # Average
    avg = tuple(sum(p) for p in all_preds)
    return avg


def ensemble_vote(models, img):
    """Majority voting on argmax predictions."""
    # Collect votes: shape (n_models, batch, 4)
    all_votes = []
    for m in models:
        pred = m(img)
        digits = [pred[j].argmax(1) for j in range(4)]
        all_votes.append(torch.stack(digits, dim=1))  # (B, 4)

    # Vote per position
    votes = torch.stack(all_votes, dim=0)  # (n_models, B, 4)
    B = votes.size(1)
    result = torch.zeros(B, 4, dtype=torch.long, device=img.device)
    for b in range(B):
        for j in range(4):
            v = votes[:, b, j].tolist()
            result[b, j] = Counter(v).most_common(1)[0][0]
    return result


def parse_result(predictions):
    """Convert digit tensor to strings."""
    char_list = [str(i) for i in range(10)] + ['']
    results = []
    for b in range(predictions.size(0)):
        s = ''.join(char_list[predictions[b, j].item()] for j in range(4))
        results.append(s)
    return results


def main():
    parser = argparse.ArgumentParser(description="Model Ensemble")
    parser.add_argument("--models", nargs='+', required=True)
    parser.add_argument("--archs", nargs='+', required=True)
    parser.add_argument("--weights", nargs='+', type=float, default=None)
    parser.add_argument("--strategy", choices=["logits", "vote"], default="logits")
    parser.add_argument("--output", default="result_ensemble.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", default="./data")
    args = parser.parse_args()

    assert len(args.models) == len(args.archs), "models and archs must match"

    if args.weights is None:
        args.weights = [1.0 / len(args.models)] * len(args.models)
    else:
        assert len(args.weights) == len(args.models)
        total = sum(args.weights)
        args.weights = [w / total for w in args.weights]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 50)
    print("  Ensemble Prediction")
    print("  Strategy: %s" % args.strategy)
    print("  Models: %d" % len(args.models))
    print("  Weights: %s" % args.weights)
    print("=" * 50)

    # Load models
    models = load_models(args.models, args.archs, device)

    # Load test data
    cfg = Config()
    cfg.dataset_path = args.data_path
    data_dir = get_data_dir(cfg)
    test_set = DigitsDataset(cfg, data_dir, mode='test')
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True)

    results = []
    with torch.no_grad():
        for img, img_names in tqdm(test_loader, desc="Ensemble"):
            img = img.to(device)

            if args.strategy == "logits":
                pred = ensemble_logits(models, img, args.weights)
                digits = torch.stack([pred[j].argmax(1) for j in range(4)], dim=1)
            else:
                digits = ensemble_vote(models, img)

            codes = parse_result(digits)
            results.extend(
                [[os.path.basename(n), c] for n, c in zip(img_names, codes)]
            )

    results.sort(key=lambda x: x[0])
    df = pd.DataFrame(results, columns=["file_name", "file_code"])
    df.to_csv(args.output, index=False)
    print("\nEnsemble results saved to %s" % args.output)


if __name__ == "__main__":
    main()
