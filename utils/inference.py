"""
推理工具 — 支持 TTA（Test-Time Augmentation）
"""

import time

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from datasets.dataset import DigitsDataset
from models import build_model


def parse2class(prediction):
    """
    将 4 个分类头输出转为字符串

    Args:
        prediction: (c1, c2, c3, c4)，每个 shape (B, 11)
    Returns:
        list[str]
    """
    char_list = [str(i) for i in range(10)] + ['']
    results = []
    indices = [p.argmax(1) for p in prediction]
    for b in range(indices[0].size(0)):
        s = ''.join(char_list[indices[j][b].item()] for j in range(4))
        results.append(s)
    return results


def write2csv(results, csv_path):
    """将预测结果写入 CSV"""
    import os
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: os.path.basename(x))
    df.to_csv(csv_path, sep=',', index=None)
    print(f'[Inference] Results saved to {csv_path}')


def predict(cfg, data_dir, model_path, csv_path):
    """
    标准推理

    Args:
        cfg: Config
        data_dir: 数据路径字典
        model_path: 模型权重路径
        csv_path: 结果 CSV 输出路径
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set = DigitsDataset(cfg, data_dir, mode='test')
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f'[Inference] Model loaded from {model_path}')

    results = []
    start_time = time.time()

    with torch.no_grad():
        for img, img_names in tqdm(test_loader, desc='Predicting'):
            img = img.to(device)

            if cfg.use_tta:
                pred = tta_predict(model, img)
            else:
                pred = model(img)

            codes = parse2class(pred)
            results.extend([[name, code] for name, code in zip(img_names, codes)])

    elapsed = time.time() - start_time
    results.sort(key=lambda x: x[0])
    write2csv(results, csv_path)

    speed = len(results) / elapsed
    print(f'[Inference] {len(results)} images in {elapsed:.1f}s ({speed:.1f} img/s)')
    return results


def tta_predict(model, img):
    """
    TTA: 原图 + 水平翻转，对 logits 取平均

    Args:
        model: nn.Module
        img: Tensor (B, C, H, W)
    Returns:
        tuple of 4 tensors
    """
    # 原始预测
    pred_orig = model(img)

    # 水平翻转预测
    img_flip = torch.flip(img, dims=[3])
    pred_flip = model(img_flip)

    # 取平均
    pred_avg = tuple(
        (pred_orig[j] + pred_flip[j]) / 2.0 for j in range(4)
    )
    return pred_avg
