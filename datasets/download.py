"""
数据集下载、解压与 YOLO 格式转换

亮点保留:
    - 临时文件下载 + 中断自动清理 (来自 CausalYOLOSvhn 的 get_dataset.py)
    - 进度条显示文件大小和下载速度
    - 支持原始 CSV 链接下载和直接 URL 下载两种方式
    - 自动将 SVHN 标注转换为 YOLO 格式
"""

import json
import os
import shutil
import tempfile
import zipfile

import requests
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
#  天池竞赛数据集下载链接
# ─────────────────────────────────────────────

SVHN_URLS = {
    "mchar_train.zip": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.zip",
    "mchar_train.json": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json",
    "mchar_val.zip": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.zip",
    "mchar_val.json": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json",
    "mchar_test_a.zip": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_test_a.zip",
    "mchar_sample_submit_A.csv": "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_sample_submit_A.csv",
}


# ─────────────────────────────────────────────
#  安全下载 (临时文件 + 中断清理)
# ─────────────────────────────────────────────

def download_file(url, filepath):
    """
    下载文件到指定路径, 带进度条

    使用临时文件中转: 先下载到同目录的临时文件, 完成后原子性移动到目标路径。
    如果下载中断, 临时文件会被自动清理, 不会留下损坏的半成品文件。

    亮点: 此实现来自 CausalYOLOSvhn, 比原版直接写目标文件更安全。

    Args:
        url:      下载链接
        filepath: 保存路径
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
                delete=False, dir=os.path.dirname(filepath)) as tmp:
            tmp_path = tmp.name
            with tqdm(desc="  下载 %s" % os.path.basename(filepath),
                      total=total_size, unit="B",
                      unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = tmp.write(chunk)
                    bar.update(size)
        shutil.move(tmp_path, filepath)
    except BaseException:
        # 中断时清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ─────────────────────────────────────────────
#  下载与解压
# ─────────────────────────────────────────────

def download_and_extract(csv_path=None, dataset_path='./data'):
    """
    下载 SVHN 数据集并解压

    支持两种方式:
        1. csv_path 不为 None: 从 CSV 文件中读取链接 (旧方式, 兼容)
        2. csv_path 为 None:   使用内置 SVHN_URLS 直接下载 (推荐)

    Args:
        csv_path:     包含下载链接的 CSV 文件路径 (可选)
        dataset_path: 数据集存放目录, 默认 ./data
    """
    os.makedirs(dataset_path, exist_ok=True)

    # 选择下载源
    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        links = pd.read_csv(csv_path)
        files_to_download = {
            links['file'][i]: links['link'][i]
            for i in range(len(links))
        }
    else:
        if csv_path:
            print("[INFO] CSV 文件 %s 不存在, 使用内置链接下载" % csv_path)
        files_to_download = SVHN_URLS

    print("[Download] 数据目录: %s" % dataset_path)

    # 下载文件
    for filename, url in files_to_download.items():
        filepath = os.path.join(dataset_path, filename)
        if os.path.exists(filepath):
            print("  [SKIP] %s 已存在" % filename)
            continue
        print("  正在下载 %s ..." % filename)
        try:
            download_file(url, filepath)
            print("  [OK] %s" % filename)
        except Exception as e:
            print("  [FAIL] %s: %s" % (filename, e))

    # 解压 zip 文件
    zip_list = ['mchar_train', 'mchar_val', 'mchar_test_a']
    for name in zip_list:
        target = os.path.join(dataset_path, name)
        zip_path = os.path.join(dataset_path, "%s.zip" % name)
        if not os.path.exists(target) and os.path.exists(zip_path):
            print("  正在解压 %s.zip ..." % name)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(path=dataset_path)
            # 清理 macOS 残留
            macosx = os.path.join(dataset_path, "__MACOSX")
            if os.path.isdir(macosx):
                shutil.rmtree(macosx)
            print("  [OK] %s 解压完成" % name)

    print("[Download] 数据准备完成")


# ─────────────────────────────────────────────
#  SVHN → YOLO 格式转换
# ─────────────────────────────────────────────

def convert_svhn_to_yolo(raw_data_dir, yolo_output_dir):
    """
    将 SVHN 原始标注 (JSON) 转换为 YOLO 检测格式

    亮点: 此功能来自 CausalYOLOSvhn 的 get_dataset.py, 实现了完整的
    SVHN → YOLO 格式转换, 包括训练集、验证集和测试集。

    转换规则:
        - SVHN JSON 标注: {filename: {label:[], left:[], top:[], width:[], height:[]}}
        - YOLO 标注: class_id x_center y_center width height (归一化)

    Args:
        raw_data_dir:   原始数据目录 (包含 mchar_train/, mchar_val/, JSON 等)
        yolo_output_dir: YOLO 数据集输出目录

    目录结构:
        yolo_output_dir/
        ├── train/images/  train/labels/
        ├── val/images/    val/labels/
        ├── test/images/
        └── svhn.yaml
    """
    splits = [
        ("train", "mchar_train.json", "mchar_train"),
        ("val", "mchar_val.json", "mchar_val"),
    ]

    for split_name, json_name, img_folder in splits:
        json_path = os.path.join(raw_data_dir, json_name)
        img_dir = os.path.join(raw_data_dir, img_folder)

        if not os.path.exists(json_path):
            print("[WARN] 标签文件不存在: %s, 跳过 %s" % (json_path, split_name))
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        out_img = os.path.join(yolo_output_dir, split_name, "images")
        out_lbl = os.path.join(yolo_output_dir, split_name, "labels")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        for img_name, attrs in tqdm(data.items(), desc="转换 %s" % split_name):
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # 生成 YOLO 标注
            lines = []
            for label, left, top, w, h in zip(
                    attrs["label"], attrs["left"], attrs["top"],
                    attrs["width"], attrs["height"]):
                xc = (left + w / 2) / img_w
                yc = (top + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append("%d %.6f %.6f %.6f %.6f" % (label, xc, yc, nw, nh))

            # 写标签
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            with open(os.path.join(out_lbl, txt_name), "w") as f:
                f.write("\n".join(lines))

            # 复制图像
            shutil.copy(img_path, out_img)

    # 测试集 (仅图像, 无标签)
    test_src = os.path.join(raw_data_dir, "mchar_test_a")
    test_dst = os.path.join(yolo_output_dir, "test", "images")
    if os.path.isdir(test_src):
        os.makedirs(test_dst, exist_ok=True)
        for f in tqdm(os.listdir(test_src), desc="转换 test"):
            if os.path.isfile(os.path.join(test_src, f)):
                shutil.copy(os.path.join(test_src, f), test_dst)

    # 生成 YOLO 配置
    config_path = os.path.join(yolo_output_dir, "svhn.yaml")
    with open(config_path, "w") as f:
        f.write("path: %s\n" % os.path.abspath(yolo_output_dir))
        f.write("train: train/images\nval: val/images\ntest: test/images\n\n")
        f.write("nc: 10\nnames:\n")
        for i in range(10):
            f.write("- '%d'\n" % i)

    print("[YOLO] 格式转换完成: %s" % yolo_output_dir)
    print("[YOLO] 配置文件: %s" % config_path)


if __name__ == '__main__':
    download_and_extract()
