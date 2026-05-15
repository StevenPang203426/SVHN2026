"""
将 SVHN JSON 标注转换为 YOLO 格式

YOLO 格式：每张图对应一个 .txt 文件
  每行: class_id center_x center_y width height (归一化到 0-1)
  class_id: 0-9 对应数字 0-9

用法:
    python datasets/convert_to_yolo.py --data_dir ./data --output_dir ./data/yolo
"""
import json
import os
import shutil
from glob import glob
from PIL import Image
import argparse


def convert_split(img_dir, label_path, output_dir, split_name):
    """转换一个数据子集 (train/val)"""
    img_out = os.path.join(output_dir, "images", split_name)
    lbl_out = os.path.join(output_dir, "labels", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    labels = json.load(open(label_path, 'r'))
    count = 0

    for img_name, ann in labels.items():
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path)
        W, H = img.size

        # 生成 YOLO 标签
        txt_name = img_name.replace('.png', '.txt')
        lines = []
        for i in range(len(ann['label'])):
            cls = ann['label'][i]
            if cls == 10:
                continue
            left = ann['left'][i]
            top = ann['top'][i]
            w = ann['width'][i]
            h = ann['height'][i]

            # 转为 YOLO 格式: center_x, center_y, w, h (归一化)
            cx = (left + w / 2.0) / W
            cy = (top + h / 2.0) / H
            nw = w / W
            nh = h / H

            # 裁剪到合法范围
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            lines.append("%d %.6f %.6f %.6f %.6f" % (cls, cx, cy, nw, nh))

        if lines:
            with open(os.path.join(lbl_out, txt_name), 'w') as f:
                f.write('\n'.join(lines))
            # 创建图片的符号链接或复制
            dst = os.path.join(img_out, img_name)
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)
            count += 1

    print("  %s: %d images converted" % (split_name, count))


def create_yaml(output_dir):
    """生成 YOLO 数据集配置 YAML"""
    yaml_content = """# SVHN YOLO Dataset Config
path: %s
train: images/train
val: images/val

nc: 10
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
""" % os.path.abspath(output_dir)

    yaml_path = os.path.join(output_dir, "svhn.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print("  YAML config: %s" % yaml_path)
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./data/yolo")
    args = parser.parse_args()

    print("Converting SVHN to YOLO format...")

    # Train
    train_img = os.path.join(args.data_dir, "mchar_train")
    train_lbl = os.path.join(args.data_dir, "mchar_train.json")
    if os.path.exists(train_lbl) and os.path.getsize(train_lbl) > 0:
        convert_split(train_img, train_lbl, args.output_dir, "train")
    else:
        print("  [SKIP] train label not found or empty")

    # Val
    val_img = os.path.join(args.data_dir, "mchar_val")
    val_lbl = os.path.join(args.data_dir, "mchar_val.json")
    if os.path.exists(val_lbl) and os.path.getsize(val_lbl) > 0:
        convert_split(val_img, val_lbl, args.output_dir, "val")
    else:
        print("  [SKIP] val label not found or empty")

    # YAML
    yaml_path = create_yaml(args.output_dir)
    print("\nDone! To train YOLO:")
    print("  yolo detect train data=%s model=yolov8n.pt epochs=50 imgsz=320" % yaml_path)


if __name__ == '__main__':
    main()
