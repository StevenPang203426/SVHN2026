"""
YOLO 目标检测方案 - 推理脚本

将检测到的数字按 x 坐标从左到右排列, 拼接为最终预测字符串

用法:
    python predict_yolo.py --model checkpoints/yolo_detect/train/weights/best.pt
    python predict_yolo.py --model best.pt --config yolo --conf 0.3
"""
import argparse
import os
import sys
from glob import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config, get_data_dir


def yolo_predict(model_path, test_dir, output_csv, imgsz=320, conf=0.25, batch_size=64):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        return

    model = YOLO(model_path)
    test_imgs = sorted(glob(os.path.join(test_dir, "*.png")))
    print("Predicting %d images..." % len(test_imgs))

    results_list = []
    for start in range(0, len(test_imgs), batch_size):
        batch = test_imgs[start:start + batch_size]
        preds = model.predict(batch, imgsz=imgsz, conf=conf, verbose=False)

        for img_path, result in zip(batch, preds):
            fname = os.path.basename(img_path)
            boxes = result.boxes
            if len(boxes) == 0:
                results_list.append([fname, ""])
                continue

            detections = []
            for box in boxes:
                cls = int(box.cls[0].item())
                x1 = box.xyxy[0][0].item()
                detections.append((x1, cls))

            detections.sort(key=lambda d: d[0])
            code = "".join(str(d[1]) for d in detections[:4])
            results_list.append([fname, code])

    results_list.sort(key=lambda r: r[0])
    df = pd.DataFrame(results_list, columns=["file_name", "file_code"])
    df.to_csv(output_csv, index=False)
    print("Results saved to %s" % output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--config", default="yolo", help="配置文件名")
    parser.add_argument("--output", default="result_yolo.csv")
    parser.add_argument("--conf", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = get_data_dir(cfg)

    imgsz = getattr(cfg, 'yolo_imgsz', 320)
    conf = args.conf if args.conf is not None else getattr(cfg, 'yolo_conf', 0.25)
    test_dir = data_dir['test_data']

    yolo_predict(args.model, test_dir, args.output, imgsz, conf, cfg.batch_size)


if __name__ == "__main__":
    main()
