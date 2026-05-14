"""
YOLO 目标检测方案 - 推理脚本

将检测到的数字按 x 坐标从左到右排列, 拼接为最终预测字符串

用法:
    python predict_yolo.py --model runs/detect/svhn_yolo/weights/best.pt
"""
import argparse
import os
import sys
from glob import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def yolo_predict(model_path, test_dir, output_csv, imgsz=320, conf=0.25):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        return

    model = YOLO(model_path)
    test_imgs = sorted(glob(os.path.join(test_dir, "*.png")))
    print("Predicting %d images..." % len(test_imgs))

    results_list = []
    # Process in batches
    batch_size = 64
    for start in range(0, len(test_imgs), batch_size):
        batch = test_imgs[start:start + batch_size]
        preds = model.predict(batch, imgsz=imgsz, conf=conf, verbose=False)

        for img_path, result in zip(batch, preds):
            fname = os.path.basename(img_path)
            boxes = result.boxes
            if len(boxes) == 0:
                results_list.append([fname, ""])
                continue

            # 按 x 坐标排序, 拼接数字
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
    parser.add_argument("--test_dir", default="./data/mchar_test_a/mchar_test_a")
    parser.add_argument("--output", default="result_yolo.csv")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    yolo_predict(args.model, args.test_dir, args.output, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
