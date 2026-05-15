"""YOLO Training - supports YOLOv8/YOLO11 with causal augmentation"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config

def _run_causal_aug(cfg):
    try:
        from datasets.causal_augment import augment_yolo_dataset
    except ImportError as e:
        print("[WARN] %s" % e)
        return
    src = getattr(cfg, 'yolo_src_dataset', './data/yolo')
    dst = getattr(cfg, 'yolo_enhanced_dataset', './data/yolo_causal_enhanced')
    yml = os.path.join(dst, "svhn.yaml")
    if os.path.exists(yml):
        print("[CausalAug] Already exists, skip")
        cfg.yolo_data = yml
        return
    if not os.path.isdir(src):
        print("[WARN] No source: %s" % src)
        return
    print("[CausalAug] Running...")
    augment_yolo_dataset(cfg, src, dst)
    cfg.yolo_data = yml

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="yolo")
    pa.add_argument("--epochs", type=int, default=None)
    pa.add_argument("--batch_size", type=int, default=None)
    pa.add_argument("--yolo_model", type=str, default=None)
    pa.add_argument("--yolo_imgsz", type=int, default=None)
    pa.add_argument("--use_wandb", action="store_true", default=None)
    pa.add_argument("--skip_augment", action="store_true")
    args = pa.parse_args()
    cfg = load_config(args.config)
    for k in ['epochs','batch_size','yolo_model','yolo_imgsz','use_wandb']:
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    causal = getattr(cfg, 'use_causal_augment', False)
    if causal and not args.skip_augment:
        _run_causal_aug(cfg)
    dy = getattr(cfg, 'yolo_data', './data/yolo/svhn.yaml')
    if not os.path.exists(dy):
        print("[ERROR] Dataset not found: %s" % dy)
        return
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics")
        return
    pdir = os.path.join(cfg.checkpoints, cfg.experiment_name)
    sd = getattr(cfg, 'yolo_seed', 42)
    print("=" * 60)
    info = "Model=%s Epochs=%d Batch=%d Seed=%d" % (cfg.yolo_model, cfg.epochs, cfg.batch_size, sd)
    print("  YOLO: " + info)
    print("  Data: %s" % dy)
    c_str = "ON" if causal else "OFF"
    print("  Causal=%s Save=%s" % (c_str, pdir))
    print("=" * 60)
    m = YOLO(cfg.yolo_model)
    kw = dict(data=dy, epochs=cfg.epochs, imgsz=cfg.yolo_imgsz, batch=cfg.batch_size,
              project=pdir, name="train", patience=getattr(cfg,'yolo_patience',10),
              save=True, plots=True, exist_ok=True, seed=sd)
    m.train(**kw)
    bp = os.path.join(pdir, "train", "weights", "best.pt")
    print("\nDone! Best: %s" % bp)

if __name__ == "__main__":
    main()
