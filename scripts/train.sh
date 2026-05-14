#!/bin/bash
# =============================================================================
#  SVHN 街道字符识别 — 全实验自动化训练脚本
#
#  用法:
#    bash scripts/train.sh              # 运行所有实验
#    bash scripts/train.sh baseline     # 只运行 baseline
#    bash scripts/train.sh improved_v1  # 只运行改进方案1
# =============================================================================

set -e

# cd 到项目根目录（scripts/ 的上一级）
cd "$(dirname "$0")/.."

# 确保数据已准备
if [ ! -f "data/mchar_train.json" ] || [ ! -s "data/mchar_train.json" ]; then
    echo "Data labels missing. Run: python scripts/fix_labels.py"
    exit 1
fi

EXPERIMENT=${1:-"all"}

run_experiment() {
    local name=$1
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  开始实验: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python train.py --config "$name" --use_wandb 2>&1 | tee "logs/${name}.log"
    echo "  实验 $name 完成"
}

mkdir -p logs checkpoints

if [ "$EXPERIMENT" = "all" ]; then
    echo "运行全部实验方案"
    echo ""

    run_experiment "baseline"
    run_experiment "improved_v1"
    run_experiment "improved_v2"
    run_experiment "ablation_no_bn"
    run_experiment "ablation_extreme_aug"

    echo ""
    echo "  All classification experiments done."
    echo ""

    # YOLO 目标检测实验
    if python -c "import ultralytics" 2>/dev/null; then
        echo "  Starting YOLO experiment..."
        python datasets/convert_to_yolo.py 2>&1 | tee logs/yolo_convert.log
        python train_yolo.py --epochs 50 --imgsz 320 2>&1 | tee logs/yolo_train.log
    else
        echo "  [SKIP] YOLO: pip install ultralytics to enable"
    fi

    echo ""
    echo "  All experiments complete! Check logs/ and checkpoints/"
elif [ "$EXPERIMENT" = "yolo" ]; then
    python datasets/convert_to_yolo.py
    python train_yolo.py --epochs 50 --imgsz 320 2>&1 | tee logs/yolo_train.log
else
    run_experiment "$EXPERIMENT"
fi
