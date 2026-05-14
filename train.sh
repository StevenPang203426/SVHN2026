#!/bin/bash
# =============================================================================
#  SVHN 街道字符识别 — 全实验自动化训练脚本
#
#  用法:
#    chmod +x train.sh
#    ./train.sh              # 运行所有实验
#    ./train.sh baseline     # 只运行 baseline
#    ./train.sh improved_v1  # 只运行改进方案1
# =============================================================================

set -e

cd "$(dirname "$0")"

# 确保数据已准备
if [ ! -f "data/mchar_train.json" ] || [ ! -s "data/mchar_train.json" ]; then
    echo "Data labels missing. Run: python fix_labels.py"
    exit 1
fi

EXPERIMENT=${1:-"all"}

run_experiment() {
    local name=$1
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  开始实验: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python train.py --experiment "$name" --use_wandb 2>&1 | tee "logs/${name}.log"
    echo "  ✓ 实验 $name 完成"
}

mkdir -p logs checkpoints

if [ "$EXPERIMENT" = "all" ]; then
    echo "🚀 运行全部实验方案"
    echo ""

    # 实验 1: Baseline — ResNet50 + LabelSmooth + medium aug
    run_experiment "baseline"

    # 实验 2: 改进方案 1 — SE-ResNet50 + strong aug + Cutout
    run_experiment "improved_v1"

    # 实验 3: 改进方案 2 — ResNet50 + Focal Loss + MixUp + OneCycle
    run_experiment "improved_v2"

    # 实验 4: 消融实验 — 去除 BatchNorm
    run_experiment "ablation_no_bn"

    # 实验 5: 消融实验 — 过度数据增强
    run_experiment "ablation_extreme_aug"

    echo ""
    echo "  All classification experiments done."
    echo ""

    # YOLO 目标检测实验 (需要 ultralytics)
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
