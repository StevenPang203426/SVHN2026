#!/bin/bash
# =============================================================================
#  SVHN 街道字符识别 — 推理脚本
#
#  用法:
#    chmod +x predict.sh
#    ./predict.sh                                    # 使用最新 best 模型
#    ./predict.sh checkpoints/xxx.pth result.csv     # 指定模型和输出
# =============================================================================

set -e

cd "$(dirname "$0")"

MODEL_PATH=${1:-""}
OUTPUT=${2:-"result.csv"}
USE_TTA=${3:-"false"}

# 如果未指定模型路径，自动查找最新的 checkpoint
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=$(ls -t checkpoints/*.pth 2>/dev/null | head -1)
    if [ -z "$MODEL_PATH" ]; then
        echo "❌ 未找到模型权重。请先运行训练或指定模型路径。"
        echo "用法: ./predict.sh <model_path> [output.csv] [true|false for TTA]"
        exit 1
    fi
    echo "🔍 自动选择最新模型: $MODEL_PATH"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  推理配置"
echo "  模型: $MODEL_PATH"
echo "  输出: $OUTPUT"
echo "  TTA:  $USE_TTA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$USE_TTA" = "true" ]; then
    python predict.py --model "$MODEL_PATH" --output "$OUTPUT" --use_tta
else
    python predict.py --model "$MODEL_PATH" --output "$OUTPUT"
fi

echo ""
echo "✅ 推理完成！结果保存至: $OUTPUT"
echo "📊 结果预览:"
head -6 "$OUTPUT"
