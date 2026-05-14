#!/bin/bash
# ============================================================
#  环境搭建脚本 — 使用 uv + 国内镜像源
#
#  用法:
#    chmod +x setup.sh
#    ./setup.sh
# ============================================================

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  SVHN 街景字符识别 — 环境搭建"
echo "============================================"
echo ""

# 1. 检查 uv 是否已安装
if ! command -v uv &> /dev/null; then
    echo "[1/4] 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "[1/4] uv 已安装: $(uv --version)"
fi

# 2. 创建虚拟环境
echo "[2/4] 创建虚拟环境 (.venv)..."
uv venv .venv --python 3.10
echo "  虚拟环境创建完成"

# 3. 激活环境提示
echo ""
echo "  请手动激活环境:"
echo "    Linux/Mac:  source .venv/bin/activate"
echo "    Windows:    .venv\\Scripts\\activate"
echo ""

# 4. 安装依赖（使用清华镜像源）
echo "[3/4] 安装依赖（清华镜像源）..."
uv pip install -r requirements.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cu118

echo ""
echo "[4/4] 安装可选依赖..."
# wandb（实验追踪）
uv pip install wandb \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true

# ultralytics（YOLO，可选）
read -p "  是否安装 ultralytics (YOLO)? [y/N] " yn
if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then
    uv pip install ultralytics \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo ""
echo "============================================"
echo "  环境搭建完成!"
echo ""
echo "  下一步:"
echo "    1. source .venv/bin/activate"
echo "    2. python fix_labels.py"
echo "    3. python train.py --experiment baseline"
echo "============================================"
