#!/usr/bin/env bash
# ============================================
#  AIGC 文本检测器 v7 - Linux/macOS 打包脚本
#  用法: bash build_linux.sh
#  依赖: python3 + pip (pyinstaller 自动安装)
# ============================================
set -e
echo "[1/3] 检查 Python..."
command -v python3 >/dev/null || { echo "未找到 python3"; exit 1; }

echo "[2/3] 安装 PyInstaller..."
pip3 install pyinstaller --quiet 2>/dev/null || pip install pyinstaller --quiet

echo "[3/3] 打包中..."
pyinstaller -F -w -n AIGC-Detector-v7 --clean \
    --hidden-import=aigc_detector \
    python/aigc_gui.py

echo ""
echo "[完成] 已生成: dist/AIGC-Detector-v7"
echo "运行: ./dist/AIGC-Detector-v7"