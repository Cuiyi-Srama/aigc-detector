@echo off
chcp 65001 >nul
REM ============================================
REM  AIGC 文本检测器 v7 - Windows 一键打包脚本
REM  用法: 双击运行 build_windows.bat
REM  依赖: 需先安装 Python 3.8+ (勾选 Add to PATH)
REM ============================================
title AIGC Detector - Windows Build

echo [1/3] 检查 Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [2/3] 安装 PyInstaller...
pip install pyinstaller --quiet

echo [3/3] 打包中 (GUI 版)...
pyinstaller -F -w -n AIGC-Detector-v7 --clean ^
    --hidden-import=aigc_detector ^
    python\aigc_gui.py

if errorlevel 1 (
    echo [错误] 打包失败，请检查上方日志
    pause
    exit /b 1
)

echo.
echo [完成] 已生成: dist\AIGC-Detector-v7.exe
echo 双击 dist\AIGC-Detector-v7.exe 即可运行，无需 Python 环境。
pause