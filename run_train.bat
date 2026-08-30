@echo off
cd /d "%~dp0"
chcp 65001 > NUL
echo 正在启动 YOLOv8 模型训练...
python train_yolo.py --epochs 50 --batch 16
if errorlevel 1 (
    echo [ERROR] 运行失败，请确认 Python 环境与依赖已正确安装。
)
pause
