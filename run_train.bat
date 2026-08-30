@echo off
chcp 65001 > NUL
echo 正在启动 YOLOv8 模型训练...
"C:\Users\Chico\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" train_yolo.py --epochs 50 --batch 16
pause
