@echo off
chcp 65001 > NUL
echo 正在启动 冒险岛 2D 透明素材自动合成生成器...
"C:\Users\Chico\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" generate_synthetic_dataset.py --num 50
pause
