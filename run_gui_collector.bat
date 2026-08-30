@echo off
cd /d "%~dp0"
chcp 65001 > NUL
echo 正在启动 图像采集助手 GUI...
python gui_collector.py
if errorlevel 1 (
    echo [ERROR] 运行失败，请确认 Python 环境与依赖已正确安装。
)
pause
