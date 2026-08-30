@echo off
cd /d "%~dp0"
chcp 65001 > NUL
echo 正在启动 AnyLabeling 标注工具...
python -m anylabeling.app
if errorlevel 1 (
    echo [ERROR] 未找到 anylabeling，请先执行: pip install anylabeling
)
pause
