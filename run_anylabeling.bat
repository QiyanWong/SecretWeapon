@echo off
chcp 65001 > NUL
echo 正在启动 AnyLabeling 标注工具...
"C:\Users\Chico\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m anylabeling.app
pause
