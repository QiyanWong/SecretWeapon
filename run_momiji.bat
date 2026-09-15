@echo off
chcp 65001 > NUL
echo 正在启动 冒险岛纸娃娃模拟器 Momiji...
cd /d "%~dp0momiji\output"
start "" "momiji.exe"
