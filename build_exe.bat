@echo off
chcp 65001 >nul
echo ========================================================
echo ⚡ SecretWeapon 一键 PyInstaller 单文件打包脚本 (Windows)
echo ========================================================
echo.

echo 1. 正在检查并安装必要的打包依赖...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo 2. 开始打包 yolo_detector.py 为单文件 Executable...
pyinstaller --noconfirm --onefile --windowed ^
  --name "SecretWeapon" ^
  --collect-all ultralytics ^
  --collect-all torch ^
  --add-data "dataset;dataset" ^
  --add-data "map;map" ^
  --add-data "best.pt;." ^
  --add-data "yolov8n.pt;." ^
  yolo_detector.py

echo.
echo ========================================================
if exist "dist\SecretWeapon.exe" (
    echo ✅ 打包成功！生成的单文件位于: dist\SecretWeapon.exe
) else (
    echo ❌ 打包失败，请检查上方控制台报错信息。
)
echo ========================================================
pause
