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
echo 2. 用系统 System32 中的新版 VC++ 运行库替换 PyQt5 自带的旧版 (避免 c10.dll WinError 1114)...
echo    (请确保本机已安装最新 vc_redist.x64.exe)
for /f "delims=" %%i in ('python -c "import PyQt5,os;print(os.path.dirname(PyQt5.__file__))"') do set "PYQT_DIR=%%i"
for /r "%PYQT_DIR%" %%f in (msvcp140*.dll vcruntime140*.dll) do (
    if exist "%SystemRoot%\System32\%%~nxf" (
        copy /y "%SystemRoot%\System32\%%~nxf" "%%f" >nul
        echo    已替换: %%f
    )
)

echo.
echo 3. 开始打包 yolo_detector.py 为单文件 Executable...
pyinstaller --noconfirm --onefile --windowed ^
  --name "SecretWeapon" ^
  --collect-all ultralytics ^
  --collect-all torch ^
  --collect-all torchvision ^
  --copy-metadata ultralytics ^
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
