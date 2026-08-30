@echo off
cd /d "%~dp0"
chcp 65001 > NUL
echo Converting AnyLabeling/Labelme JSON to YOLO txt format...
python convert_json_to_yolo.py
if errorlevel 1 (
    echo [ERROR] Conversion failed!
) else (
    echo.
    echo Conversion Completed!
)
pause
