@echo off
chcp 65001 > NUL
echo Converting AnyLabeling/Labelme JSON to YOLO txt format...
"C:\Users\Chico\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" convert_json_to_yolo.py
echo.
echo Conversion Completed!
pause
