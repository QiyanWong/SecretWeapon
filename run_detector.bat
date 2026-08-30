@echo off
cd /d "%~dp0"
chcp 65001 > NUL

:: Check for Administrator privileges
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo =======================================================
    echo [ERROR] Admin privileges required!
    echo Please right-click run_detector.bat and select "Run as Administrator"!
    echo Otherwise game keystroke simulation may be blocked.
    echo =======================================================
    pause
    exit /B
)

echo [OK] Starting YOLO detector and tracking engine...
python yolo_detector.py
if errorlevel 1 (
    echo [ERROR] Failed to run python. Please make sure Python 3.10+ is installed and on PATH.
    pause
)
