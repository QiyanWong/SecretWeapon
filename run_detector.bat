@echo off
cd /d "%~dp0"

:: Check for Administrator privileges
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo =======================================================
    echo [ERROR] Admin privileges required!
    echo Please right-click run_detector.bat and select "Run as Administrator"!
    echo Otherwise the game's anti-cheat will block fake keystrokes.
    echo =======================================================
    pause
    exit /B
)

echo [OK] Starting YOLO detector and tracking engine...
"C:\Users\Chico\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" yolo_detector.py
pause
