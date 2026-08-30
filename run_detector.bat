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

:: Intelligent Python discovery
set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
if "%PYTHON_EXE%"=="" if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if "%PYTHON_EXE%"=="" (
    where.exe python >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if "%PYTHON_EXE%"=="" (
    where.exe py >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=py"
)

if "%PYTHON_EXE%"=="" (
    echo =======================================================
    echo [ERROR] Python not found!
    echo Please make sure Python 3.10+ is installed and on PATH.
    echo =======================================================
    pause
    exit /B 1
)

echo [OK] Starting YOLO detector and tracking engine...
"%PYTHON_EXE%" yolo_detector.py
if errorlevel 1 (
    echo [ERROR] Execution failed.
    pause
)
