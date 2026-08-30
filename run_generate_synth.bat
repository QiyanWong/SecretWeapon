@echo off
cd /d "%~dp0"
chcp 65001 > NUL

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
    echo [ERROR] Python not found! Please install Python 3.10+ and add it to PATH.
    pause
    exit /B 1
)

echo [OK] 正在启动 2D 透明素材自动合成生成器...
"%PYTHON_EXE%" generate_synthetic_dataset.py --num 50
pause
