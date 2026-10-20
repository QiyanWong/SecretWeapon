@echo off
cd /d "%~dp0"

:: Load user-specific runtime config if present
if exist "%~dp0runtime_config.bat" call "%~dp0runtime_config.bat"

:: Locate Python executable
if "%PYTHON_EXE%"=="" if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
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
    echo Please make sure Python is installed and on PATH.
    echo =======================================================
    pause
    exit /B 1
)

:: Run Map Pipeline Manager in interactive mode or with passed arguments
"%PYTHON_EXE%" map_pipeline_manager.py %*
if errorlevel 1 (
    echo.
    echo =======================================================
    echo [ERROR] Program exited with an error.
    echo =======================================================
    pause
)
