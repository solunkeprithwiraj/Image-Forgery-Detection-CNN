@echo off
setlocal

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install it before running this script.
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories if they don't exist
if not exist uploads mkdir uploads
if not exist outputs mkdir outputs

REM Run the Flask application
echo Starting the Flask application...
python run.py 