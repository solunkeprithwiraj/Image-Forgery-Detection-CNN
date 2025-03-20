@echo off
echo Installing and Running Image Forgery Detection React Frontend

echo Checking for Node.js installation...
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Node.js is not installed. Please install Node.js from https://nodejs.org/
    exit /b 1
)

echo Node.js found. Moving to frontend directory...
cd frontend

echo Installing dependencies (this may take a few minutes)...
call npm install

if %ERRORLEVEL% neq 0 (
    echo Error installing dependencies.
    exit /b 1
)

echo Dependencies installed successfully.
echo Starting the React development server...
echo.
echo Your application will be available at: http://localhost:3000
echo.
echo FEATURES:
echo - 3D visualizations of forgery detection results
echo - Interactive UI with smooth animations
echo - Advanced image uploading and analysis
echo - Responsive design for all devices
echo.
echo Press Ctrl+C to stop the server when finished.
echo.

call npm start

cd .. 