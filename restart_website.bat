@echo off
echo Restarting Image Forgery Detection Website...
echo Stopping any running instances...
taskkill /f /im python.exe 2>nul
echo Starting Flask server at http://localhost:5000
echo FEATURES:
echo - Improved tampering localization with contour detection
echo - Multiple visualization options (heatmap, overlay, contour)
echo - Detailed analysis of tampered regions
cd website
python app.py 