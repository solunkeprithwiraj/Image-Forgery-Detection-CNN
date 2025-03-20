@echo off
echo Starting Image Forgery Detection System (Full Stack)

echo Starting Flask backend server in a new window...
start cmd /k "title Flask Backend && echo Starting Flask Backend && cd website && python app.py"

echo Starting React frontend in a new window...
start cmd /k "title React Frontend && echo Starting React Frontend && cd frontend && npm start"

echo.
echo Backend will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:3000
echo.
echo To stop both servers, close their respective command windows.
echo. 