@echo off
echo Installing frontend dependencies...
cd react-frontend

echo Installing chart.js and react-chartjs-2...
npm install chart.js@4.2.1 react-chartjs-2@5.2.0 --legacy-peer-deps

echo Installing framer-motion...
npm install framer-motion@10.2.3 --legacy-peer-deps

echo Installing react-lottie...
npm install react-lottie@1.2.3 --legacy-peer-deps

echo Installing react-confetti and react-use...
npm install react-confetti@6.1.0 react-use@17.4.0 --legacy-peer-deps

echo Installing react-typed and react-scroll...
npm install react-typed@1.2.0 react-scroll@1.8.9 --legacy-peer-deps

echo Installing react-dropzone...
npm install react-dropzone@14.2.3 --legacy-peer-deps

echo All dependencies installed successfully!
echo.
echo To start the application, run .\start_app.bat
pause 