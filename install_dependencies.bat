@echo off
echo Installing dependencies for the Improved Image Forgery Detection Model...

:: Install core dependencies
pip install torch==1.8.1 torchvision==0.9.1
pip install scikit-learn==0.24.1 scikit-image==0.18.1 scipy==1.6.0
pip install numpy==1.19.5 pandas==1.2.1 matplotlib==3.3.3 seaborn==0.11.1
pip install opencv-python==4.5.1.48 joblib==1.0.0

:: Install additional dependencies for the improved model
pip install xgboost==1.5.0
pip install imbalanced-learn==0.8.0
pip install tqdm==4.62.3
pip install efficientnet-pytorch==0.7.1
pip install albumentations==1.1.0
pip install tensorboard==2.8.0

echo All dependencies installed successfully!
echo You can now run the improved model using run_improved_model.bat
pause 