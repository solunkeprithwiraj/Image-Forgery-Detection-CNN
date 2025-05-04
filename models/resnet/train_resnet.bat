@echo off
REM Batch script to train ResNet models for image forgery detection

REM Create output directories
mkdir "data\output\models" 2>nul
mkdir "data\output\resnet18" 2>nul
mkdir "data\output\resnet34" 2>nul
mkdir "data\output\resnet50" 2>nul

REM Train ResNet18 model
echo Training ResNet18 model...
python models\resnet\train_resnet.py ^
    --data_dir data\processed ^
    --output_dir data\output ^
    --model_type resnet18 ^
    --batch_size 32 ^
    --epochs 30 ^
    --learning_rate 0.001 ^
    --optimizer adam

REM Train ResNet34 model (uncomment to train)
REM echo Training ResNet34 model...
REM python models\resnet\train_resnet.py ^
REM    --data_dir data\processed ^
REM    --output_dir data\output ^
REM    --model_type resnet34 ^
REM    --batch_size 32 ^
REM    --epochs 30 ^
REM    --learning_rate 0.001 ^
REM    --optimizer adam

REM Train ResNet50 model (uncomment to train)
REM echo Training ResNet50 model...
REM python models\resnet\train_resnet.py ^
REM    --data_dir data\processed ^
REM    --output_dir data\output ^
REM    --model_type resnet50 ^
REM    --batch_size 16 ^
REM    --epochs 30 ^
REM    --learning_rate 0.0005 ^
REM    --optimizer adam

echo Training complete! 