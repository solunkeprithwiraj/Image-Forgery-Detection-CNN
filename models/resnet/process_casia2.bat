@echo off
REM Batch script to preprocess CASIA2 dataset for ResNet training

REM Create output directories
mkdir "data\processed\authentic" 2>nul
mkdir "data\processed\tampered" 2>nul
mkdir "data\output\models" 2>nul

REM Set parameters
set INPUT_DIR=data\casia2
set OUTPUT_DIR=data\processed
set RESIZE=256
set CROP_SIZE=224

echo Processing CASIA2 dataset...
python models\resnet\process_casia2.py ^
    --input_dir %INPUT_DIR% ^
    --output_dir %OUTPUT_DIR% ^
    --resize %RESIZE% ^
    --crop_size %CROP_SIZE%

echo Preprocessing complete! 