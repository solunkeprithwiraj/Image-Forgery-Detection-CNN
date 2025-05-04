@echo off
REM Batch script to train ResNet models on CASIA2 dataset

REM Create output directories
mkdir "data\output\models" 2>nul
mkdir "data\output\resnet18_casia2" 2>nul

REM Train ResNet18 model
echo Training ResNet18 model on CASIA2 dataset with GPU acceleration...
python models\resnet\train_resnet.py ^
    --data_dir data\processed ^
    --output_dir data\output ^
    --model_type resnet18 ^
    --batch_size 32 ^
    --epochs 30 ^
    --learning_rate 0.001 ^
    --optimizer adam ^
    --scheduler step ^
    --num_workers 2 ^
    --use_srm ^
    --use_gpu ^
    --pin_memory

echo Training complete! 