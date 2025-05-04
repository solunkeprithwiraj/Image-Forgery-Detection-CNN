@echo off
REM Batch script to visualize ResNet model results on CASIA2 dataset

REM Create output directory for visualization results
mkdir "data\output\visualization" 2>nul
mkdir "data\output\visualization\resnet18_casia2" 2>nul

REM Visualize ResNet18 model results
echo Visualizing ResNet18 model results on CASIA2 dataset...
python models\resnet\visualize_results.py ^
    --data_dir data\processed ^
    --model_path data\output\models\resnet18_best.pth ^
    --model_type resnet18 ^
    --output_dir data\output\visualization\resnet18_casia2 ^
    --num_samples 10

echo Visualization complete! 