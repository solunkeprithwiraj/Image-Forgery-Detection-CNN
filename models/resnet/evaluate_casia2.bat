@echo off
REM Batch script to evaluate ResNet models on CASIA2 dataset

REM Create output directory for evaluation results
mkdir "data\output\evaluation" 2>nul
mkdir "data\output\evaluation\resnet18_casia2" 2>nul

REM Evaluate ResNet18 model
echo Evaluating ResNet18 model on CASIA2 dataset...
python models\resnet\evaluate_resnet.py ^
    --data_dir data\processed ^
    --model_path data\output\models\resnet18_best.pth ^
    --model_type resnet18 ^
    --batch_size 32 ^
    --output_dir data\output\evaluation\resnet18_casia2

echo Evaluation complete! 