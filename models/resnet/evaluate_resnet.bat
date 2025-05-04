@echo off
REM Batch script to evaluate ResNet models for image forgery detection

REM Create output directory for evaluation results
mkdir "data\output\evaluation" 2>nul

REM Evaluate ResNet18 model
echo Evaluating ResNet18 model...
python models\resnet\evaluate_resnet.py ^
    --data_dir data\processed ^
    --model_path data\output\models\resnet18_best.pth ^
    --model_type resnet18 ^
    --batch_size 32 ^
    --output_dir data\output\evaluation\resnet18

REM Uncomment to evaluate ResNet34 model
REM echo Evaluating ResNet34 model...
REM python models\resnet\evaluate_resnet.py ^
REM    --data_dir data\processed ^
REM    --model_path data\output\models\resnet34_best.pth ^
REM    --model_type resnet34 ^
REM    --batch_size 32 ^
REM    --output_dir data\output\evaluation\resnet34

REM Uncomment to evaluate ResNet50 model
REM echo Evaluating ResNet50 model...
REM python models\resnet\evaluate_resnet.py ^
REM    --data_dir data\processed ^
REM    --model_path data\output\models\resnet50_best.pth ^
REM    --model_type resnet50 ^
REM    --batch_size 16 ^
REM    --output_dir data\output\evaluation\resnet50

echo Evaluation complete! 