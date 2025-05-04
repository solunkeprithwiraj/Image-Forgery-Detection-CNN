#!/bin/bash
# Script to train ResNet models for image forgery detection

# Create output directories
mkdir -p data/output/models
mkdir -p data/output/resnet18
mkdir -p data/output/resnet34
mkdir -p data/output/resnet50

# Train ResNet18 model
echo "Training ResNet18 model..."
python models/resnet/train_resnet.py \
    --data_dir data/processed \
    --output_dir data/output \
    --model_type resnet18 \
    --batch_size 32 \
    --epochs 30 \
    --learning_rate 0.001 \
    --optimizer adam

# Train ResNet34 model
# echo "Training ResNet34 model..."
# python models/resnet/train_resnet.py \
#    --data_dir data/processed \
#    --output_dir data/output \
#    --model_type resnet34 \
#    --batch_size 32 \
#    --epochs 30 \
#    --learning_rate 0.001 \
#    --optimizer adam

# Train ResNet50 model
# echo "Training ResNet50 model..."
# python models/resnet/train_resnet.py \
#    --data_dir data/processed \
#    --output_dir data/output \
#    --model_type resnet50 \
#    --batch_size 16 \
#    --epochs 30 \
#    --learning_rate 0.0005 \
#    --optimizer adam

echo "Training complete!" 