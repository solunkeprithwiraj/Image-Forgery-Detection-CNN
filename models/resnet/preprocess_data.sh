#!/bin/bash
# Shell script to preprocess data for ResNet training

# Create output directories
mkdir -p data/processed/authentic data/processed/tampered data/output/models

# Set parameters
INPUT_DIR="data/raw"
OUTPUT_DIR="data/processed"
RESIZE=256
CROP_SIZE=224

echo "Running preprocessing script..."
echo "Processing data from $INPUT_DIR to $OUTPUT_DIR"

# Run the Python preprocessing script
python models/resnet/preprocess_data.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --resize $RESIZE \
    --crop_size $CROP_SIZE

echo "Preprocessing complete!" 