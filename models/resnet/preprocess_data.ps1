# PowerShell script to preprocess data for ResNet training

# Create output directories if they don't exist
New-Item -Force -Path "data\processed\authentic", "data\processed\tampered", "data\output\models" -ItemType Directory | Out-Null

# Set parameters
$inputDir = "data\raw"
$outputDir = "data\processed"
$resize = 256
$cropSize = 224

Write-Host "Running preprocessing script..."
Write-Host "Processing data from $inputDir to $outputDir"

# Run the Python preprocessing script
python models\resnet\preprocess_data.py --input_dir $inputDir --output_dir $outputDir --resize $resize --crop_size $cropSize

Write-Host "Preprocessing complete!" 