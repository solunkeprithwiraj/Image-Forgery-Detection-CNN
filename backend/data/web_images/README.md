# Web Images for Image Forgery Detection

This directory contains web images for fine-tuning the image forgery detection model. The model trained on CASIA2 dataset may not perform well on web images due to domain differences. The fine-tuning process helps the model adapt to web images.

## Directory Structure

- `authentic/`: Authentic (non-tampered) images from the web
- `tampered/`: Tampered/manipulated images from the web
- `downloads/`: Temporary storage for downloaded images

## How to Use This Dataset

Follow these steps to set up and use the web images dataset:

### 1. Setup the Dataset

Run the setup script to create the directory structure:

```bash
python scripts/setup_web_dataset.py
```

### 2. Collect Web Images

Either:
- Manually download images from the web (recommended)
- Use the automatic download feature (for demonstration only)

```bash
python scripts/setup_web_dataset.py --download --authentic 10 --tampered 10
```

### 3. Fine-tune the Model

Once you have collected web images, fine-tune the model using:

```bash
python scripts/fine_tune_web_model.py --casia_dir data/CASIA2 --web_dir data/web_images --epochs 10
```

### 4. Test the Fine-tuned Model

Test the fine-tuned model on new web images:

```bash
python scripts/test_web_images.py --dir path/to/test/images --visualize
```

Or test a single image:

```bash
python scripts/test_web_images.py --image path/to/image.jpg --visualize
```

## Image Collection Guidelines

For best results, collect images that represent real-world scenarios:

### Authentic Images
- Original, unedited photos
- Various sources (social media, news sites, personal photos)
- Different lighting conditions and quality levels
- Various content (landscapes, portraits, objects)

### Tampered Images
- Images with clear manipulation
- Include various manipulation types:
  - Copy-paste forgeries
  - Splicing
  - Object removal
  - Face swaps
  - Content-aware fill

## Improving Model Performance

To further improve model performance:

1. Add more diverse web images
2. Increase the number of training epochs
3. Try different learning rates (--learning_rate parameter)
4. Adjust the domain adaptation weight (lambda_domain in fine_tune_web_model.py)
5. Combine with other techniques like ensemble methods

## Troubleshooting

- If the model still performs poorly, try collecting more web images that match your specific use case
- If certain image types cause issues, add more of those to the training set
- If training is unstable, reduce the learning rate or use early stopping
- For memory issues, reduce batch size (--batch_size parameter) 