"""
Script to preprocess data for ResNet training.
This script organizes and transforms images for training the ResNet models.
"""

import os
import argparse
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess data for ResNet training')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing the raw dataset')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Directory to save processed data')
    parser.add_argument('--train_split', type=float, default=0.8,
                      help='Percentage of data to use for training')
    parser.add_argument('--resize', type=int, default=256,
                      help='Size to resize images')
    parser.add_argument('--crop_size', type=int, default=224,
                      help='Size to center crop images')
    return parser.parse_args()

def resize_and_crop(img, resize_size, crop_size):
    """Resize and center crop an image."""
    # Resize the image
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    
    # Calculate center crop coordinates
    width, height = img.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    # Crop the image
    img = img.crop((left, top, right, bottom))
    return img

def process_images(input_dir, output_dir, resize_size, crop_size):
    """Process all images in the input directory."""
    # Create output directories
    authentic_dir = os.path.join(output_dir, 'authentic')
    tampered_dir = os.path.join(output_dir, 'tampered')
    os.makedirs(authentic_dir, exist_ok=True)
    os.makedirs(tampered_dir, exist_ok=True)
    
    # Process authentic images
    authentic_path = os.path.join(input_dir, 'authentic')
    if os.path.exists(authentic_path):
        print(f"Processing authentic images from {authentic_path}")
        for filename in tqdm(os.listdir(authentic_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                input_path = os.path.join(authentic_path, filename)
                output_path = os.path.join(authentic_dir, filename)
                
                try:
                    img = Image.open(input_path).convert('RGB')
                    img = resize_and_crop(img, resize_size, crop_size)
                    img.save(output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
    
    # Process tampered images
    tampered_path = os.path.join(input_dir, 'tampered')
    if os.path.exists(tampered_path):
        print(f"Processing tampered images from {tampered_path}")
        for filename in tqdm(os.listdir(tampered_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                input_path = os.path.join(tampered_path, filename)
                output_path = os.path.join(tampered_dir, filename)
                
                try:
                    img = Image.open(input_path).convert('RGB')
                    img = resize_and_crop(img, resize_size, crop_size)
                    img.save(output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

def main():
    """Main function to preprocess data."""
    args = parse_args()
    
    print(f"Preprocessing data from {args.input_dir} to {args.output_dir}")
    process_images(args.input_dir, args.output_dir, args.resize, args.crop_size)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main() 