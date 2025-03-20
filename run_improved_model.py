"""
Run script for the improved image forgery detection model.
This script can be run directly from the project root.
"""
import os
import sys

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main function from the demo script
from src.improved_model_demo import main

if __name__ == "__main__":
    print("Starting the improved image forgery detection model demo...")
    main()
    print("Demo completed successfully!") 