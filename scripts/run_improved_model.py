"""
A simple script to run the improved model demo.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main function from the improved_model_demo module
from scripts.improved_model_demo import main

if __name__ == "__main__":
    print("Starting the improved image forgery detection model demo...")
    main()
    print("Demo completed successfully!") 