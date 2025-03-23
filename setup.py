#!/usr/bin/env python
"""
Setup script for Neural Image Enhancer.
This script creates the necessary directory structure and downloads required model files.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    print("Checking requirements...")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print("Error: requirements.txt not found.")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
        print("All requirements installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements.")
        return False

def create_directory_structure():
    """Create the necessary directory structure"""
    print("Creating directory structure...")
    
    # Create directories
    directories = [
        "models",
        "models/luts",
        "models/filters",
        "uploads",
        "results",
        "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created {directory}/")

def download_models():
    """Download pre-trained models"""
    print("Downloading models...")
    
    try:
        # Run the model downloader script
        subprocess.run([sys.executable, "download_models.py"], check=True)
        print("All models downloaded successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error: Failed to download models.")
        return False

def check_template_files():
    """Check if all template files exist"""
    print("Checking template files...")
    
    template_files = [
        "templates/base.html",
        "templates/index.html",
        "templates/enhance.html",
        "templates/result.html"
    ]
    
    missing_files = []
    for file in template_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: The following template files are missing: {', '.join(missing_files)}")
        return False
    
    print("All template files found.")
    return True

def check_python_files():
    """Check if all Python files exist"""
    print("Checking Python files...")
    
    python_files = [
        "app.py",
        "neural_enhancement_pipeline.py",
        "download_models.py",
        "model_adapter.py"
    ]
    
    missing_files = []
    for file in python_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: The following Python files are missing: {', '.join(missing_files)}")
        return False
    
    print("All Python files found.")
    return True

def run_tests():
    """Run basic tests to ensure everything is working"""
    print("Running basic tests...")
    
    # Test importing the neural enhancement pipeline
    try:
        from neural_enhancement_pipeline import NeuralEnhancementPipeline
        print("  Successfully imported NeuralEnhancementPipeline")
    except ImportError as e:
        print(f"  Error importing NeuralEnhancementPipeline: {str(e)}")
        return False
    
    # Test creating an instance of the pipeline
    try:
        pipeline = NeuralEnhancementPipeline(use_cuda=False)
        print("  Successfully created NeuralEnhancementPipeline instance")
    except Exception as e:
        print(f"  Error creating NeuralEnhancementPipeline instance: {str(e)}")
        return False
    
    print("All tests passed.")
    return True

def main():
    """Main function to run the setup"""
    print("=== Neural Image Enhancer Setup ===")
    
    # Steps
    steps = [
        ("Checking requirements", check_requirements),
        ("Creating directory structure", create_directory_structure),
        ("Downloading models", download_models),
        ("Checking template files", check_template_files),
        ("Checking Python files", check_python_files),
        ("Running tests", run_tests)
    ]
    
    # Run each step
    for step_name, step_function in steps:
        print(f"\n=== {step_name} ===")
        success = step_function()
        if not success:
            print(f"\nWarning: {step_name} did not complete successfully.")
            choice = input("Continue anyway? (y/n): ")
            if choice.lower() != 'y':
                print("Setup aborted.")
                return
    
    print("\n=== Setup Complete ===")
    print("Your Neural Image Enhancer is ready to use.")
    print("To start the web application, run: python app.py")

if __name__ == "__main__":
    main()