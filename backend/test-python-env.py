#!/usr/bin/env python3
"""
Simple Python environment test script for Render deployment
This helps diagnose what's available in the Python environment
"""

import sys
import os

def test_imports():
    """Test importing various packages"""
    print("=== Python Environment Test ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Test basic imports
    basic_packages = ['os', 'sys', 'json', 'pathlib']
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✓ {package} - OK")
        except ImportError as e:
            print(f"✗ {package} - FAILED: {e}")
    
    # Test ML-related imports
    ml_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('ultralytics', 'Ultralytics'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow')
    ]
    
    print("\n=== ML Package Tests ===")
    for package, name in ml_packages:
        try:
            __import__(package)
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name} ({package}) - OK (v{version})")
        except ImportError as e:
            print(f"✗ {name} ({package}) - FAILED: {e}")
    
    # Test if we can access model files
    print("\n=== Model File Tests ===")
    model_files = ['best-solar-panel.pt', 'pool-best.pt']
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"✓ {model_file} - Found ({size} bytes)")
        else:
            print(f"✗ {model_file} - Not found")
    
    # Test directories
    print("\n=== Directory Tests ===")
    directories = ['uploads', 'annotated_images', 'models']
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}/ - Exists")
        else:
            print(f"✗ {directory}/ - Missing")

if __name__ == "__main__":
    test_imports()
    print("\n=== Test Complete ===") 