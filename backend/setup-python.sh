#!/bin/bash

echo "Setting up Python environment for RiffAI Solar Detection..."

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Found Python3: $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "Found Python: $(python --version)"
else
    echo "Error: Python not found. Please install Python 3.8+ first."
    exit 1
fi

# Check pip
echo "Checking pip installation..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "Error: pip not found. Please install pip first."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
if [ -f "requirements-render.txt" ]; then
    echo "Using Render-optimized requirements..."
    $PYTHON_CMD -m pip install -r requirements-render.txt
else
    echo "Using standard requirements..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Verify installation
echo "Verifying installation..."
$PYTHON_CMD -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
except ImportError:
    print('PyTorch not installed')

try:
    import ultralytics
    print(f'Ultralytics version: {ultralytics.__version__}')
except ImportError:
    print('Ultralytics not installed')

try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except ImportError:
    print('OpenCV not installed')

try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
except ImportError:
    print('NumPy not installed')
"

echo "Python environment setup completed!" 
