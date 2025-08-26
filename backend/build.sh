#!/bin/bash

echo "Starting build process for Render..."

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    
    # Check if Python3 is available
    if command -v python3 &> /dev/null; then
        echo "Python3 found, installing dependencies..."
        python3 -m pip install -r requirements.txt
    elif command -v python &> /dev/null; then
        echo "Python found, installing dependencies..."
        python -m pip install -r requirements.txt
    else
        echo "Warning: Python not found, skipping Python dependencies"
        echo "Set DISABLE_ML_DETECTION=true in environment variables"
    fi
else
    echo "No requirements.txt found, skipping Python dependencies"
fi

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p annotated_images
mkdir -p models

# Set permissions
echo "Setting permissions..."
chmod 755 uploads
chmod 755 annotated_images
chmod 755 models

# Check if model files exist
if [ -f "best-solar-panel.pt" ] && [ -f "pool-best.pt" ]; then
    echo "ML model files found"
    # Copy models to models directory if they exist
    cp best-solar-panel.pt models/ 2>/dev/null || echo "Could not copy solar panel model"
    cp pool-best.pt models/ 2>/dev/null || echo "Could not copy pool model"
else
    echo "Warning: ML model files not found. Set DISABLE_ML_DETECTION=true in environment variables."
fi

# Check Python installation
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    python3 --version
    python3 -c "import sys; print('Python path:', sys.executable)"
elif command -v python &> /dev/null; then
    python --version
    python -c "import sys; print('Python path:', sys.executable)"
else
    echo "Python not found in PATH"
fi

echo "Build completed successfully!" 