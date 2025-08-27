#!/bin/bash

# 🚀 Quick Start Solar Panel Detection Training
# =============================================

echo "🚀 Starting Solar Panel Detection Training Pipeline"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "data.yaml" ]; then
    echo "❌ data.yaml not found. Please run this script in the dataset directory."
    echo "💡 You can download the dataset first by running:"
    echo "   curl -L 'https://universe.roboflow.com/ds/dCRuFFJd7j?key=9SOXLBGP8y' > roboflow.zip"
    echo "   unzip roboflow.zip"
    echo "   rm roboflow.zip"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p training_data
mkdir -p logs

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Install requirements
echo "🔧 Installing requirements..."
pip3 install ultralytics opencv-python matplotlib pillow numpy

# Check if training scripts exist
if [ ! -f "train_simple_solar.py" ]; then
    echo "❌ Training script not found. Please ensure train_simple_solar.py exists."
    exit 1
fi

# Start training
echo "🚀 Starting training..."
echo "💡 This will take 2-4 hours depending on your system."
echo "💡 You can monitor progress in the logs/ directory."

# Run training with logging
python3 train_simple_solar.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "📁 Output files:"
    echo "   - Trained model: solar_detection_simple/yolo11n_solar/weights/best.pt"
    echo "   - ONNX model: models/best-solar-panel-improved.onnx"
    echo "   - Training logs: logs/"
    echo ""
    echo "🧪 Next steps:"
    echo "   1. Test the new model: python3 test_new_model.py"
    echo "   2. Compare with current models"
    echo "   3. Update your backend to use the new model"
    echo ""
    echo "💡 You can also run advanced training with:"
    echo "   python3 train_improved_solar_detection.py"
else
    echo ""
    echo "❌ Training failed. Check the logs for details."
    echo "💡 Common issues:"
    echo "   - Insufficient memory: Reduce batch size"
    echo "   - Missing dependencies: Run pip install -r requirements.txt"
    echo "   - Dataset issues: Check data.yaml format"
fi 