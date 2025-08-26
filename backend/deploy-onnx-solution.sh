#!/bin/bash

echo "🚀 Deploying ONNX-Only ML Backend to Railway"
echo "=============================================="

# Check if ONNX models exist
echo "📋 Checking ONNX models..."
if [ ! -f "models/best-solar-panel.onnx" ]; then
    echo "❌ Missing: models/best-solar-panel.onnx"
    exit 1
fi

if [ ! -f "models/pool-best.onnx" ]; then
    echo "❌ Missing: models/pool-best.onnx"
    exit 1
fi

echo "✅ ONNX models found!"

# Check if ONNX script exists
echo "📋 Checking ONNX detection script..."
if [ ! -f "run-solar-panel-and-pool-detection-onnx-only.py" ]; then
    echo "❌ Missing: run-solar-panel-and-pool-detection-onnx-only.py"
    exit 1
fi

echo "✅ ONNX detection script found!"

# Check if ONNX requirements exist
echo "📋 Checking ONNX requirements..."
if [ ! -f "requirements-onnx-only.txt" ]; then
    echo "❌ Missing: requirements-onnx-only.txt"
    exit 1
fi

echo "✅ ONNX requirements found!"

# Check if server is updated
echo "📋 Checking server configuration..."
if ! grep -q "run-solar-panel-and-pool-detection-onnx-only.py" server.prod.js; then
    echo "❌ Server not updated for ONNX script"
    exit 1
fi

echo "✅ Server configured for ONNX!"

# Check Railway CLI
echo "📋 Checking Railway CLI..."
if ! command -v railway &> /dev/null; then
    echo "⚠️  Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

echo "✅ Railway CLI ready!"

# Deploy to Railway
echo "🚀 Deploying to Railway..."
echo "   This will use ONNX-only requirements (no PyTorch)"
echo "   Models: best-solar-panel.onnx, pool-best.onnx"
echo "   Script: run-solar-panel-and-pool-detection-onnx-only.py"
echo ""

railway up

echo ""
echo "🎉 Deployment complete!"
echo "📊 Your ONNX-only ML backend should now work without PyTorch errors!"
echo "🔍 Test by uploading an image to the /detect endpoint" 