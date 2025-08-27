#!/bin/bash

echo "🚀 Deploying ONNX Fix to Current Backend"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Not in backend directory. Please run from backend folder."
    exit 1
fi

echo "📋 Checking current setup..."

# 1. Check if ONNX models exist
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

# 2. Check if ONNX script exists
echo "📋 Checking ONNX detection script..."
if [ ! -f "run-solar-panel-and-pool-detection-onnx-only.py" ]; then
    echo "❌ Missing: run-solar-panel-and-pool-detection-onnx-only.py"
    exit 1
fi
echo "✅ ONNX detection script found!"

# 3. Check if ONNX requirements exist
echo "📋 Checking ONNX requirements..."
if [ ! -f "requirements-onnx-only.txt" ]; then
    echo "❌ Missing: requirements-onnx-only.txt"
    exit 1
fi
echo "✅ ONNX requirements found!"

# 4. Test ONNX functionality locally
echo "📋 Testing ONNX functionality..."
if ! python3 debug-ml-detection.py > /dev/null 2>&1; then
    echo "❌ ONNX test failed locally"
    exit 1
fi
echo "✅ ONNX test passed locally!"

# 5. Check what platform you're currently using
echo "📋 Checking current deployment platform..."

# Check for Railway
if command -v railway &> /dev/null; then
    echo "✅ Railway CLI found"
    echo "🚀 Deploying to Railway..."
    railway up
elif [ -f "railway.json" ]; then
    echo "📁 Railway config found but CLI not installed"
    echo "⚠️  Install Railway CLI: npm install -g @railway/cli"
    echo "🚀 Then run: railway up"
elif [ -f "Dockerfile" ]; then
    echo "🐳 Dockerfile found"
    echo "🚀 Build and deploy with Docker"
    echo "   docker build -t ml-backend ."
    echo "   docker run -p 10000:10000 ml-backend"
else
    echo "❓ Unknown deployment platform"
    echo "📋 Available options:"
    echo "   1. Railway: railway up"
    echo "   2. Docker: docker build && docker run"
    echo "   3. Local: npm run start:prod"
fi

echo ""
echo "🎯 Next steps:"
echo "   1. Wait for deployment to complete"
echo "   2. Test the health endpoint: curl YOUR_URL/health"
echo "   3. Test ML detection with an image upload"
echo ""
echo "📊 Expected results:"
echo "   - No more PyTorch compatibility errors"
echo "   - ONNX models load in 3-5 seconds"
echo "   - Fast and reliable ML detection" 