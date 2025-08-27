#!/bin/bash

echo "🚀 Deploying Improved ONNX Detection to Google Cloud"
echo "===================================================="

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Not in backend directory. Please run from backend folder."
    exit 1
fi

echo "📋 Checking improved detection setup..."

# 1. Check ONNX models
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

# 2. Check improved ONNX script
echo "📋 Checking improved ONNX detection script..."
if [ ! -f "run-solar-panel-and-pool-detection-improved.py" ]; then
    echo "❌ Missing: run-solar-panel-and-pool-detection-improved.py"
    exit 1
fi
echo "✅ Improved ONNX detection script found!"

# 3. Check ONNX requirements
echo "📋 Checking ONNX requirements..."
if [ ! -f "requirements-onnx-only.txt" ]; then
    echo "❌ Missing: requirements-onnx-only.txt"
    exit 1
fi
echo "✅ ONNX requirements found!"

# 4. Check server configuration
echo "📋 Checking server configuration..."
if ! grep -q "run-solar-panel-and-pool-detection-improved.py" server.prod.js; then
    echo "❌ Server not updated for improved ONNX script"
    exit 1
fi
echo "✅ Server configured for improved ONNX!"

# 5. Test improved ONNX functionality locally
echo "📋 Testing improved ONNX functionality..."
if ! python3 debug-ml-detection.py > /dev/null 2>&1; then
    echo "❌ ONNX test failed locally"
    exit 1
fi
echo "✅ ONNX test passed locally!"

echo ""
echo "🎯 Improved Detection Features:"
echo "   - Lower confidence threshold (0.3 instead of 0.5)"
echo "   - Duplicate detection filtering (IoU-based)"
echo "   - Better detection sensitivity"
echo "   - More detailed response format"
echo ""

# 6. Check deployment platform
echo "📋 Checking deployment platform..."

# Check for Google Cloud
if [ -f "Dockerfile" ] && [ -f "cloudbuild.yaml" ]; then
    echo "✅ Google Cloud deployment files found"
    echo "🚀 Deploying to Google Cloud Run..."
    echo ""
    echo "📋 Deployment options:"
    echo "   1. Manual deployment: gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/riffai-solar-backend"
    echo "   2. Cloud Build automation: Push to google-cloud branch"
    echo "   3. Local Docker test: docker build -t ml-backend . && docker run -p 8080:8080 ml-backend"
    echo ""
    echo "🎯 Recommended: Push to google-cloud branch for automatic Cloud Build deployment"
    
elif [ -f "railway.json" ]; then
    echo "✅ Railway deployment found"
    echo "🚀 Deploying to Railway..."
    if command -v railway &> /dev/null; then
        railway up
    else
        echo "⚠️  Railway CLI not found. Install with: npm install -g @railway/cli"
        echo "🚀 Then run: railway up"
    fi
    
else
    echo "❓ Unknown deployment platform"
    echo "📋 Available options:"
    echo "   1. Google Cloud: Push to google-cloud branch"
    echo "   2. Railway: railway up"
    echo "   3. Docker: docker build && docker run"
    echo "   4. Local: npm run start:prod"
fi

echo ""
echo "🎯 Next steps:"
echo "   1. Deploy the improved detection script"
echo "   2. Test with the same image that gave 8 detections"
echo "   3. Expect 12-16 detections (matching local backend)"
echo ""
echo "📊 Expected improvements:"
echo "   - More solar panels detected (confidence > 0.3)"
echo "   - Better duplicate filtering"
echo "   - Detection count closer to local backend (16 vs 8)"
echo "   - Same fast performance (2-3 seconds)" 