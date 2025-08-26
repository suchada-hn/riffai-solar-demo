#!/bin/bash

echo "🚀 Railway ONNX-Only ML Backend Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    print_error "Not in backend directory. Please run from backend folder."
    exit 1
fi

echo "📋 Pre-deployment checks..."

# 1. Check ONNX models
print_status "Checking ONNX models..."
if [ ! -f "models/best-solar-panel.onnx" ]; then
    print_error "Missing: models/best-solar-panel.onnx"
    exit 1
fi

if [ ! -f "models/pool-best.onnx" ]; then
    print_error "Missing: models/pool-best.onnx"
    exit 1
fi
print_status "ONNX models found"

# 2. Check ONNX script
print_status "Checking ONNX detection script..."
if [ ! -f "run-solar-panel-and-pool-detection-onnx-only.py" ]; then
    print_error "Missing: run-solar-panel-and-pool-detection-onnx-only.py"
    exit 1
fi
print_status "ONNX detection script found"

# 3. Check ONNX requirements
print_status "Checking ONNX requirements..."
if [ ! -f "requirements-onnx-only.txt" ]; then
    print_error "Missing: requirements-onnx-only.txt"
    exit 1
fi
print_status "ONNX requirements found"

# 4. Check server configuration
print_status "Checking server configuration..."
if ! grep -q "run-solar-panel-and-pool-detection-onnx-only.py" server.prod.js; then
    print_error "Server not configured for ONNX script"
    exit 1
fi
print_status "Server configured for ONNX"

# 5. Check Railway configuration
print_status "Checking Railway configuration..."
if ! grep -q "requirements-onnx-only.txt" railway.json; then
    print_error "Railway not configured for ONNX requirements"
    exit 1
fi
print_status "Railway configured for ONNX"

# 6. Test ONNX locally
print_status "Testing ONNX models locally..."
if ! python3 test-onnx-simple.py > /dev/null 2>&1; then
    print_error "ONNX test failed locally"
    exit 1
fi
print_status "ONNX test passed locally"

# 7. Check Railway CLI
print_status "Checking Railway CLI..."
if ! command -v railway &> /dev/null; then
    print_warning "Railway CLI not found. Installing..."
    npm install -g @railway/cli
    if ! command -v railway &> /dev/null; then
        print_error "Failed to install Railway CLI"
        exit 1
    fi
fi
print_status "Railway CLI ready"

# 8. Check if logged in to Railway
print_status "Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    print_warning "Not logged in to Railway. Please login:"
    railway login
    if ! railway whoami &> /dev/null; then
        print_error "Railway login failed"
        exit 1
    fi
fi
print_status "Railway authenticated"

echo ""
echo "🚀 All checks passed! Deploying to Railway..."
echo ""

# Deploy to Railway
print_status "Starting Railway deployment..."
railway up

# Check deployment status
if [ $? -eq 0 ]; then
    echo ""
    print_status "Deployment completed successfully!"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Wait for Railway to finish building (2-3 minutes)"
    echo "   2. Test the health endpoint: curl https://your-app.railway.app/health"
    echo "   3. Test ML detection with an image upload"
    echo ""
    echo "📊 Expected results:"
    echo "   - No more PyTorch compatibility errors"
    echo "   - ONNX models load in 3-5 seconds"
    echo "   - Fast and reliable ML detection"
else
    print_error "Railway deployment failed"
    exit 1
fi 