#!/bin/bash

# 🚂 Optimized Railway Deployment Script
# This script deploys your backend to Railway with build optimizations

echo "🚂 Optimized Railway Deployment"
echo "================================"

# Check if ONNX models exist
if [ ! -d "models" ] || [ ! -f "models/best-solar-panel.onnx" ] || [ ! -f "models/pool-best.onnx" ]; then
    echo "❌ ONNX models not found!"
    echo "Please run the conversion script first:"
    echo "python3 convert_models_workaround.py"
    exit 1
fi

echo "✅ ONNX models found:"
echo "   - models/best-solar-panel.onnx"
echo "   - models/pool-best.onnx"

# Check if Railway CLI is available
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

echo "🔄 Preparing for Railway deployment..."

# Create optimized build files
echo "📦 Creating optimized build configuration..."

# Check if we need to initialize Railway project
if [ ! -f ".railway" ] && [ ! -d ".railway" ]; then
    echo "🚀 Initializing Railway project..."
    railway init
fi

echo "🔧 Deploying with optimized configuration..."
echo "   - Using Dockerfile.railway for faster builds"
echo "   - Excluding unnecessary files with .dockerignore"
echo "   - Multi-stage build optimization"

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment completed successfully!"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Get your Railway URL from the dashboard"
    echo "2. Test the /health endpoint"
    echo "3. Test ML detection endpoints"
    echo "4. Update your frontend configuration"
    echo ""
    echo "📚 Useful commands:"
    echo "   railway status          # Check deployment status"
    echo "   railway logs           # View deployment logs"
    echo "   railway open           # Open in browser"
    echo "   railway variables      # Manage environment variables"
else
    echo ""
    echo "❌ Deployment failed!"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "1. Check Railway dashboard for error details"
    echo "2. Verify your Dockerfile.railway is correct"
    echo "3. Ensure all dependencies are properly specified"
    echo "4. Check Railway build logs for specific errors"
fi 