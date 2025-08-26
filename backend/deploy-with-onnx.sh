#!/bin/bash

# 🚀 Deploy Backend with ONNX Models
# This script deploys your backend using the converted ONNX models for faster performance

echo "🚀 Deploying Backend with ONNX Models"
echo "======================================"

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
if command -v railway &> /dev/null; then
    echo "🚂 Railway CLI found. Deploying to Railway..."
    
    # Deploy to Railway
    railway up
    
    echo "✅ Deployment completed!"
    echo "Your backend is now running with ONNX models for faster performance!"
    
elif command -v vercel &> /dev/null; then
    echo "☁️  Vercel CLI found. Deploying to Vercel..."
    
    # Deploy to Vercel
    vercel --prod
    
    echo "✅ Deployment completed!"
    echo "Your backend is now running with ONNX models for faster performance!"
    
else
    echo "⚠️  No deployment CLI found."
    echo ""
    echo "Available deployment options:"
    echo "1. Railway (Recommended for ML workloads):"
    echo "   npm install -g @railway/cli"
    echo "   railway login"
    echo "   railway up"
    echo ""
    echo "2. Vercel:"
    echo "   npm install -g vercel"
    echo "   vercel --prod"
    echo ""
    echo "3. Manual deployment:"
    echo "   - Copy your backend files to your hosting provider"
    echo "   - Ensure ONNX models are included in the models/ directory"
    echo "   - Use the ONNX-optimized detection script"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Test your deployed backend with the ONNX models"
echo "2. Monitor performance improvements"
echo "3. Update your frontend to use the new backend URL"
echo ""
echo "📚 Documentation:"
echo "- ONNX Conversion Success: ONNX_CONVERSION_SUCCESS.md"
echo "- Deployment Alternatives: DEPLOYMENT_ALTERNATIVES.md"
echo "- Railway Deployment: railway-deploy.md" 