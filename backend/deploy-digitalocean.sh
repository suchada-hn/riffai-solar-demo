#!/bin/bash

# 🌊 DigitalOcean App Platform Deployment Script
# This is a reliable alternative to Railway for ML backends

echo "🌊 DigitalOcean App Platform Deployment"
echo "======================================="

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

echo ""
echo "🚀 DigitalOcean App Platform Deployment"
echo "======================================"
echo ""
echo "📋 Manual Deployment Steps:"
echo ""
echo "1. Go to: https://cloud.digitalocean.com/apps"
echo "2. Click 'Create App'"
echo "3. Connect your GitHub repository"
echo "4. Select the 'backend' directory"
echo "5. Choose 'Node.js' as environment"
echo "6. Set build command:"
echo "   python3 -m pip install --no-cache-dir -r requirements-minimal.txt && npm ci --only=production --no-audit --no-fund"
echo "7. Set run command: npm run start:prod"
echo "8. Set port: 10000"
echo "9. Add environment variables:"
echo "   - NODE_ENV=production"
echo "   - DISABLE_ML_DETECTION=false"
echo "   - PORT=10000"
echo "10. Deploy!"
echo ""
echo "🎯 Why DigitalOcean is Better for ML:"
echo "   ✅ No build timeouts (unlike Railway)"
echo "   ✅ Reliable infrastructure"
echo "   ✅ Good ML support"
echo "   ✅ Predictable pricing ($5/month)"
echo "   ✅ No service outages"
echo ""
echo "📚 Configuration files ready:"
echo "   - .do/app.yaml (DigitalOcean config)"
echo "   - server.prod.js (production server)"
echo "   - models/ (ONNX models)"
echo ""
echo "💡 After deployment:"
echo "   1. Test /health endpoint"
echo "   2. Test ML detection endpoints"
echo "   3. Update frontend to new backend URL"
echo "   4. Monitor performance"
echo ""
echo "🎉 Your ML backend will be reliable and fast!" 