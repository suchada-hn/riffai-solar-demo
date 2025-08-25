#!/bin/bash

echo "🚀 Starting deployment to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if user is logged in to Vercel
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please login to Vercel..."
    vercel login
fi

echo "📦 Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "🚀 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo "📝 Don't forget to:"
echo "   1. Set environment variables in Vercel dashboard:"
echo "      - REACT_APP_MAPBOX_TOKEN"
echo "      - REACT_APP_BACKEND_URL"
echo "   2. Deploy your backend separately"
echo "   3. Update REACT_APP_BACKEND_URL to point to your backend" 