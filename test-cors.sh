#!/bin/bash

echo "🧪 Testing CORS Configuration..."

BACKEND_URL="https://riffai-solar-demo.onrender.com"
FRONTEND_ORIGIN="https://riffai-solar-platform.vercel.app"

echo "📍 Backend URL: $BACKEND_URL"
echo "🌐 Frontend Origin: $FRONTEND_ORIGIN"
echo ""

echo "1️⃣ Testing CORS endpoint..."
curl -s -H "Origin: $FRONTEND_ORIGIN" "$BACKEND_URL/test-cors" | jq '.' 2>/dev/null || curl -s -H "Origin: $FRONTEND_ORIGIN" "$BACKEND_URL/test-cors"

echo ""
echo "2️⃣ Testing OPTIONS preflight request..."
curl -s -H "Origin: $FRONTEND_ORIGIN" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: X-Requested-With" \
     -X OPTIONS \
     "$BACKEND_URL/detections" \
     -v 2>&1 | grep -E "(Access-Control-Allow-Origin|HTTP/)"

echo ""
echo "3️⃣ Testing detections endpoint..."
curl -s -H "Origin: $FRONTEND_ORIGIN" "$BACKEND_URL/detections" | jq '.' 2>/dev/null || curl -s -H "Origin: $FRONTEND_ORIGIN" "$BACKEND_URL/detections"

echo ""
echo "✅ CORS test complete!"
echo ""
echo "📝 If you see CORS errors, make sure to:"
echo "   1. Deploy the updated backend to Render"
echo "   2. Wait 2-3 minutes for deployment to complete"
echo "   3. Check Render logs for any errors" 