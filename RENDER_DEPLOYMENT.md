# 🚀 Render Backend Deployment - Fix CORS Issue

## Current Issue
Your frontend at `https://riffai-solar-platform.vercel.app` is getting CORS errors when trying to access your backend at `https://riffai-solar-demo.onrender.com`.

## ✅ Solution Applied
I've updated your `backend/server.js` with:
1. **Enhanced CORS configuration** that properly handles your Vercel domain
2. **CORS preflight handling** for OPTIONS requests
3. **Debug endpoint** at `/test-cors` to help troubleshoot

## 🔧 Deploy Updated Backend to Render

### Option 1: Automatic Deploy (Recommended)
1. **Push your updated code to GitHub:**
   ```bash
   git add .
   git commit -m "Fix CORS configuration for Vercel frontend"
   git push origin main
   ```
2. **Render will automatically redeploy** your backend

### Option 2: Manual Deploy
1. **Go to [Render Dashboard](https://dashboard.render.com)**
2. **Select your backend service** (`riffai-solar-demo`)
3. **Click "Manual Deploy"** → "Deploy latest commit"

## 🧪 Test the Fix

### 1. Test CORS Endpoint
Visit: `https://riffai-solar-demo.onrender.com/test-cors`

Expected response:
```json
{
  "message": "CORS test successful",
  "origin": "https://riffai-solar-platform.vercel.app",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 2. Test Detections Endpoint
Visit: `https://riffai-solar-demo.onrender.com/detections`

### 3. Test from Frontend
- Go to your Vercel frontend
- Try to load the detections
- Check browser console for CORS errors

## 🔍 Debugging Steps

### If CORS still fails:

1. **Check Render logs:**
   - Go to Render dashboard
   - Click on your service
   - Go to "Logs" tab
   - Look for CORS-related messages

2. **Test with curl:**
   ```bash
   curl -H "Origin: https://riffai-solar-platform.vercel.app" \
        -H "Access-Control-Request-Method: GET" \
        -H "Access-Control-Request-Headers: X-Requested-With" \
        -X OPTIONS \
        https://riffai-solar-demo.onrender.com/detections
   ```

3. **Check browser Network tab:**
   - Open DevTools
   - Go to Network tab
   - Look for the failed request
   - Check if OPTIONS preflight is successful

## 🚨 Common Issues & Solutions

### Issue: "Still getting CORS errors"
**Solution:** Wait 2-3 minutes for Render to fully deploy the changes

### Issue: "OPTIONS request failing"
**Solution:** The preflight handler should fix this. Check if the request reaches your backend.

### Issue: "Origin not in allowed list"
**Solution:** Check the console logs in Render for the exact origin being blocked.

## 📋 Environment Variables in Render

Ensure these are set in your Render service:
- `NODE_ENV`: `production`
- `PORT`: `10000` (or whatever Render assigns)
- `DATABASE_URL`: Your Supabase connection string

## 🔄 Force Redeploy

If the issue persists:
1. **Go to Render dashboard**
2. **Select your service**
3. **Click "Manual Deploy"**
4. **Choose "Clear build cache & deploy"**

## 📞 Support

If you still have CORS issues after deploying:
1. Check Render logs for errors
2. Test the `/test-cors` endpoint
3. Verify the request origin in browser DevTools
4. Check if your frontend is making the request with the correct origin

---

**After deploying, test the `/test-cors` endpoint first to verify CORS is working!** 