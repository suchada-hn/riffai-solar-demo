# 🚀 Quick Start - Deploy to Vercel in 5 Minutes

## Immediate Deployment

### 1. Deploy Frontend (Vercel)
```bash
# Run the automated deployment script
./deploy.sh

# OR deploy manually
vercel --prod
```

### 2. Set Environment Variables in Vercel Dashboard
- `REACT_APP_MAPBOX_TOKEN`: Your Mapbox access token
- `REACT_APP_BACKEND_URL`: Your backend URL (set after step 3)

### 3. Deploy Backend (Railway - Recommended)
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Set root directory to `backend`
4. Add environment variables:
   ```env
   NODE_ENV=production
   DATABASE_URL=your_postgresql_connection
   ```
5. Deploy and copy the URL

### 4. Update Frontend
- Go back to Vercel dashboard
- Set `REACT_APP_BACKEND_URL` to your Railway backend URL

## 🎯 What You Get

- ✅ **Frontend**: Deployed on Vercel with custom domain
- ✅ **Backend**: Running on Railway with PostgreSQL
- ✅ **ML Models**: Ready for detection (models need cloud storage)
- ✅ **Database**: Production-ready PostgreSQL
- ✅ **CORS**: Configured for cross-origin requests

## 📱 Test Your Deployment

1. **Frontend**: Visit your Vercel URL
2. **Backend Health**: `GET /health` endpoint
3. **API Test**: Try the detections endpoint

## 🆘 Need Help?

- 📖 [Full Deployment Guide](./DEPLOYMENT.md)
- ✅ [Deployment Checklist](./DEPLOYMENT_CHECKLIST.md)
- 🐛 [Troubleshooting](./DEPLOYMENT.md#troubleshooting)

## 🚨 Important Notes

- **ML Models**: Large `.pt` files need cloud storage for production
- **Database**: Use PostgreSQL, not SQLite for production
- **File Storage**: Implement cloud storage for uploads
- **Environment Variables**: Secure all sensitive data

---

**Ready to deploy? Run `./deploy.sh` and follow the prompts!** 🚀 