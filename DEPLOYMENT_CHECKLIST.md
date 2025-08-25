# Deployment Checklist

## ✅ Frontend (Vercel)

- [ ] Install Vercel CLI: `npm install -g vercel`
- [ ] Login to Vercel: `vercel login`
- [ ] Deploy frontend: `vercel --prod`
- [ ] Set environment variables in Vercel dashboard:
  - [ ] `REACT_APP_MAPBOX_TOKEN`
  - [ ] `REACT_APP_BACKEND_URL`
- [ ] Test frontend deployment
- [ ] Note your Vercel domain

## ✅ Backend (Choose one platform)

### Railway (Recommended)
- [ ] Sign up at [railway.app](https://railway.app)
- [ ] Connect GitHub repository
- [ ] Set root directory to `backend`
- [ ] Set environment variables:
  - [ ] `NODE_ENV=production`
  - [ ] `PORT=5001`
  - [ ] `DATABASE_URL` (PostgreSQL connection string)
- [ ] Deploy and note the URL

### Render
- [ ] Sign up at [render.com](https://render.com)
- [ ] Create Web Service
- [ ] Connect GitHub repository
- [ ] Set root directory to `backend`
- [ ] Set build command: `npm install`
- [ ] Set start command: `node server.prod.js`
- [ ] Deploy and note the URL

### DigitalOcean App Platform
- [ ] Sign up at [digitalocean.com](https://digitalocean.com)
- [ ] Create new App
- [ ] Connect GitHub repository
- [ ] Configure backend service
- [ ] Deploy and note the URL

## ✅ Database Setup

- [ ] Choose database provider (Railway, Render, etc.)
- [ ] Create PostgreSQL database
- [ ] Get connection string or credentials
- [ ] Test database connection
- [ ] Update backend environment variables

## ✅ Environment Configuration

### Frontend (Vercel)
- [ ] `REACT_APP_MAPBOX_TOKEN` = Your Mapbox token
- [ ] `REACT_APP_BACKEND_URL` = Your backend URL

### Backend
- [ ] `NODE_ENV` = production
- [ ] `PORT` = 5001 (or platform default)
- [ ] `DATABASE_URL` = Your PostgreSQL connection string
- [ ] `FRONTEND_URL` = Your Vercel frontend URL

## ✅ Testing

- [ ] Test frontend loads correctly
- [ ] Test backend health endpoint: `GET /health`
- [ ] Test frontend can connect to backend
- [ ] Test CORS is working
- [ ] Test basic functionality

## ✅ Production Considerations

- [ ] ML models stored in cloud storage (not in repo)
- [ ] File uploads configured for production
- [ ] Database using PostgreSQL (not SQLite)
- [ ] Environment variables secured
- [ ] Monitoring and logging configured
- [ ] Custom domain configured (optional)

## 🚨 Common Issues

- **CORS errors**: Update backend CORS to allow Vercel domain
- **Database connection**: Verify connection string and credentials
- **Environment variables**: Check all required variables are set
- **File permissions**: Ensure upload directories exist and are writable
- **ML models**: Large files may cause deployment timeouts

## 📞 Support

If you encounter issues:
1. Check the [Deployment Guide](./DEPLOYMENT.md)
2. Review platform-specific logs
3. Verify environment variables
4. Test endpoints individually
5. Check CORS configuration 