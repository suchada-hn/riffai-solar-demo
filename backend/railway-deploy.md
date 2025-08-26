# Railway Deployment Guide

## Why Railway is Better for ML Workloads

Railway is optimized for ML applications and provides:
- **Faster cold starts** (no 15-minute timeout like Render)
- **Better resource allocation** for ML workloads
- **Built-in ML support** with optimized buildpacks
- **Health check flexibility** (up to 5 minutes timeout)

## Deployment Steps

### 1. Install Railway CLI
```bash
npm install -g @railway/cli
```

### 2. Login to Railway
```bash
railway login
```

### 3. Initialize Project
```bash
cd backend
railway init
```

### 4. Deploy
```bash
railway up
```

### 5. Set Environment Variables
```bash
railway variables set NODE_ENV=production
railway variables set DISABLE_ML_DETECTION=false
# Add your database variables if using PostgreSQL
# railway variables set DATABASE_URL=your_postgres_connection_string
```

## Key Advantages Over Render

1. **No 15-minute timeout** - Railway handles ML model loading gracefully
2. **Better ML support** - Optimized for PyTorch, OpenCV, and YOLOv8
3. **Faster deployments** - NIXPACKS builder is faster than Docker
4. **Health check flexibility** - 5-minute timeout vs Render's strict limits
5. **Better resource scaling** - Automatically scales based on demand

## Cost Comparison

- **Render Starter**: $7/month (limited resources, strict timeouts)
- **Railway**: $5/month + usage (better ML support, no timeouts)

## Troubleshooting

If you still experience issues:
1. Set `DISABLE_ML_DETECTION=true` temporarily
2. Use lighter ML models (convert to ONNX format)
3. Implement model caching and lazy loading
4. Consider using Railway's GPU instances for heavy ML workloads 