# Detection GIS - Solar Panel and Pool Detection

A full-stack application for detecting solar panels and pools from satellite imagery using machine learning models.

## 🚀 Quick Deploy to Vercel

### Option 1: Automated Deployment
```bash
./deploy.sh
```

### Option 2: Manual Deployment
```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

## 📁 Project Structure

```
detection-gis-master/
├── frontend/                 # React frontend application
│   ├── src/                 # Source code
│   ├── public/              # Public assets
│   └── package.json         # Frontend dependencies
├── backend/                  # Node.js backend server
│   ├── server.js            # Main server file
│   ├── *.pt                 # ML models (PyTorch)
│   └── package.json         # Backend dependencies
├── vercel.json              # Vercel configuration
├── .vercelignore            # Files to exclude from Vercel
└── deploy.sh                # Deployment script
```

## 🛠️ Features

- **Interactive Map**: Built with Mapbox GL JS and React Map GL
- **ML Detection**: YOLOv8 models for solar panel and pool detection
- **Real-time Processing**: Upload satellite images for instant detection
- **Responsive UI**: Modern React interface with detection overlays
- **GIS Integration**: Geographic coordinate handling and mapping

## 🔧 Prerequisites

- Node.js 16+ 
- npm or yarn
- Vercel account (for frontend deployment)
- Mapbox access token
- Backend hosting solution (for ML models)

## 🚀 Deployment

### Frontend (Vercel)

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel --prod
   ```

3. **Set Environment Variables in Vercel Dashboard:**
   - `REACT_APP_MAPBOX_TOKEN`: Your Mapbox access token
   - `REACT_APP_BACKEND_URL`: Your backend API URL

### Backend (Separate Hosting Required)

The backend contains ML models and requires separate hosting. Recommended options:

- **Railway**: Good for Node.js + ML workloads
- **Render**: Free tier available
- **DigitalOcean App Platform**: Production-ready
- **AWS/GCP/Azure**: Full control, complex setup

## 🌍 Environment Variables

### Frontend (.env)
```env
REACT_APP_MAPBOX_TOKEN=your_mapbox_token
REACT_APP_BACKEND_URL=https://your-backend-domain.com
```

### Backend (.env)
```env
PORT=5001
NODE_ENV=production
# Add other required variables
```

## 📱 Usage

1. **Search Location**: Use the search bar to find specific coordinates
2. **Detect Objects**: Click "Detect" to analyze satellite imagery
3. **View Results**: See detected pools (red) and solar panels (blue)
4. **Toggle Overlay**: Show/hide detection bounding boxes

## 🔍 API Endpoints

- `GET /detections` - Retrieve all detections
- `POST /detect` - Process new satellite image
- `GET /health` - Health check endpoint

## 🚨 Important Notes

- **ML Models**: Large `.pt` files should be stored in cloud storage
- **Database**: Use cloud database instead of SQLite for production
- **File Storage**: Implement cloud storage for uploaded images
- **CORS**: Configure backend to allow Vercel domain requests

## 🐛 Troubleshooting

- **Build Errors**: Ensure all dependencies are installed
- **Environment Variables**: Check Vercel dashboard configuration
- **CORS Issues**: Verify backend CORS settings
- **API Errors**: Check backend logs and connectivity

## 📚 Documentation

- [Deployment Guide](./DEPLOYMENT.md) - Detailed deployment instructions
- [Frontend README](./frontend/README.md) - React app documentation
- [Vercel Documentation](https://vercel.com/docs) - Vercel deployment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For deployment issues:
1. Check the [Deployment Guide](./DEPLOYMENT.md)
2. Review Vercel dashboard logs
3. Verify environment variables
4. Test backend connectivity 