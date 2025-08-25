# Deployment Guide

## Frontend Deployment to Vercel

This project is configured to deploy the React frontend to Vercel. The backend needs to be deployed separately since Vercel is primarily for frontend applications.

### Prerequisites

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Make sure you have a Vercel account at [vercel.com](https://vercel.com)

### Frontend Deployment Steps

1. **Login to Vercel:**
   ```bash
   vercel login
   ```

2. **Deploy the frontend:**
   ```bash
   vercel
   ```

3. **Follow the prompts:**
   - Set up and deploy: `Y`
   - Which scope: Select your account
   - Link to existing project: `N`
   - Project name: `detection-gis-frontend` (or your preferred name)
   - Directory: `./` (root directory)
   - Override settings: `N`

4. **Set Environment Variables:**
   After deployment, go to your Vercel dashboard and set these environment variables:
   - `REACT_APP_MAPBOX_TOKEN`: Your Mapbox access token
   - `REACT_APP_BACKEND_URL`: Your backend API URL

### Backend Deployment Options

Since Vercel doesn't support the ML models and heavy backend operations, consider these alternatives:

#### Option 1: Railway (Recommended for ML workloads)

1. **Sign up at [railway.app](https://railway.app)**
2. **Connect your GitHub repository**
3. **Deploy the backend directory:**
   ```bash
   # In Railway dashboard, set the root directory to: backend
   ```

4. **Set environment variables:**
   ```env
   NODE_ENV=production
   PORT=5001
   DATABASE_URL=your_postgresql_connection_string
   ```

5. **Use the production server:**
   - Railway will automatically run `npm start`
   - Make sure `package.json` has the correct start script

#### Option 2: Render

1. **Sign up at [render.com](https://render.com)**
2. **Create a new Web Service**
3. **Connect your GitHub repository**
4. **Configure:**
   - Build Command: `npm install`
   - Start Command: `node server.prod.js`
   - Root Directory: `backend`

#### Option 3: DigitalOcean App Platform

1. **Sign up at [digitalocean.com](https://digitalocean.com)**
2. **Create a new App**
3. **Connect your GitHub repository**
4. **Configure the backend service**

### Backend Production Configuration

The project includes a production-ready server configuration:

1. **Use the production server:**
   ```bash
   # Update package.json start script
   "start": "node server.prod.js"
   ```

2. **Set production environment variables:**
   ```bash
   # Copy the template
   cp backend/env.production.example backend/.env
   
   # Edit with your values
   nano backend/.env
   ```

3. **Key production considerations:**
   - **Database**: Use PostgreSQL instead of SQLite
   - **File Storage**: Implement cloud storage (S3, etc.)
   - **ML Models**: Store large `.pt` files in cloud storage
   - **CORS**: Configure to allow your Vercel domain

### Environment Variables for Backend

#### Required Variables:
```env
NODE_ENV=production
PORT=5001
```

#### Database (choose one):
```env
# Option 1: Single connection string
DATABASE_URL=postgresql://user:pass@host:port/db

# Option 2: Individual variables
DATABASE_HOST=your-host.com
DATABASE_USER=your-user
DATABASE_NAME=your-db
DATABASE_PASSWORD=your-password
DATABASE_PORT=5432
```

#### CORS Configuration:
```env
FRONTEND_URL=https://your-app.vercel.app
```

### Important Notes

1. **ML Models**: The `.pt` files (PyTorch models) are large and should be stored in cloud storage (S3, Google Cloud Storage, etc.) rather than in the repository.

2. **Database**: Consider using a cloud database service instead of SQLite for production.

3. **File Uploads**: The backend handles file uploads which need persistent storage in production.

4. **CORS**: Update CORS settings in the backend to allow requests from your Vercel frontend domain.

5. **Production Server**: Use `server.prod.js` instead of `server.js` for production deployments.

### Post-Deployment

1. Update the `REACT_APP_BACKEND_URL` in Vercel to point to your deployed backend
2. Test the application functionality
3. Set up custom domain if needed
4. Configure monitoring and logging
5. Test the health check endpoint: `https://your-backend.com/health`

### Troubleshooting

- **Build Errors**: Check that all dependencies are properly installed
- **Environment Variables**: Ensure all required environment variables are set
- **CORS Issues**: Verify backend CORS configuration allows your Vercel domain
- **API Errors**: Check backend logs and ensure the backend is accessible
- **Database Connection**: Verify database credentials and connection strings
- **File Permissions**: Ensure upload directories exist and are writable

### Quick Backend Deployment Commands

#### Railway:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

#### Render:
```bash
# Render uses GitHub integration
# Just push to your repository and configure in the dashboard
git push origin main
```

#### DigitalOcean:
```bash
# DigitalOcean uses GitHub integration
# Configure in the dashboard after connecting your repository
``` 