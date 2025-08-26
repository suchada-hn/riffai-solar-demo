const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
require('dotenv').config({ path: '.env.production' });

const app = express();

// Production CORS configuration
const allowedOrigins = [
    'http://localhost:3000',
    'http://localhost:3001', 
    'http://127.0.0.1:3000',
    'http://127.0.0.1:3001',
    'https://riffai-solar-platform.vercel.app',
    /https:\/\/.*\.vercel\.app$/
];

app.use(cors({
    origin: function (origin, callback) {
        if (!origin) return callback(null, true);
        
        if (allowedOrigins.some(allowed => {
            if (typeof allowed === 'string') {
                return allowed === origin;
            }
            if (allowed instanceof RegExp) {
                return allowed.test(origin);
            }
            return false;
        })) {
            return callback(null, true);
        }
        
        callback(new Error('Not allowed by CORS'));
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

// Ensure required directories exist
const ensureDirectories = () => {
    const dirs = ['uploads', 'annotated_images'];
    dirs.forEach(dir => {
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
            console.log(`Created directory: ${dir}`);
        }
    });
};

ensureDirectories();

// Serve static files
app.use('/annotated_images', express.static('annotated_images'));
app.use('/uploads', express.static('uploads'));

// Configure multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

// Database configuration
const { Pool } = require('pg');
const sqlite3 = require('sqlite3').verbose();

let pool;
let db;
let databaseStatus = 'disconnected';

// Initialize database with retry logic
const initializeDatabase = async () => {
    try {
        if (process.env.DATABASE_URL) {
            pool = new Pool({
                connectionString: process.env.DATABASE_URL,
                ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
            });
            console.log('Using PostgreSQL with DATABASE_URL');
            
            await pool.query('SELECT NOW()');
            databaseStatus = 'connected';
            console.log('PostgreSQL connected successfully');
        } else if (process.env.DATABASE_HOST && process.env.DATABASE_HOST !== 'localhost') {
            pool = new Pool({
                user: process.env.DATABASE_USER,
                host: process.env.DATABASE_HOST,
                database: process.env.DATABASE_NAME,
                password: process.env.DATABASE_PASSWORD,
                port: process.env.DATABASE_PORT || 5432,
                ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
            });
            console.log('Using PostgreSQL with individual variables');
            
            await pool.query('SELECT NOW()');
            databaseStatus = 'connected';
            console.log('PostgreSQL connected successfully');
        } else {
            // Fallback to SQLite for development
            db = new sqlite3.Database('detections.db');
            console.log('Using SQLite database');
            databaseStatus = 'connected';
        }
    } catch (error) {
        console.error('Database connection failed:', error);
        databaseStatus = 'failed';
        // Don't fail the server startup - continue without database
    }
};

// ML Model Management with lazy loading
let mlModelsLoaded = false;
let mlModelsLoading = false;

const loadMLModels = async () => {
    if (mlModelsLoaded || mlModelsLoading) return;
    
    mlModelsLoading = true;
    try {
        console.log('Loading ML models...');
        
        // Check if models exist
        const solarModelPath = path.join(__dirname, 'best-solar-panel.pt');
        const poolModelPath = path.join(__dirname, 'pool-best.pt');
        
        if (!fs.existsSync(solarModelPath) || !fs.existsSync(poolModelPath)) {
            console.log('ML models not found, ML detection will be disabled');
            mlModelsLoaded = false;
            return;
        }
        
        // Models exist, mark as loaded
        mlModelsLoaded = true;
        console.log('ML models ready for detection');
    } catch (error) {
        console.error('Error loading ML models:', error);
        mlModelsLoaded = false;
    } finally {
        mlModelsLoading = false;
    }
};

// Health check endpoint with ML status
app.get('/health', async (req, res) => {
    try {
        const healthStatus = {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            environment: process.env.NODE_ENV || 'development',
            database: databaseStatus,
            ml_models: mlModelsLoaded ? 'loaded' : 'not_loaded',
            uptime: process.uptime()
        };
        
        res.json(healthStatus);
    } catch (error) {
        res.status(500).json({ 
            status: 'error', 
            message: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// CORS test endpoint
app.get('/test-cors', (req, res) => {
    res.json({ 
        message: 'CORS test successful',
        timestamp: new Date().toISOString()
    });
});

// File upload endpoint with ML detection
app.post('/detect', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const imagePath = req.file.path;
        console.log(`Processing image: ${imagePath}`);

        // Check if ML detection is disabled
        if (process.env.DISABLE_ML_DETECTION === 'true') {
            return res.json({
                message: 'ML detection is disabled',
                imagePath: imagePath,
                timestamp: new Date().toISOString()
            });
        }

        // Ensure ML models are loaded
        if (!mlModelsLoaded) {
            await loadMLModels();
        }

        if (!mlModelsLoaded) {
            return res.status(503).json({
                error: 'ML models not available',
                message: 'Please try again later or contact support'
            });
        }

        // Run detection with timeout
        const detectionPromise = new Promise((resolve, reject) => {
            const pythonProcess = spawn('python3', [
                'run-solar-panel-and-pool-detection.py',
                imagePath
            ]);

            let output = '';
            let errorOutput = '';

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(output);
                } else {
                    reject(new Error(`Python process exited with code ${code}: ${errorOutput}`));
                }
            });

            // Set timeout for ML processing (5 minutes)
            setTimeout(() => {
                pythonProcess.kill();
                reject(new Error('ML detection timeout - process took too long'));
            }, 5 * 60 * 1000);
        });

        const result = await detectionPromise;
        
        res.json({
            message: 'Detection completed successfully',
            result: result,
            imagePath: imagePath,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Detection error:', error);
        res.status(500).json({
            error: 'Detection failed',
            message: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Initialize database and ML models on startup
const initializeApp = async () => {
    try {
        console.log('Initializing application...');
        
        // Initialize database
        await initializeDatabase();
        
        // Load ML models in background (don't block startup)
        if (process.env.DISABLE_ML_DETECTION !== 'true') {
            loadMLModels().catch(console.error);
        }
        
        console.log('Application initialization completed');
    } catch (error) {
        console.error('Application initialization failed:', error);
        // Continue running the server even if initialization fails
    }
};

// Start server
const PORT = process.env.PORT || 10000;

app.listen(PORT, async () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`ML Detection: ${process.env.DISABLE_ML_DETECTION === 'true' ? 'disabled' : 'enabled'}`);
    
    // Initialize app after server starts
    await initializeApp();
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    if (pool) pool.end();
    if (db) db.close();
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully');
    if (pool) pool.end();
    if (db) db.close();
    process.exit(0);
}); 