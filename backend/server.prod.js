const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
require('dotenv').config();

const app = express();

// Production CORS configuration - allow Vercel domain
const allowedOrigins = [
    'http://localhost:3000',
    'http://localhost:3001', 
    'http://127.0.0.1:3000',
    'http://127.0.0.1:3001',
    // Add your Vercel domain here
    'https://your-app.vercel.app',
    // Allow all Vercel preview deployments
    /https:\/\/.*\.vercel\.app$/
];

app.use(cors({
    origin: function (origin, callback) {
        // Allow requests with no origin (like mobile apps or curl requests)
        if (!origin) return callback(null, true);
        
        // Check if origin is in allowed list
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

// Serve static files from image directories
app.use('/annotated_images', express.static('annotated_images'));
app.use('/uploads', express.static('uploads'));

// Configure multer for file uploads
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

// Use environment variables for database configuration
if (process.env.DATABASE_URL) {
    // Use DATABASE_URL (common for services like Railway, Render, etc.)
    pool = new Pool({
        connectionString: process.env.DATABASE_URL,
        ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
    });
    console.log('Using PostgreSQL with DATABASE_URL');
} else if (process.env.DATABASE_HOST && process.env.DATABASE_HOST !== 'localhost') {
    // Use individual database environment variables
    pool = new Pool({
        user: process.env.DATABASE_USER,
        host: process.env.DATABASE_HOST,
        database: process.env.DATABASE_NAME,
        password: process.env.DATABASE_PASSWORD,
        port: process.env.DATABASE_PORT || 5432,
        ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
    });
    console.log('Using PostgreSQL with individual environment variables');
} else {
    // Fallback to SQLite for local development
    db = new sqlite3.Database('./detections.db');
    console.log('Using SQLite database for local development');
}

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        database: db ? 'SQLite' : 'PostgreSQL'
    });
});

// Get detections endpoint
app.get('/detections', async (req, res) => {
    try {
        if (db) {
            // SQLite query
            db.all('SELECT * FROM detections ORDER BY id DESC', (err, rows) => {
                if (err) {
                    console.error('SQLite error:', err);
                    res.status(500).json({ error: 'Database error' });
                    return;
                }
                res.json(rows);
            });
        } else if (pool) {
            // PostgreSQL query
            const result = await pool.query('SELECT * FROM detections ORDER BY id DESC');
            res.json(result.rows);
        } else {
            res.status(500).json({ error: 'No database connection' });
        }
    } catch (error) {
        console.error('Error fetching detections:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Detection endpoint
app.post('/detect', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const { latitude, longitude, zoom } = req.body;
        
        // For production, you might want to implement actual ML detection here
        // This is a placeholder response
        res.json({
            message: 'Detection endpoint reached',
            file: req.file.filename,
            coordinates: { latitude, longitude, zoom },
            note: 'ML detection not implemented in this production server'
        });
        
    } catch (error) {
        console.error('Detection error:', error);
        res.status(500).json({ error: 'Detection failed' });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

const PORT = process.env.PORT || 5001;

app.listen(PORT, () => {
    console.log(`Production server running on port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app; 