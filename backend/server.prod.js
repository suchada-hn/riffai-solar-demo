const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
require('dotenv').config({ path: '.env.production' });

const app = express();

// Production CORS configuration - allow Vercel domain
const allowedOrigins = [
    'http://localhost:3000',
    'http://localhost:3001', 
    'http://127.0.0.1:3000',
    'http://127.0.0.1:3001',
    // Add your Vercel domain here
    'https://riffai-solar-platform.vercel.app',
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
let databaseStatus = 'disconnected';

// Use environment variables for database configuration
const initializeDatabase = async () => {
    try {
        if (true) {
            // Use DATABASE_URL (common for services like Railway, Render, etc.)
            pool = new Pool({
                connectionString: "postgresql://postgres.ewcpjdsepzegbkzndlyn:bringspacedown2earth@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres?pgbouncer=true",
                ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
            });
            console.log('Using PostgreSQL with DATABASE_URL');
            
            // Test connection
            await pool.query('SELECT NOW()');
            databaseStatus = 'connected';
            console.log('PostgreSQL connected successfully');
            
            // Create table if it doesn't exist
            await createDetectionsTable();
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
            
            // Test connection
            await pool.query('SELECT NOW()');
            databaseStatus = 'connected';
            console.log('PostgreSQL connected successfully');
            
            // Create table if it doesn't exist
            await createDetectionsTable();
        } else {
            // Fallback to SQLite for local development
            db = new sqlite3.Database('./detections.db');
            console.log('Using SQLite database for local development');
            databaseStatus = 'connected';
            
            // Create table if it doesn't exist
            await createDetectionsTable();
        }
    } catch (error) {
        console.error('Database initialization failed:', error.message);
        databaseStatus = 'error';
        
        // Don't exit process, just log the error
        if (pool) {
            pool.end();
            pool = null;
        }
        if (db) {
            db.close();
            db = null;
        }
    }
};

const createDetectionsTable = async () => {
    const query = `
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            class INTEGER NOT NULL,
            name TEXT NOT NULL,
            bbox_xmin REAL NOT NULL,
            bbox_ymin REAL NOT NULL,
            bbox_xmax REAL NOT NULL,
            bbox_ymax REAL NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            confidence REAL NOT NULL,
            original_image_path TEXT,
            annotated_image_path TEXT,
            center_latitude REAL,
            center_longitude REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    `;

    try {
        if (db) {
            // SQLite
            return new Promise((resolve, reject) => {
                db.run(query, (err) => {
                    if (err) {
                        console.error('Error creating SQLite table:', err.message);
                        reject(err);
                    } else {
                        console.log('SQLite detections table created successfully.');
                        resolve();
                    }
                });
            });
        } else if (pool) {
            // PostgreSQL
            await pool.query(query);
            console.log('PostgreSQL detections table created successfully.');
        }
    } catch (error) {
        console.error('Error creating detections table:', error.message);
    }
};

// Helper functions for detection processing
const convertPixelsToCoords = (detection) => {
    const IMAGE_WIDTH = 800;
    const IMAGE_HEIGHT = 600;
    const DEFAULT_ZOOM = 18.65;
    const SCALE_ADJUSTMENT = 0.875;
    const LONGITUDE_OFFSET = 0.00000042;
    const LATITUDE_OFFSET = -0.00000072;

    const classAdjustments = {
        1: { // pool
            scale: 0.93,
            lonOffset: -0.0000002,
            latOffset: -0.0000001
        },
        2: { // solar-panel
            scale: 0.94,
            lonOffset: 0.0000001,
            latOffset: -0.0000004
        }
    };

    const classAdjust = classAdjustments[detection.class] || { scale: 1, lonOffset: 0, latOffset: 0 };
    const verticalScaleFactor = detection.class === 2 ? 0.92 : 1;

    const metersPerPixelAtEquator = (156543.03392 * Math.cos(detection.latitude * Math.PI / 180) / Math.pow(2, DEFAULT_ZOOM))
        * SCALE_ADJUSTMENT * classAdjust.scale;

    const metersToDegreesAtEquator = 1 / 111319.9;
    const latCorrectionFactor = Math.cos(detection.latitude * Math.PI / 180);
    const degreesPerPixel = metersPerPixelAtEquator * metersToDegreesAtEquator;
    const lngPerPixel = degreesPerPixel / latCorrectionFactor;
    const latPerPixel = degreesPerPixel * verticalScaleFactor;

    const offsetX = (IMAGE_WIDTH / 2) * 0.988;
    const offsetY = (IMAGE_HEIGHT / 2) * 0.988;

    // Extract bounding box coordinates
    const bbox_xmin = detection.bbox[0];
    const bbox_xmax = detection.bbox[2];
    const bbox_ymin = detection.bbox[1];
    const bbox_ymax = detection.bbox[3];

    // Calculate center pixel coordinates
    const centerPixelX = (bbox_xmin + bbox_xmax) / 2;
    const centerPixelY = (bbox_ymin + bbox_ymax) / 2;

    // Convert pixel coordinates to latitude and longitude
    const centerLongitude = detection.longitude +
        (centerPixelX - offsetX) * lngPerPixel +
        LONGITUDE_OFFSET +
        classAdjust.lonOffset;

    const centerLatitude = detection.latitude -
        (centerPixelY - offsetY) * latPerPixel +
        LATITUDE_OFFSET +
        classAdjust.latOffset;

    return { centerLatitude, centerLongitude };
};

const detectionExists = async (centerLatitude, centerLongitude) => {
    const MARGIN_OF_ERROR = 0.0001;
    try {
        if (db) {
            // SQLite query
            const query = `
                SELECT 1 FROM detections
                WHERE ABS(center_latitude - ?) < ?
                AND ABS(center_longitude - ?) < ?
                LIMIT 1;
            `;
            const values = [centerLatitude, MARGIN_OF_ERROR, centerLongitude, MARGIN_OF_ERROR];

            return new Promise((resolve, reject) => {
                db.get(query, values, (err, row) => {
                    if (err) {
                        console.error('SQLite error:', err.message);
                        resolve(false);
                    } else {
                        resolve(row !== undefined);
                    }
                });
            });
        } else if (pool) {
            // PostgreSQL query
            const query = `
                SELECT 1 FROM detections
                WHERE ABS(center_latitude - $1) < $3
                AND ABS(center_longitude - $2) < $3
                LIMIT 1;
            `;
            const values = [centerLatitude, centerLongitude, MARGIN_OF_ERROR];

            const result = await pool.query(query, values);
            return result.rows.length > 0;
        } else {
            return false;
        }
    } catch (error) {
        console.error('Error checking for existing detection:', error.message);
        return false;
    }
};

const insertDetection = async (detection, originalImagePath, annotatedImagePath) => {
    if (detection.confidence <= 0.5) {
        console.log('Skipping detection with confidence below threshold:', detection.confidence);
        return { status: 'skipped', reason: 'low_confidence' };
    }

    // Calculate center coordinates using the conversion function
    const { centerLatitude, centerLongitude } = convertPixelsToCoords(detection);

    // Check if a similar detection already exists
    const exists = await detectionExists(centerLatitude, centerLongitude);
    if (exists) {
        console.log('Skipping repeated detection at coordinates:', { centerLatitude, centerLongitude });
        return { status: 'skipped', reason: 'duplicate' };
    }

    try {
        console.log('Inserting detection with converted center coordinates:', {
            centerLatitude,
            centerLongitude
        });

        if (db) {
            // SQLite insert
            const query = `
                INSERT INTO detections (
                    class, 
                    name, 
                    bbox_xmin, 
                    bbox_ymin, 
                    bbox_xmax, 
                    bbox_ymax, 
                    latitude, 
                    longitude, 
                    confidence,
                    original_image_path,
                    annotated_image_path,
                    center_latitude,
                    center_longitude
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            `;

            const values = [
                detection.class,
                detection.name,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3],
                detection.latitude,
                detection.longitude,
                detection.confidence,
                originalImagePath,
                annotatedImagePath,
                centerLatitude,
                centerLongitude
            ];

            return new Promise((resolve, reject) => {
                db.run(query, values, function(err) {
                    if (err) {
                        console.error('SQLite error:', err.message);
                        resolve({ status: 'error', reason: err.message });
                    } else {
                        console.log('Detection inserted successfully.');
                        resolve({ status: 'inserted' });
                    }
                });
            });
        } else if (pool) {
            // PostgreSQL insert
            const query = `
                INSERT INTO detections (
                    class, 
                    name, 
                    bbox_xmin, 
                    bbox_ymin, 
                    bbox_xmax, 
                    bbox_ymax, 
                    latitude, 
                    longitude, 
                    confidence,
                    original_image_path,
                    annotated_image_path,
                    center_latitude,
                    center_longitude
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            `;

            const values = [
                detection.class,
                detection.name,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3],
                detection.latitude,
                detection.longitude,
                detection.confidence,
                originalImagePath,
                annotatedImagePath,
                centerLatitude,
                centerLongitude
            ];

            await pool.query(query, values);
            console.log('Detection inserted successfully.');
            return { status: 'inserted' };
        } else {
            return { status: 'error', reason: 'No database connection available' };
        }
    } catch (error) {
        console.error('Error inserting detection:', error.message);
        return { status: 'error', reason: error.message };
    }
};

// Initialize database
initializeDatabase();

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        database: databaseStatus,
        environment: process.env.NODE_ENV || 'development'
    });
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'RiffAI Solar Detection API',
        version: '1.0.0',
        status: 'running',
        timestamp: new Date().toISOString(),
        endpoints: {
            health: '/health',
            detections: '/detections',
            detect: '/detect (POST)',
            cors_test: '/test-cors'
        },
        documentation: 'This API provides solar panel and pool detection services using machine learning.'
    });
});

// Test endpoint for CORS debugging
app.get('/test-cors', (req, res) => {
    console.log('CORS test endpoint hit');
    console.log('Origin:', req.headers.origin);
    console.log('User-Agent:', req.headers['user-agent']);
    res.json({ 
        message: 'CORS test successful',
        origin: req.headers.origin,
        timestamp: new Date().toISOString()
    });
});

// Get detections endpoint
app.get('/detections', async (req, res) => {
    try {
        if (db) {
            // SQLite query
            const query = `
                SELECT d.*,
                        CASE 
                            WHEN d.original_image_path IS NOT NULL 
                            THEN '/uploads/' || substr(d.original_image_path, instr(d.original_image_path, '/') + 1)
                            ELSE NULL 
                        END as original_image_url,
                        CASE 
                            WHEN d.annotated_image_path IS NOT NULL 
                            THEN '/annotated_images/' || substr(d.annotated_image_path, instr(d.annotated_image_path, '/') + 1)
                            ELSE NULL 
                        END as annotated_image_url,
                        center_latitude,
                        center_longitude
                FROM detections d
                ORDER BY created_at DESC;
            `;
            
            return new Promise((resolve, reject) => {
                db.all(query, [], (err, rows) => {
                    if (err) {
                        console.error('SQLite error:', err.message);
                        res.status(500).send('Error fetching detections');
                    } else {
                        res.json(rows);
                    }
                });
            });
        } else if (pool) {
            // PostgreSQL query
            const query = `
                SELECT d.*,
                        CASE 
                            WHEN d.original_image_path IS NOT NULL 
                            THEN '/uploads/' || split_part(d.original_image_path, '/', -1)
                            ELSE NULL 
                        END as original_image_url,
                        CASE 
                            WHEN d.annotated_image_path IS NOT NULL 
                            THEN '/annotated_images/' || split_part(d.annotated_image_path, '/', -1)
                            ELSE NULL 
                        END as annotated_image_url,
                        center_latitude,
                        center_longitude
                FROM detections d
                ORDER BY created_at DESC;
            `;
            const result = await pool.query(query);
            res.json(result.rows);
        } else {
            res.status(503).json({ error: 'Database not available', status: databaseStatus });
        }
    } catch (error) {
        console.error('Error fetching detections:', error.message);
        res.status(500).send('Error fetching detections');
    }
});

// Detection endpoint
app.post('/detect', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const { latitude, longitude, zoom } = req.body;
        
        // Check if ML detection is disabled
        if (process.env.DISABLE_ML_DETECTION === 'true') {
            return res.json({
                message: 'Detection endpoint reached (ML disabled)',
                file: req.file.filename,
                coordinates: { latitude, longitude, zoom },
                note: 'ML detection is disabled in production'
            });
        }
        
        // Try to run ML detection if available
        try {
            const result = await runMLDetection(req.file.path, latitude, longitude);
            
            // Process the result to match local server format
            if (result && result.detections) {
                // Add image URLs for frontend access
                const path = require('path');
                result.original_image_url = `/uploads/${path.basename(req.file.path)}`;
                
                // Check if detection_image exists in result
                if (result.detection_image) {
                    result.annotated_image_url = `/annotated_images/${path.basename(result.detection_image)}`;
                } else {
                    result.annotated_image_url = null;
                }
                
                // Ensure location format matches local server
                if (result.location) {
                    result.location.latitude = parseFloat(result.location.latitude);
                    result.location.longitude = parseFloat(result.location.longitude);
                }
                
                // Save detections to the database with image paths
                let skippedDetections = [];
                if (result.detections && result.detections.length > 0) {
                    console.log('Detections found, saving to database...');
                    for (const detection of result.detections) {
                        const dbResult = await insertDetection(
                            detection,
                            req.file.path.replace(/\\/g, '/'),  // Convert Windows path to Unix style
                            result.detection_image ? result.detection_image.replace(/\\/g, '/') : null  // Convert Windows path to Unix style
                        );
                        if (dbResult.status === 'skipped') {
                            skippedDetections.push(dbResult.reason);
                        }
                    }
                }
                
                // Add skipped detections info to result
                if (skippedDetections.length > 0) {
                    result.skippedDetections = skippedDetections;
                }
            }
            
            res.json(result);
        } catch (mlError) {
            console.error('ML detection failed, returning fallback:', mlError.message);
            // Fallback response
            res.json({
                message: 'Detection endpoint reached (ML failed)',
                file: req.file.filename,
                coordinates: { latitude, longitude, zoom },
                note: 'ML detection failed, using fallback response',
                error: mlError.message
            });
        }
        
    } catch (error) {
        console.error('Detection error:', error);
        res.status(500).json({ error: 'Detection failed' });
    }
});

// Endpoint for fetching a single detection by ID
app.get('/detections/:id', async (req, res) => {
    try {
        const { id } = req.params;
        
        if (db) {
            // SQLite query
            const query = `
                SELECT d.*, 
                        CASE 
                            WHEN d.annotated_image_path IS NOT NULL 
                            THEN '/annotated_images/' || substr(d.annotated_image_path, instr(d.annotated_image_path, '/') + 1)
                            ELSE NULL 
                        END as annotated_image_url
                FROM detections d
                WHERE d.id = ?;
            `;
            
            return new Promise((resolve, reject) => {
                db.get(query, [id], (err, row) => {
                    if (err) {
                        console.error('SQLite error:', err.message);
                        res.status(500).send('Error fetching detection');
                    } else if (!row) {
                        res.status(404).json({ error: 'Detection not found' });
                    } else {
                        res.json(row);
                    }
                });
            });
        } else if (pool) {
            // PostgreSQL query
            const query = `
                SELECT d.*, 
                        CASE 
                            WHEN d.annotated_image_path IS NOT NULL 
                            THEN '/annotated_images/' || split_part(d.annotated_image_path, '/', -1)
                            ELSE NULL 
                        END as annotated_image_url
                FROM detections d
                WHERE d.id = $1;
            `;
            const result = await pool.query(query, [id]);

            if (result.rows.length === 0) {
                return res.status(404).json({ error: 'Detection not found' });
            }

            res.json(result.rows[0]);
        } else {
            res.status(503).json({ error: 'Database not available', status: databaseStatus });
        }
    } catch (error) {
        console.error('Error fetching detection:', error.message);
        res.status(500).send('Error fetching detection');
    }
});

app.delete('/detections/:id', async (req, res) => {
    const { id } = req.params;
    try {
        if (db) {
            // SQLite delete
            const query = 'DELETE FROM detections WHERE id = ?';
            return new Promise((resolve, reject) => {
                db.run(query, [id], function(err) {
                    if (err) {
                        console.error('SQLite error:', err.message);
                        res.status(500).json({ error: 'Internal server error' });
                    } else if (this.changes === 0) {
                        res.status(404).json({ error: 'Detection not found' });
                    } else {
                        res.status(200).json({ message: 'Detection deleted successfully' });
                    }
                });
            });
        } else if (pool) {
            // PostgreSQL delete
            const query = 'DELETE FROM detections WHERE id = $1';
            const result = await pool.query(query, [id]);
            if (result.rowCount === 0) {
                return res.status(404).json({ error: 'Detection not found' });
            }
            res.status(200).json({ message: 'Detection deleted successfully' });
        } else {
            res.status(503).json({ error: 'Database not available', status: databaseStatus });
        }
    } catch (error) {
        console.error('Error deleting detection:', error.message);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ML Detection function
const runMLDetection = (imagePath, latitude, longitude) => {
    return new Promise((resolve, reject) => {
        // Use the same script as local server for consistency
        let scriptPath = './run-solar-panel-and-pool-detection.py';
        let scriptArgs = [scriptPath, imagePath, latitude, longitude];
        
        // Fallback to improved script if ONNX-only doesn't exist
        if (!fs.existsSync(scriptPath) && fs.existsSync('./run-solar-panel-and-pool-detection-improved.py')) {
            try {
                // Test if ONNX script works by running a quick test
                const testResult = require('child_process').spawnSync('python3', [
                    './run-solar-panel-and-pool-detection-improved.py', 
                    '--help'
                ], { timeout: 5000 });
                
                if (testResult.status === 0) {
                    scriptPath = './run-solar-panel-and-pool-detection-improved.py';
                    scriptArgs = [scriptPath, imagePath, '--latitude', latitude, '--longitude', longitude];
                    console.log('✅ Using improved ONNX script');
                } else {
                    console.log('⚠️  ONNX script test failed, using ONNX-only script');
                }
            } catch (error) {
                console.log('⚠️  ONNX script test error, using working PyTorch script');
            }
        }
        
        if (!fs.existsSync(scriptPath)) {
            reject(new Error('ML detection script not found'));
            return;
        }
        
        const pythonPath = 'python3';
        const pythonScript = spawn(pythonPath, scriptArgs);

        let output = "";
        let errorOutput = "";

        pythonScript.stdout.on('data', (data) => {
            output += data.toString();
            console.log('Python stdout:', data.toString());
        });

        pythonScript.stderr.on('data', (data) => {
            errorOutput += data.toString();
            console.error('Python stderr:', data.toString());
        });

        pythonScript.on('close', (code) => {
            console.log(`Python script exited with code ${code}`);
            if (code !== 0) {
                reject(new Error(`Python script failed with code ${code}: ${errorOutput}`));
                return;
            }
            
            try {
                const jsonLines = output.split('\n').filter(line => {
                    try {
                        JSON.parse(line);
                        return true;
                    } catch {
                        return false;
                    }
                });

                if (jsonLines.length === 0) {
                    reject(new Error('No valid JSON found in the output'));
                    return;
                }

                const detectionResult = JSON.parse(jsonLines[0]);
                resolve(detectionResult);
            } catch (error) {
                reject(new Error(`Failed to parse JSON output: ${error.message}`));
            }
        });

        pythonScript.on('error', (error) => {
            reject(new Error(`Failed to start Python script: ${error.message}`));
        });
    });
};

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    
    // Handle CORS errors specifically
    if (err.message === 'Not allowed by CORS') {
        return res.status(403).json({ error: 'CORS policy violation' });
    }
    
    res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

const PORT = process.env.PORT || 10000;

app.listen(PORT, () => {
    console.log(`Production server running on port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`Database status: ${databaseStatus}`);
});

module.exports = app; 
