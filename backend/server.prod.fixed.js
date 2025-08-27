// Fixed Production Server - Uses working script first
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();

// Simple in-memory storage for detections (replace with database later)
let detections = [];
let nextId = 1;

// CORS configuration
app.use(cors({
    origin: process.env.FRONTEND_URL || '*',
    credentials: true
}));

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Static file serving
app.use('/uploads', express.static('uploads'));
app.use('/annotated_images', express.static('annotated_images'));

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV || 'production',
        ml_models: 'available'
    });
});

// File upload configuration
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = 'uploads';
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const timestamp = Date.now();
        const random = Math.floor(Math.random() * 1000000);
        cb(null, `image-${timestamp}-${random}.jpg`);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files are allowed'));
        }
    }
});

// ML Detection endpoint
app.post('/detect', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const { latitude, longitude } = req.body;
        const imagePath = req.file.path;

        console.log(`Detection request for: ${req.file.filename}`);
        console.log(`Coordinates: ${latitude}, ${longitude}`);

        // Run ML detection
        const result = await runMLDetection(imagePath, latitude, longitude);
        
        if (result.success) {
            // Store detection result
            const detectionRecord = {
                id: nextId++,
                filename: req.file.filename,
                detections: result.detections,
                location: {
                    latitude: latitude,
                    longitude: longitude
                },
                timestamp: new Date().toISOString(),
                original_image_url: `/uploads/${req.file.filename}`,
                annotated_image_url: null
            };
            
            detections.push(detectionRecord);
            
            res.status(200).json({
                detections: result.detections,
                location: {
                    latitude: latitude,
                    longitude: longitude
                },
                detection_image: null,
                original_image_url: `/uploads/${req.file.filename}`,
                annotated_image_url: null
            });
        } else {
            res.status(500).json({
                error: 'ML detection failed',
                details: result.error
            });
        }

    } catch (error) {
        console.error('Detection error:', error);
        res.status(500).json({
            error: 'Internal server error',
            details: error.message
        });
    }
});

// Get all detections endpoint
app.get('/detections', (req, res) => {
    try {
        res.status(200).json(detections);
    } catch (error) {
        console.error('Error fetching detections:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Get specific detection by ID
app.get('/detections/:id', (req, res) => {
    try {
        const { id } = req.params;
        const detection = detections.find(d => d.id == id);
        
        if (!detection) {
            return res.status(404).json({ error: 'Detection not found' });
        }
        
        res.status(200).json(detection);
    } catch (error) {
        console.error('Error fetching detection:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Delete detection by ID
app.delete('/detections/:id', (req, res) => {
    try {
        const { id } = req.params;
        const index = detections.findIndex(d => d.id == id);
        
        if (index === -1) {
            return res.status(404).json({ error: 'Detection not found' });
        }
        
        detections.splice(index, 1);
        res.status(200).json({ message: 'Detection deleted successfully' });
    } catch (error) {
        console.error('Error deleting detection:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ML Detection function - Uses working script first
const runMLDetection = (imagePath, latitude, longitude) => {
    return new Promise((resolve, reject) => {
        // Use the working PyTorch script first (same as server.js)
        let scriptPath = './run-solar-panel-and-pool-detection.py';
        let scriptArgs = [scriptPath, imagePath, latitude, longitude];
        
        // Only try ONNX if PyTorch script fails
        if (!fs.existsSync(scriptPath)) {
            if (fs.existsSync('./run-solar-panel-and-pool-detection-improved.py')) {
                scriptPath = './run-solar-panel-and-pool-detection-improved.py';
                scriptArgs = [scriptPath, imagePath, '--latitude', latitude, '--longitude', longitude];
                console.log('✅ Using improved ONNX script (PyTorch not available)');
            } else {
                reject(new Error('No detection script found'));
                return;
            }
        } else {
            console.log('✅ Using working PyTorch script');
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
                
                // Check if the result has detections
                if (detectionResult.detections && Array.isArray(detectionResult.detections)) {
                    // Add success flag to match expected format
                    detectionResult.success = true;
                    resolve(detectionResult);
                } else {
                    reject(new Error('Invalid detection result format'));
                }
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
    res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

const PORT = process.env.PORT || 10000;

app.listen(PORT, () => {
    console.log(`Fixed production server running on port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'production'}`);
    console.log(`Using working detection script first`);
});

module.exports = app; 