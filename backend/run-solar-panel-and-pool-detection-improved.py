#!/usr/bin/env python3
"""
Improved Solar Panel and Pool Detection using ONNX models
This matches the detection quality of the local PyTorch backend
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import json
from pathlib import Path

def load_onnx_model(model_path):
    """Load ONNX model using ONNX Runtime (no PyTorch dependency)"""
    try:
        import onnxruntime as ort
        
        if not os.path.exists(model_path):
            print(f"❌ ONNX model not found: {model_path}")
            return None
            
        print(f"🚀 Loading ONNX model: {model_path}")
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        print(f"✅ ONNX model loaded successfully!")
        return session
        
    except ImportError:
        print("❌ ONNX Runtime not available. Installing...")
        os.system("pip install onnxruntime")
        import onnxruntime as ort
        return load_onnx_model(model_path)
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        return None

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for ONNX model input"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize and normalize
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert BGR to RGB and add batch dimension
    img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_rgb, axis=0)
    
    # Transpose to NCHW format (batch, channels, height, width)
    img_nchw = np.transpose(img_batch, (0, 3, 1, 2))
    
    return img_nchw, img_resized

def run_onnx_detection(session, input_data, confidence_threshold=0.3):
    """Run detection using ONNX model with lower confidence threshold"""
    try:
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        # Process outputs (assuming YOLOv8 format)
        predictions = outputs[0]  # Shape: (1, num_detections, 85)
        
        # Filter by confidence (lower threshold for better detection)
        detections = []
        for detection in predictions[0]:
            confidence = float(detection[4])  # Convert to Python float
            if confidence > confidence_threshold:
                class_id = int(detection[5])
                bbox = [float(x) for x in detection[0:4]]  # Convert to Python list of floats
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return detections
        
    except Exception as e:
        print(f"❌ Error during ONNX inference: {e}")
        return []

def filter_duplicates(detections, iou_threshold=0.5):
    """Filter duplicate detections using IoU"""
    if len(detections) <= 1:
        return detections
    
    filtered = []
    for i, det1 in enumerate(detections):
        is_duplicate = False
        for j, det2 in enumerate(filtered):
            if det1['class_id'] == det2['class_id']:
                # Calculate IoU
                iou = calculate_iou(det1['bbox'], det2['bbox'])
                if iou > iou_threshold:
                    # Keep the one with higher confidence
                    if det1['confidence'] > det2['confidence']:
                        filtered[j] = det1
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered.append(det1)
    
    return filtered

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def run_detection(image_path, latitude=None, longitude=None, output_dir="annotated_images"):
    """Run improved solar panel and pool detection using ONNX models"""
    start_time = time.time()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"🔄 Starting improved ONNX detection on: {image_path}")
        
        # Load ONNX models
        print("\n📥 Loading ONNX detection models...")
        
        solar_model_path = "models/best-solar-panel.onnx"
        pool_model_path = "models/pool-best.onnx"
        
        solar_session = load_onnx_model(solar_model_path)
        pool_session = load_onnx_model(pool_model_path)
        
        if not solar_session or not pool_session:
            print("❌ Failed to load ONNX models")
            return False
        
        model_loading_time = time.time() - start_time
        print(f"⏱️  Model loading time: {model_loading_time:.2f} seconds")
        
        # Preprocess image
        print(f"\n🖼️  Preprocessing image...")
        input_data, img_resized = preprocess_image(image_path)
        
        # Run detections with lower confidence threshold
        print(f"\n🔍 Running solar panel detection (confidence > 0.3)...")
        solar_detections = run_onnx_detection(solar_session, input_data, confidence_threshold=0.3)
        
        print(f"🔍 Running pool detection (confidence > 0.3)...")
        pool_detections = run_onnx_detection(pool_session, input_data, confidence_threshold=0.3)
        
        # Filter duplicates
        print(f"\n🔄 Filtering duplicate detections...")
        solar_detections = filter_duplicates(solar_detections, iou_threshold=0.5)
        pool_detections = filter_duplicates(pool_detections, iou_threshold=0.5)
        
        total_time = time.time() - start_time
        
        print(f"\n🎯 Improved Detection Results:")
        print(f"   Solar Panels detected: {len(solar_detections)}")
        print(f"   Pools detected: {len(pool_detections)}")
        print(f"   Total processing time: {total_time:.2f} seconds")
        
        # Create result matching PyTorch script format
        all_detections = []
        
        # Add pool detections
        for det in pool_detections:
            all_detections.append({
                "class": 1,  # Pool class
                "name": "pool",
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "latitude": latitude,
                "longitude": longitude
            })
        
        # Add solar panel detections
        for det in solar_detections:
            all_detections.append({
                "class": 2,  # Solar panel class
                "name": "solar_panel",
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "latitude": latitude,
                "longitude": longitude
            })
        
        result = {
            "detections": all_detections,
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "detection_image": None
        }
        
        # Print JSON result for the server to capture
        print(json.dumps(result))
        
        return True
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude
            }
        }
        print(json.dumps(error_result))
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Improved ONNX Solar Panel and Pool Detection")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--latitude", "-lat", type=float, help="Latitude coordinate")
    parser.add_argument("--longitude", "-lon", type=float, help="Longitude coordinate")
    parser.add_argument("--output", "-o", default="annotated_images", 
                       help="Output directory for annotated images")
    
    args = parser.parse_args()
    
    print("🚀 Improved ONNX Solar Panel and Pool Detection")
    print("=" * 55)
    
    # Check if ONNX models exist
    required_models = ["models/best-solar-panel.onnx", "models/pool-best.onnx"]
    missing_models = []
    
    for model in required_models:
        if not os.path.exists(model):
            missing_models.append(model)
    
    if missing_models:
        print(f"❌ Missing ONNX models: {', '.join(missing_models)}")
        print("Please ensure ONNX models are in the models/ directory")
        sys.exit(1)
    
    print(f"✅ ONNX models found and ready!")
    print(f"   - models/best-solar-panel.onnx")
    print(f"   - models/pool-best.onnx")
    
    print(f"\n🖼️  Processing image: {args.image_path}")
    print(f"📁 Output directory: {args.output}")
    if args.latitude and args.longitude:
        print(f"📍 Coordinates: {args.latitude}, {args.longitude}")
    
    # Run detection
    success = run_detection(args.image_path, args.latitude, args.longitude, args.output)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 