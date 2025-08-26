#!/usr/bin/env python3
"""
Solar Panel and Pool Detection using ONLY ONNX models
This bypasses PyTorch completely for reliable deployment
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
    """
    Load ONNX model using ONNX Runtime (no PyTorch dependency)
    """
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
    """
    Preprocess image for ONNX model input
    """
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

def run_onnx_detection(session, input_data, confidence_threshold=0.5):
    """
    Run detection using ONNX model
    """
    try:
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        
        # Process outputs (assuming YOLOv8 format)
        predictions = outputs[0]  # Shape: (1, num_detections, 85)
        
        # Filter by confidence
        detections = []
        for detection in predictions[0]:
            confidence = detection[4]
            if confidence > confidence_threshold:
                class_id = int(detection[5])
                bbox = detection[0:4]  # x1, y1, x2, y2
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return detections
        
    except Exception as e:
        print(f"❌ Error during ONNX inference: {e}")
        return []

def draw_detections(image, detections, class_names):
    """
    Draw detection boxes on image
    """
    img_with_boxes = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        class_name = class_names.get(class_id, f"Class {class_id}")
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(img_with_boxes, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_with_boxes

def run_detection(image_path, latitude=None, longitude=None, output_dir="annotated_images"):
    """
    Run solar panel and pool detection using ONNX models only
    """
    start_time = time.time()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"🔄 Starting ONNX-only detection on: {image_path}")
        
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
        
        # Run detections
        print(f"\n🔍 Running solar panel detection...")
        solar_detections = run_onnx_detection(solar_session, input_data)
        
        print(f"🔍 Running pool detection...")
        pool_detections = run_onnx_detection(pool_session, input_data)
        
        # Class names for visualization
        solar_class_names = {0: "solar-panel"}
        pool_class_names = {0: "pool"}
        
        # Draw detections
        print(f"\n🎨 Drawing detection results...")
        
        # Solar panel detections
        img_solar = draw_detections(img_resized, solar_detections, solar_class_names)
        solar_output_path = os.path.join(output_dir, "solar_panels", "detection_result.jpg")
        os.makedirs(os.path.dirname(solar_output_path), exist_ok=True)
        cv2.imwrite(solar_output_path, img_solar)
        
        # Pool detections
        img_pool = draw_detections(img_resized, pool_detections, pool_class_names)
        pool_output_path = os.path.join(output_dir, "pools", "detection_result.jpg")
        os.makedirs(os.path.dirname(pool_output_path), exist_ok=True)
        cv2.imwrite(pool_output_path, img_pool)
        
        total_time = time.time() - start_time
        
        print(f"\n🎯 Detection Results:")
        print(f"   Solar Panels detected: {len(solar_detections)}")
        print(f"   Pools detected: {len(pool_detections)}")
        print(f"   Total processing time: {total_time:.2f} seconds")
        print(f"   Model loading time: {model_loading_time:.2f} seconds")
        print(f"   Detection time: {total_time - model_loading_time:.2f} seconds")
        
        # Create JSON result for the server
        result = {
            "success": True,
            "detections": {
                "solar_panels": len(solar_detections),
                "pools": len(pool_detections),
                "total": len(solar_detections) + len(pool_detections)
            },
            "processing_time": total_time,
            "model_loading_time": model_loading_time,
            "detection_time": total_time - model_loading_time,
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude
            },
            "output_files": {
                "solar_panels": solar_output_path,
                "pools": pool_output_path
            }
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
    parser = argparse.ArgumentParser(description="ONNX-Only Solar Panel and Pool Detection")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--latitude", "-lat", type=float, help="Latitude coordinate")
    parser.add_argument("--longitude", "-lon", type=float, help="Longitude coordinate")
    parser.add_argument("--output", "-o", default="annotated_images", 
                       help="Output directory for annotated images")
    
    args = parser.parse_args()
    
    print("🚀 ONNX-Only Solar Panel and Pool Detection")
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