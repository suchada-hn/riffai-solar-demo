#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test New Solar Panel Detection Models
====================================

This script tests newly trained models and compares them with current ones
to verify improvements in detection quality.
"""

import os
import sys
import time
import json
from pathlib import Path

def test_model_performance(model_path, test_image, model_type="ONNX"):
    """Test a single model's performance"""
    print(f"\n🔍 Testing {model_type} model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        if model_type == "ONNX":
            return test_onnx_model(model_path, test_image)
        elif model_type == "PyTorch":
            return test_pytorch_model(model_path, test_image)
        else:
            print(f"❌ Unknown model type: {model_type}")
            return None
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return None

def test_onnx_model(model_path, test_image):
    """Test ONNX model performance"""
    try:
        import onnxruntime as ort
        import cv2
        import numpy as np
        
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Load and preprocess image
        img = cv2.imread(test_image)
        if img is None:
            print(f"❌ Could not load image: {test_image}")
            return None
        
        # Resize to model input size (assuming 640x640)
        img_resized = cv2.resize(img, (640, 640))
        input_data = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        start_time = time.time()
        outputs = session.run(None, {input_name: input_data})
        inference_time = time.time() - start_time
        
        # Process outputs (simplified)
        predictions = outputs[0]
        detections = []
        
        for detection in predictions[0]:
            confidence = float(detection[4])
            if confidence > 0.25:  # Confidence threshold
                class_id = int(detection[5])
                bbox = [float(x) for x in detection[0:4]]
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return {
            'model_type': 'ONNX',
            'model_path': model_path,
            'detections_count': len(detections),
            'inference_time': inference_time,
            'detections': detections,
            'avg_confidence': sum(d['confidence'] for d in detections) / len(detections) if detections else 0
        }
        
    except ImportError:
        print("❌ ONNX Runtime not available. Install with: pip install onnxruntime")
        return None
    except Exception as e:
        print(f"❌ ONNX testing failed: {e}")
        return None

def test_pytorch_model(model_path, test_image):
    """Test PyTorch model performance"""
    try:
        from ultralytics import YOLO
        
        # Load PyTorch model
        model = YOLO(model_path)
        
        # Run inference
        start_time = time.time()
        results = model.predict(source=test_image, conf=0.25, iou=0.45, max_det=300)
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append({
                        'bbox': box.xyxy[0].tolist(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0])
                    })
        
        return {
            'model_type': 'PyTorch',
            'model_path': model_path,
            'detections_count': len(detections),
            'inference_time': inference_time,
            'detections': detections,
            'avg_confidence': sum(d['confidence'] for d in detections) / len(detections) if detections else 0
        }
        
    except ImportError:
        print("❌ Ultralytics not available. Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"❌ PyTorch testing failed: {e}")
        return None

def compare_models(results):
    """Compare model performance results"""
    print("\n📊 Model Performance Comparison")
    print("=" * 50)
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Sort by detection count (descending)
    results.sort(key=lambda x: x['detections_count'] if x else 0, reverse=True)
    
    print(f"{'Model':<30} {'Type':<10} {'Detections':<12} {'Avg Conf':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for result in results:
        if result:
            print(f"{Path(result['model_path']).name:<30} "
                  f"{result['model_type']:<10} "
                  f"{result['detections_count']:<12} "
                  f"{result['avg_confidence']:<10.3f} "
                  f"{result['inference_time']:<10.3f}")
    
    # Find best model
    best_result = max(results, key=lambda x: x['detections_count'] if x else 0)
    if best_result:
        print(f"\n🏆 Best Model: {Path(best_result['model_path']).name}")
        print(f"   Detections: {best_result['detections_count']}")
        print(f"   Average Confidence: {best_result['avg_confidence']:.3f}")
        print(f"   Inference Time: {best_result['inference_time']:.3f}s")

def find_test_images():
    """Find test images in the backend directory"""
    test_images = []
    
    # Look for common image directories
    search_paths = [
        "uploads",
        "test_images", 
        "sample_images",
        "training_data/valid/images",
        "training_data/test/images"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(path, file))
    
    # Also look for images in current directory
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_images.append(file)
    
    return test_images[:3]  # Return first 3 images

def main():
    """Main testing function"""
    print("🧪 Solar Panel Detection Model Testing")
    print("=" * 40)
    
    # Find test images
    test_images = find_test_images()
    if not test_images:
        print("❌ No test images found. Please add some images to test with.")
        return
    
    print(f"📸 Found {len(test_images)} test images:")
    for img in test_images:
        print(f"   - {img}")
    
    # Test current models
    current_models = [
        ("models/best-solar-panel.onnx", "ONNX"),
        ("models/pool-best.onnx", "ONNX"),
        ("run-solar-panel-and-pool-detection.py", "PyTorch")  # Script path
    ]
    
    # Test new models (if they exist)
    new_models = [
        ("models/best-solar-panel-improved.onnx", "ONNX"),
        ("solar_detection_simple/yolo11n_solar/weights/best.pt", "PyTorch"),
        ("solar_detection_improved/yolo11n_solar_optimized/weights/best.pt", "PyTorch")
    ]
    
    all_models = current_models + new_models
    results = []
    
    # Test each model with first test image
    test_image = test_images[0]
    print(f"\n🎯 Testing models with: {test_image}")
    
    for model_path, model_type in all_models:
        result = test_model_performance(model_path, test_image, model_type)
        results.append(result)
    
    # Compare results
    compare_models(results)
    
    # Save results to file
    output_file = "model_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if any(r and r['detections_count'] > 20 for r in results):
        print("   ✅ Some models show good detection count (>20)")
    else:
        print("   ⚠️ All models show low detection count (<20)")
    
    if any(r and r['avg_confidence'] > 0.5 for r in results):
        print("   ✅ Some models show good confidence (>0.5)")
    else:
        print("   ⚠️ All models show low confidence (<0.5)")
    
    print("\n🔧 Next steps:")
    print("   1. Train new models if current ones are insufficient")
    print("   2. Update backend to use best performing model")
    print("   3. Test with more images for validation")

if __name__ == "__main__":
    main() 