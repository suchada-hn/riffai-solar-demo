#!/usr/bin/env python3
"""
Solar Panel and Pool Detection using ONNX models for faster loading
This script can use both .pt and .onnx models, with ONNX being preferred for speed
"""

import os
import sys
import argparse
import time
import torch
from pathlib import Path

def load_model(model_path):
    """
    Load a YOLO model, preferring ONNX format if available
    
    Args:
        model_path (str): Path to the model file (.pt or .onnx)
    
    Returns:
        YOLO model object
    """
    try:
        from ultralytics import YOLO
        
        # Check if ONNX version exists in models directory
        onnx_path = os.path.join("models", str(model_path).replace('.pt', '.onnx'))
        if os.path.exists(onnx_path):
            print(f"🚀 Loading ONNX model: {onnx_path}")
            model = YOLO(onnx_path)
            print(f"✅ ONNX model loaded successfully!")
            return model
        else:
            print(f"📦 Loading PyTorch model: {model_path}")
            # Use the same workaround for PyTorch models
            try:
                # Workaround: temporarily patch torch.load to allow weights_only=False
                original_load = torch.load
                
                def patched_load(f, *args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(f, *args, **kwargs)
                
                torch.load = patched_load
                model = YOLO(model_path)
                torch.load = original_load  # Restore original
                return model
            except Exception as e:
                print(f"❌ Error loading PyTorch model: {e}")
                raise
            
    except ImportError as e:
        print(f"❌ Error importing ultralytics: {e}")
        print("Please install: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def run_detection(image_path, output_dir="annotated_images"):
    """
    Run solar panel and pool detection on an image
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save annotated images
    """
    start_time = time.time()
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"🔄 Starting detection on: {image_path}")
        
        # Load models (prefer ONNX versions)
        print("\n📥 Loading detection models...")
        
        # Solar panel model
        solar_model_path = "best-solar-panel.pt"
        if not os.path.exists(solar_model_path):
            print(f"❌ Solar panel model not found: {solar_model_path}")
            return False
        
        solar_model = load_model(solar_model_path)
        
        # Pool model
        pool_model_path = "pool-best.pt"
        if not os.path.exists(pool_model_path):
            print(f"❌ Pool model not found: {pool_model_path}")
            return False
        
        pool_model = load_model(pool_model_path)
        
        model_loading_time = time.time() - start_time
        print(f"⏱️  Model loading time: {model_loading_time:.2f} seconds")
        
        # Run detections
        print(f"\n🔍 Running solar panel detection...")
        solar_results = solar_model(image_path, save=True, project=output_dir, name="solar_panels")
        
        print(f"🔍 Running pool detection...")
        pool_results = pool_model(image_path, save=True, project=output_dir, name="pools")
        
        # Get detection results
        solar_detections = len(solar_results[0].boxes) if solar_results[0].boxes is not None else 0
        pool_detections = len(pool_results[0].boxes) if pool_results[0].boxes is not None else 0
        
        total_time = time.time() - start_time
        
        print(f"\n🎯 Detection Results:")
        print(f"   Solar Panels detected: {solar_detections}")
        print(f"   Pools detected: {pool_detections}")
        print(f"   Total processing time: {total_time:.2f} seconds")
        print(f"   Model loading time: {model_loading_time:.2f} seconds")
        print(f"   Detection time: {total_time - model_loading_time:.2f} seconds")
        
        # Save results to file
        results_file = os.path.join(output_dir, "detection_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Detection Results for: {image_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Solar Panels detected: {solar_detections}\n")
            f.write(f"Pools detected: {pool_detections}\n")
            f.write(f"Total processing time: {total_time:.2f} seconds\n")
            f.write(f"Model loading time: {model_loading_time:.2f} seconds\n")
            f.write(f"Detection time: {total_time - model_loading_time:.2f} seconds\n")
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Show output file locations
        print(f"\n📁 Annotated images saved to:")
        solar_output = os.path.join(output_dir, "solar_panels")
        pool_output = os.path.join(output_dir, "pools")
        
        if os.path.exists(solar_output):
            for file in os.listdir(solar_output):
                if file.endswith(('.jpg', '.png')):
                    print(f"   Solar panels: {os.path.join(solar_output, file)}")
        
        if os.path.exists(pool_output):
            for file in os.listdir(pool_output):
                if file.endswith(('.jpg', '.png')):
                    print(f"   Pools: {os.path.join(pool_output, file)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Solar Panel and Pool Detection with ONNX support")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", default="annotated_images", 
                       help="Output directory for annotated images (default: annotated_images)")
    
    args = parser.parse_args()
    
    print("🚀 Solar Panel and Pool Detection (ONNX Optimized)")
    print("=" * 60)
    
    # Check if models exist
    required_models = ["best-solar-panel.pt", "pool-best.pt"]
    missing_models = []
    
    for model in required_models:
        if not os.path.exists(model):
            missing_models.append(model)
    
    if missing_models:
        print(f"❌ Missing required models: {', '.join(missing_models)}")
        print("Please ensure you're in the backend directory with the model files")
        print("Or run the conversion script first: python convert_models_to_onnx.py")
        return
    
    # Check for ONNX models
    onnx_models = []
    for model in required_models:
        onnx_path = os.path.join("models", model.replace('.pt', '.onnx'))
        if os.path.exists(onnx_path):
            onnx_models.append(onnx_path)
    
    if onnx_models:
        print(f"✅ Found ONNX models: {', '.join(onnx_path.split('/')[-1] for onnx_path in onnx_models)}")
        print("   These will load much faster than the original .pt models")
    else:
        print("⚠️  No ONNX models found. Using original .pt models (slower loading)")
        print("   Run: python convert_models_to_onnx.py to convert models for faster loading")
    
    print(f"\n🖼️  Processing image: {args.image_path}")
    print(f"📁 Output directory: {args.output}")
    
    # Run detection
    success = run_detection(args.image_path, args.output)
    
    if success:
        print(f"\n🎉 Detection completed successfully!")
    else:
        print(f"\n❌ Detection failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 