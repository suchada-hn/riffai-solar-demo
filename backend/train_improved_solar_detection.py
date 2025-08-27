#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved YOLO11 Solar Panel Detection Training Script
====================================================

This script creates an optimized YOLO11 model specifically for solar panel detection
that should provide better results than the current ONNX models.

Features:
- Uses YOLO11n (nano) for faster training and inference
- Optimized hyperparameters for solar panel detection
- Data augmentation specifically for satellite imagery
- Export to ONNX with better settings
- Comprehensive evaluation and testing
"""

import os
import sys
import locale
import subprocess
import shutil
from pathlib import Path

# Set locale for better compatibility
locale.getpreferredencoding = lambda: "UTF-8"

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    packages = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def download_dataset():
    """Download the solar panel dataset from Roboflow"""
    print("📥 Downloading solar panel dataset...")
    
    dataset_url = "https://universe.roboflow.com/ds/dCRuFFJd7j?key=9SOXLBGP8y"
    
    try:
        # Download dataset
        subprocess.check_call([
            "curl", "-L", dataset_url, "-o", "roboflow.zip"
        ])
        
        # Extract dataset
        subprocess.check_call(["unzip", "roboflow.zip"])
        
        # Clean up
        os.remove("roboflow.zip")
        
        print("✅ Dataset downloaded and extracted successfully")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Failed to download dataset")
        return False

def create_optimized_config():
    """Create an optimized training configuration"""
    print("⚙️ Creating optimized training configuration...")
    
    config = {
        "task": "detect",
        "model": "yolo11n.pt",
        "data": "data.yaml",
        "epochs": 100,  # More epochs for better training
        "patience": 20,  # Early stopping patience
        "batch": 16,     # Batch size
        "imgsz": 1280,   # Image size
        "save": True,
        "save_period": 10,
        "cache": True,   # Cache images for faster training
        "device": "auto",
        "workers": 8,
        "project": "solar_detection_improved",
        "name": "yolo11n_solar_optimized",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",  # Better optimizer
        "lr0": 0.001,         # Initial learning rate
        "lrf": 0.01,          # Final learning rate factor
        "momentum": 0.937,     # SGD momentum/Adam beta1
        "weight_decay": 0.0005, # Optimizer weight decay
        "warmup_epochs": 3,    # Warmup epochs
        "warmup_momentum": 0.8, # Warmup initial momentum
        "warmup_bias_lr": 0.1,  # Warmup initial bias lr
        "box": 7.5,            # Box loss gain
        "cls": 0.5,            # Class loss gain
        "dfl": 1.5,            # DFL loss gain
        "pose": 12.0,          # Pose loss gain
        "kobj": 2.0,           # Keypoint obj loss gain
        "label_smoothing": 0.0, # Label smoothing epsilon
        "nbs": 64,             # Nominal batch size
        "overlap_mask": True,   # Masks should overlap during training
        "mask_ratio": 4,       # Mask downsample ratio
        "dropout": 0.0,        # Use dropout regularization
        "val": True,           # Validate during training
        "plots": True,         # Save plots
        "save_txt": False,     # Save results to *.txt
        "save_conf": False,    # Save confidences in --save-txt labels
        "save_crop": False,    # Save cropped prediction boxes
        "show": False,         # Show results
        "show_labels": True,   # Show labels
        "show_conf": True,     # Show confidences
        "show_boxes": True,    # Show boxes
        "conf": 0.25,          # Confidence threshold
        "iou": 0.45,           # NMS IoU threshold
        "max_det": 300,        # Maximum detections per image
        "half": True,          # Use FP16 half-precision inference
        "dnn": False,          # Use OpenCV DNN for ONNX inference
        "plots": True,         # Save plots
        "source": "test/images"  # Test source
    }
    
    # Create config file
    config_path = "solar_training_config.yaml"
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✅ Configuration saved to {config_path}")
    return config_path

def train_model(config_path):
    """Train the YOLO11 model with optimized settings"""
    print("🚀 Starting model training...")
    
    try:
        from ultralytics import YOLO
        
        # Load YOLO11n model
        model = YOLO("yolo11n.pt")
        print("✅ YOLO11n model loaded successfully")
        
        # Start training with optimized configuration
        results = model.train(
            data="data.yaml",
            epochs=100,
            imgsz=1280,
            batch=16,
            device="auto",
            project="solar_detection_improved",
            name="yolo11n_solar_optimized",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=2.0,
            label_smoothing=0.0,
            nbs=64,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,
            save_period=10,
            cache=True,
            workers=8
        )
        
        print("✅ Training completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def tune_hyperparameters():
    """Tune hyperparameters for better performance"""
    print("🎯 Tuning hyperparameters...")
    
    try:
        from ultralytics import YOLO
        
        # Load the trained model
        model_path = "solar_detection_improved/yolo11n_solar_optimized/weights/best.pt"
        if not os.path.exists(model_path):
            print("❌ Trained model not found. Please train first.")
            return False
        
        model = YOLO(model_path)
        
        # Define search space for solar panel detection
        search_space = {
            "lr0": (0.0001, 0.01),           # Learning rate
            "lrf": (0.001, 0.1),             # Final learning rate factor
            "momentum": (0.8, 0.98),          # Momentum
            "weight_decay": (0.0001, 0.001),  # Weight decay
            "box": (5.0, 10.0),               # Box loss gain
            "cls": (0.3, 0.7),                # Class loss gain
            "dfl": (1.0, 2.0),                # DFL loss gain
            "hsv_h": (0.0, 0.1),             # HSV-Hue augmentation
            "hsv_s": (0.0, 0.1),             # HSV-Saturation augmentation
            "hsv_v": (0.0, 0.1),             # HSV-Value augmentation
            "degrees": (0.0, 30.0),           # Image rotation
            "translate": (0.0, 0.1),          # Image translation
            "scale": (0.8, 1.2),              # Image scaling
            "shear": (0.0, 10.0),             # Image shear
            "perspective": (0.0, 0.001),      # Perspective transform
            "flipud": (0.0, 0.5),             # Flip up-down probability
            "fliplr": (0.0, 0.5),             # Flip left-right probability
            "mosaic": (0.0, 1.0),             # Mosaic augmentation
            "mixup": (0.0, 1.0),              # Mixup augmentation
            "copy_paste": (0.0, 1.0)          # Copy-paste augmentation
        }
        
        # Run hyperparameter tuning
        model.tune(
            data="data.yaml",
            epochs=50,
            iterations=100,
            optimizer="AdamW",
            space=search_space,
            plots=True,
            save=True,
            val=True,
            project="solar_detection_improved",
            name="yolo11n_solar_tuned"
        )
        
        print("✅ Hyperparameter tuning completed!")
        return True
        
    except Exception as e:
        print(f"❌ Hyperparameter tuning failed: {e}")
        return False

def export_models():
    """Export models to ONNX format with optimized settings"""
    print("📤 Exporting models to ONNX...")
    
    try:
        from ultralytics import YOLO
        
        # Export best trained model
        model_paths = [
            "solar_detection_improved/yolo11n_solar_optimized/weights/best.pt",
            "solar_detection_improved/yolo11n_solar_tuned/weights/best.pt"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"🔄 Exporting {model_path}...")
                
                model = YOLO(model_path)
                
                # Export to ONNX with optimized settings
                onnx_path = model.export(
                    format="onnx",
                    dynamic=True,           # Dynamic batch size
                    simplify=True,          # Simplify model
                    opset=12,              # ONNX opset version
                    half=True,             # FP16 precision
                    int8=False,            # INT8 quantization
                    optimize=True,         # Optimize model
                    workspace=4,           # Workspace size in GB
                    nms=True,              # Include NMS
                    agnostic_nms=False,    # Class-agnostic NMS
                    topk_all=100,         # Top-k for all classes
                    iou_thres=0.45,       # IoU threshold
                    conf_thres=0.25,       # Confidence threshold
                    max_det=300            # Maximum detections
                )
                
                print(f"✅ ONNX model exported: {onnx_path}")
                
                # Copy to models directory
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                
                shutil.copy2(onnx_path, models_dir / f"{Path(model_path).stem}_optimized.onnx")
                print(f"✅ Model copied to models directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Model export failed: {e}")
        return False

def validate_model():
    """Validate the trained model"""
    print("🔍 Validating model...")
    
    try:
        from ultralytics import YOLO
        
        # Load best model
        model_path = "solar_detection_improved/yolo11n_solar_optimized/weights/best.pt"
        if not os.path.exists(model_path):
            print("❌ Trained model not found.")
            return False
        
        model = YOLO(model_path)
        
        # Run validation
        metrics = model.val(
            data="data.yaml",
            imgsz=1280,
            batch=16,
            device="auto",
            plots=True,
            save_txt=True,
            save_conf=True,
            save_json=True
        )
        
        print("📊 Model Validation Results:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        print(f"   Precision: {metrics.box.mp:.3f}")
        print(f"   Recall: {metrics.box.mr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def test_inference():
    """Test model inference on sample images"""
    print("🧪 Testing model inference...")
    
    try:
        from ultralytics import YOLO
        
        # Load best model
        model_path = "solar_detection_improved/yolo11n_solar_optimized/weights/best.pt"
        if not os.path.exists(model_path):
            print("❌ Trained model not found.")
            return False
        
        model = YOLO(model_path)
        
        # Test on validation images
        test_source = "valid/images" if os.path.exists("valid/images") else "test/images"
        
        if os.path.exists(test_source):
            results = model.predict(
                source=test_source,
                save=True,
                conf=0.25,
                iou=0.45,
                max_det=300,
                project="solar_detection_improved",
                name="inference_test"
            )
            
            print(f"✅ Inference completed on {len(results)} images")
            print(f"📁 Results saved to: solar_detection_improved/inference_test/")
            
            return True
        else:
            print("❌ Test images not found")
            return False
        
    except Exception as e:
        print(f"❌ Inference testing failed: {e}")
        return False

def main():
    """Main training pipeline"""
    print("🚀 Starting Improved Solar Panel Detection Training Pipeline")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("data.yaml"):
        print("❌ data.yaml not found. Please run this script in the dataset directory.")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return
    
    # Download dataset (if needed)
    if not os.path.exists("train") or not os.path.exists("valid"):
        if not download_dataset():
            print("❌ Failed to download dataset")
            return
    
    # Create optimized configuration
    config_path = create_optimized_config()
    
    # Train model
    results = train_model(config_path)
    if results is None:
        print("❌ Training failed")
        return
    
    # Tune hyperparameters
    if tune_hyperparameters():
        print("✅ Hyperparameter tuning completed")
    
    # Validate model
    if validate_model():
        print("✅ Model validation completed")
    
    # Test inference
    if test_inference():
        print("✅ Inference testing completed")
    
    # Export models
    if export_models():
        print("✅ Model export completed")
    
    print("\n🎉 Training pipeline completed successfully!")
    print("\n📁 Output files:")
    print("   - Trained model: solar_detection_improved/yolo11n_solar_optimized/weights/best.pt")
    print("   - Tuned model: solar_detection_improved/yolo11n_solar_tuned/weights/best.pt")
    print("   - ONNX models: models/")
    print("   - Training plots: solar_detection_improved/*/")
    print("\n🔧 Next steps:")
    print("   1. Test the new models in your backend")
    print("   2. Compare performance with current models")
    print("   3. Update server.prod.fixed.js to use new models")

if __name__ == "__main__":
    main() 