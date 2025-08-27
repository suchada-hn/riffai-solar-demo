#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple YOLO11 Solar Panel Training Script
=========================================

A streamlined script to train YOLO11 models for solar panel detection.
This script focuses on the essential training steps without complex configurations.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_ultralytics():
    """Install ultralytics package"""
    print("🔧 Installing ultralytics...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✅ Ultralytics installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install ultralytics")
        return False

def download_dataset():
    """Download the solar panel dataset"""
    print("📥 Downloading solar panel dataset...")
    try:
        # Download dataset
        subprocess.check_call([
            "curl", "-L", "https://universe.roboflow.com/ds/dCRuFFJd7j?key=9SOXLBGP8y", 
            "-o", "roboflow.zip"
        ])
        
        # Extract dataset
        subprocess.check_call(["unzip", "roboflow.zip"])
        
        # Clean up
        os.remove("roboflow.zip")
        
        print("✅ Dataset downloaded and extracted")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Failed to download dataset")
        return False

def train_model():
    """Train the YOLO11 model"""
    print("🚀 Starting model training...")
    
    try:
        from ultralytics import YOLO
        
        # Load YOLO11n model
        model = YOLO("yolo11n.pt")
        print("✅ YOLO11n model loaded")
        
        # Train the model
        results = model.train(
            data="data.yaml",
            epochs=50,           # Reduced epochs for faster training
            imgsz=1280,
            batch=16,
            device="auto",
            project="solar_detection_simple",
            name="yolo11n_solar",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",   # Better optimizer
            lr0=0.001,          # Learning rate
            lrf=0.01,           # Final learning rate factor
            warmup_epochs=3,    # Warmup
            patience=15,        # Early stopping
            save_period=10,     # Save every 10 epochs
            plots=True,         # Save training plots
            val=True            # Validate during training
        )
        
        print("✅ Training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def export_to_onnx():
    """Export the trained model to ONNX"""
    print("📤 Exporting to ONNX...")
    
    try:
        from ultralytics import YOLO
        
        # Load trained model
        model_path = "solar_detection_simple/yolo11n_solar/weights/best.pt"
        if not os.path.exists(model_path):
            print("❌ Trained model not found")
            return False
        
        model = YOLO(model_path)
        
        # Export to ONNX
        onnx_path = model.export(
            format="onnx",
            dynamic=True,       # Dynamic batch size
            simplify=True,      # Simplify model
            opset=12,          # ONNX opset
            half=True,         # FP16 precision
            optimize=True      # Optimize
        )
        
        print(f"✅ ONNX model exported: {onnx_path}")
        
        # Copy to models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy2(onnx_path, models_dir / "best-solar-panel-improved.onnx")
        print("✅ Model copied to models directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def validate_model():
    """Validate the trained model"""
    print("🔍 Validating model...")
    
    try:
        from ultralytics import YOLO
        
        # Load trained model
        model_path = "solar_detection_simple/yolo11n_solar/weights/best.pt"
        if not os.path.exists(model_path):
            print("❌ Trained model not found")
            return False
        
        model = YOLO(model_path)
        
        # Validate
        metrics = model.val(
            data="data.yaml",
            imgsz=1280,
            batch=16,
            device="auto"
        )
        
        print("📊 Validation Results:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main training pipeline"""
    print("🚀 Simple Solar Panel Detection Training")
    print("=" * 40)
    
    # Check if data.yaml exists
    if not os.path.exists("data.yaml"):
        print("❌ data.yaml not found. Please run in dataset directory.")
        return
    
    # Install ultralytics
    if not install_ultralytics():
        return
    
    # Download dataset if needed
    if not os.path.exists("train") or not os.path.exists("valid"):
        if not download_dataset():
            return
    
    # Train model
    if not train_model():
        return
    
    # Validate model
    if not validate_model():
        return
    
    # Export to ONNX
    if not export_to_onnx():
        return
    
    print("\n🎉 Training completed successfully!")
    print("\n📁 Output:")
    print("   - Trained model: solar_detection_simple/yolo11n_solar/weights/best.pt")
    print("   - ONNX model: models/best-solar-panel-improved.onnx")
    print("\n🔧 Next: Test the new model in your backend!")

if __name__ == "__main__":
    main() 