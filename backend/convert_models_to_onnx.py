#!/usr/bin/env python3
"""
Convert YOLOv8 models to ONNX format for faster loading and deployment
ONNX models load much faster and are more deployment-friendly
"""

import os
import sys
from pathlib import Path

# Set environment variable to handle PyTorch 2.6+ weights_only issue
os.environ['TORCH_WEIGHTS_ONLY'] = 'false'

def convert_model_to_onnx(model_path, output_dir="models"):
    """
    Convert a YOLOv8 model to ONNX format
    
    Args:
        model_path (str): Path to the .pt model file
        output_dir (str): Directory to save the ONNX model
    """
    try:
        from ultralytics import YOLO
        import torch
        import ultralytics
        
        # Handle PyTorch 2.6+ weights_only security feature
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            try:
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
                print("✅ Added ultralytics models to PyTorch safe globals")
            except ImportError:
                print("⚠️  Could not add ultralytics models to safe globals")
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ Error importing required libraries: {e}")
        print("Please install: pip install ultralytics torch")
        return False
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"🔄 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Get model info
        model_name = Path(model_path).stem
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        print(f"🔄 Converting to ONNX format...")
        print(f"   Input model: {model_path}")
        print(f"   Output model: {onnx_path}")
        
        # Export to ONNX with optimizations
        # Use a more compatible approach for newer PyTorch versions
        try:
            success = model.export(
                format='onnx',
                dynamic=True,  # Dynamic batch size for flexibility
                simplify=True,  # Simplify the model
                opset=11,      # ONNX opset version for compatibility
                half=False,    # Keep full precision for accuracy
                imgsz=640     # Input image size
            )
        except Exception as e:
            print(f"   ⚠️  Standard export failed, trying alternative method: {e}")
            # Try alternative export method
            success = model.export(
                format='onnx',
                dynamic=True,
                simplify=False,  # Disable simplification if it causes issues
                opset=11,
                half=False,
                imgsz=640
            )
        
        if success:
            print(f"✅ Model converted successfully!")
            print(f"   ONNX model saved to: {onnx_path}")
            
            # Get file sizes for comparison
            pt_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            print(f"   Original PT size: {pt_size:.1f} MB")
            print(f"   ONNX size: {onnx_size:.1f} MB")
            print(f"   Size reduction: {((pt_size - onnx_size) / pt_size * 100):.1f}%")
            
            return True
        else:
            print(f"❌ Model conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False

def main():
    """Main conversion function"""
    print("🚀 YOLOv8 to ONNX Model Converter")
    print("=" * 50)
    
    # Define models to convert
    models = [
        "best-solar-panel.pt",
        "pool-best.pt"
    ]
    
    # Check if models exist
    existing_models = []
    for model in models:
        if os.path.exists(model):
            existing_models.append(model)
        else:
            print(f"⚠️  Model not found: {model}")
    
    if not existing_models:
        print("❌ No models found to convert!")
        print("Please ensure you're in the backend directory with the .pt model files")
        return
    
    print(f"📁 Found {len(existing_models)} model(s) to convert:")
    for model in existing_models:
        print(f"   - {model}")
    
    print("\n🔄 Starting conversion process...")
    
    # Convert each model
    success_count = 0
    for model in existing_models:
        print(f"\n{'='*30}")
        print(f"Converting: {model}")
        print(f"{'='*30}")
        
        if convert_model_to_onnx(model):
            success_count += 1
        else:
            print(f"❌ Failed to convert {model}")
    
    print(f"\n{'='*50}")
    print(f"🎯 Conversion Summary:")
    print(f"   Total models: {len(existing_models)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(existing_models) - success_count}")
    
    if success_count > 0:
        print(f"\n✅ ONNX models are ready in the 'models/' directory!")
        print(f"   These models will load much faster than the original .pt files")
        print(f"   Update your detection scripts to use the .onnx models")
        
        # Show the created models
        models_dir = "models"
        if os.path.exists(models_dir):
            print(f"\n📁 ONNX models created:")
            for file in os.listdir(models_dir):
                if file.endswith('.onnx'):
                    file_path = os.path.join(models_dir, file)
                    size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   - {file} ({size:.1f} MB)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Update your detection scripts to use .onnx models")
    print(f"   2. Test the ONNX models for accuracy")
    print(f"   3. Deploy with faster loading times!")

if __name__ == "__main__":
    main() 