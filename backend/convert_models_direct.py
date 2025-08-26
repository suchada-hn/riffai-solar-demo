#!/usr/bin/env python3
"""
Direct YOLOv8 to ONNX converter that bypasses PyTorch security restrictions
"""

import os
import sys
import torch
from pathlib import Path

def convert_model_direct(model_path, output_dir="models"):
    """
    Convert model directly using Python with workarounds for PyTorch security
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model name
        model_name = Path(model_path).stem
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        print(f"🔄 Converting {model_path} to ONNX...")
        
        # Set environment variables to handle PyTorch security
        os.environ['TORCH_WEIGHTS_ONLY'] = 'false'
        
        # Import ultralytics after setting environment
        from ultralytics import YOLO
        
        # Try to load the model with explicit weights_only=False
        print(f"   Loading model...")
        
        # Use a workaround: load with torch.load first, then create YOLO object
        try:
            # Load the model weights directly
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Create a new YOLO model and load the weights
            model = YOLO('yolov8n.pt')  # Load a base model first
            model.model.load_state_dict(checkpoint['model'], strict=False)
            
            print(f"   Model loaded successfully!")
            
        except Exception as e:
            print(f"   ⚠️  Direct loading failed: {e}")
            print(f"   Trying alternative method...")
            
            # Alternative: try to load with YOLO directly
            model = YOLO(model_path)
        
        # Export to ONNX
        print(f"   Exporting to ONNX...")
        success = model.export(
            format='onnx',
            dynamic=True,
            simplify=True,
            opset=11,
            half=False,
            imgsz=640
        )
        
        if success:
            # Check if ONNX file was created
            if os.path.exists(f"{model_name}.onnx"):
                # Move to models directory
                os.rename(f"{model_name}.onnx", onnx_path)
                print(f"✅ Model converted successfully!")
                print(f"   ONNX model saved to: {onnx_path}")
                
                # Get file sizes
                pt_size = os.path.getsize(model_path) / (1024 * 1024)
                onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
                
                print(f"   Original PT size: {pt_size:.1f} MB")
                print(f"   ONNX size: {onnx_size:.1f} MB")
                print(f"   Size reduction: {((pt_size - onnx_size) / pt_size * 100):.1f}%")
                
                return True
            else:
                print(f"❌ ONNX file not created")
                return False
        else:
            print(f"❌ Export failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def main():
    """Main conversion function"""
    print("🚀 Direct YOLOv8 to ONNX Converter")
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
        return
    
    print(f"📁 Found {len(existing_models)} model(s) to convert:")
    for model in existing_models:
        print(f"   - {model}")
    
    print(f"\n🔄 Starting conversion process...")
    
    # Convert each model
    success_count = 0
    for model in existing_models:
        print(f"\n{'='*30}")
        print(f"Converting: {model}")
        print(f"{'='*30}")
        
        if convert_model_direct(model):
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