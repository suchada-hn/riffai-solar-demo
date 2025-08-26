#!/usr/bin/env python3
"""
Simple YOLOv8 to ONNX converter that handles PyTorch security restrictions
"""

import os
import sys
import subprocess
from pathlib import Path

def convert_with_ultralytics_cli(model_path, output_dir="models"):
    """
    Convert model using ultralytics CLI command to avoid PyTorch loading issues
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model name
        model_name = Path(model_path).stem
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        print(f"🔄 Converting {model_path} to ONNX...")
        
        # Use ultralytics CLI command
        cmd = [
            "yolo", "export", 
            "model=" + model_path,
            "format=onnx",
            "dynamic=True",
            "simplify=True",
            "opset=11",
            "half=False",
            "imgsz=640"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
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
            print(f"❌ Command failed with return code: {result.returncode}")
            print(f"   Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def main():
    """Main conversion function"""
    print("🚀 Simple YOLOv8 to ONNX Converter (CLI Method)")
    print("=" * 60)
    
    # Check if ultralytics CLI is available
    try:
        result = subprocess.run(["yolo", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ultralytics CLI not available. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
            print("✅ Ultralytics installed!")
    except Exception as e:
        print(f"❌ Error checking ultralytics CLI: {e}")
        print("Installing ultralytics...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
    
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
        print(f"\n{'='*40}")
        print(f"Converting: {model}")
        print(f"{'='*40}")
        
        if convert_with_ultralytics_cli(model):
            success_count += 1
        else:
            print(f"❌ Failed to convert {model}")
    
    print(f"\n{'='*60}")
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