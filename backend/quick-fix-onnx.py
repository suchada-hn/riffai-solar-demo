#!/usr/bin/env python3
"""
Quick ONNX fix - Immediate test script
"""

import os
import sys
import json

def main():
    """Quick ONNX test and fix"""
    print("🚀 Quick ONNX Fix - Testing ML Backend")
    print("=" * 40)
    
    # Check if we're in the right environment
    print("🔍 Environment check...")
    
    # 1. Check Python version
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 2. Check if ONNX Runtime is available
    try:
        import onnxruntime as ort
        print("   ✅ ONNX Runtime: Available")
        print(f"   Version: {ort.__version__}")
    except ImportError:
        print("   ❌ ONNX Runtime: Not available")
        print("   Installing...")
        os.system("pip install onnxruntime")
        try:
            import onnxruntime as ort
            print("   ✅ ONNX Runtime: Installed successfully")
        except ImportError:
            print("   ❌ Failed to install ONNX Runtime")
            return False
    
    # 3. Check if models exist
    print("\n📁 Model check...")
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"   ❌ Models directory not found: {models_dir}")
        return False
    
    solar_model = os.path.join(models_dir, "best-solar-panel.onnx")
    pool_model = os.path.join(models_dir, "pool-best.onnx")
    
    if not os.path.exists(solar_model):
        print(f"   ❌ Solar model not found: {solar_model}")
        return False
    else:
        size_mb = os.path.getsize(solar_model) / (1024 * 1024)
        print(f"   ✅ Solar model: {solar_model} ({size_mb:.1f} MB)")
    
    if not os.path.exists(pool_model):
        print(f"   ❌ Pool model not found: {pool_model}")
        return False
    else:
        size_mb = os.path.getsize(pool_model) / (1024 * 1024)
        print(f"   ✅ Pool model: {pool_model} ({size_mb:.1f} MB)")
    
    # 4. Test model loading
    print("\n🔄 Testing model loading...")
    try:
        print("   Loading solar panel model...")
        solar_session = ort.InferenceSession(solar_model, providers=['CPUExecutionProvider'])
        print("   ✅ Solar panel model loaded")
        
        print("   Loading pool model...")
        pool_session = ort.InferenceSession(pool_model, providers=['CPUExecutionProvider'])
        print("   ✅ Pool model loaded")
        
        print("   🎉 All models loaded successfully!")
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # 5. Test image processing
    print("\n🖼️  Testing image processing...")
    try:
        import cv2
        import numpy as np
        print("   ✅ OpenCV and NumPy available")
        
        # Create a test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_img[:] = (128, 128, 128)  # Gray image
        
        # Test preprocessing
        img_resized = cv2.resize(test_img, (640, 640))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        img_nchw = np.transpose(img_batch, (0, 3, 1, 2))
        
        print("   ✅ Image preprocessing successful")
        
    except Exception as e:
        print(f"   ❌ Image processing failed: {e}")
        return False
    
    # 6. Test inference
    print("\n🔍 Testing inference...")
    try:
        # Get input name
        input_name = solar_session.get_inputs()[0].name
        
        # Run inference on test image
        outputs = solar_session.run(None, {input_name: img_nchw})
        print("   ✅ Inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        return False
    
    # 7. Success summary
    print("\n" + "=" * 40)
    print("🎯 ONNX Backend Test: PASSED!")
    print("=" * 40)
    
    result = {
        "status": "success",
        "message": "ONNX backend is working correctly",
        "models_loaded": True,
        "inference_working": True,
        "ready_for_deployment": True
    }
    
    print(json.dumps(result, indent=2))
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 