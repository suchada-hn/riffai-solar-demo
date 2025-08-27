#!/usr/bin/env python3
"""
Test script for improved ONNX detection
"""

import os
import sys
import json

def test_improved_detection():
    """Test the improved detection script"""
    print("🧪 Testing Improved ONNX Detection")
    print("=" * 40)
    
    # Check if improved script exists
    script_path = "run-solar-panel-and-pool-detection-improved.py"
    if not os.path.exists(script_path):
        print(f"❌ Improved script not found: {script_path}")
        return False
    
    print(f"✅ Improved script found: {script_path}")
    
    # Check if models exist
    models = ["models/best-solar-panel.onnx", "models/pool-best.onnx"]
    for model in models:
        if not os.path.exists(model):
            print(f"❌ Model not found: {model}")
            return False
        print(f"✅ Model found: {model}")
    
    # Check if we can import required modules
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime available")
    except ImportError as e:
        print(f"❌ ONNX Runtime not available: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError as e:
        print(f"❌ OpenCV not available: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError as e:
        print(f"❌ NumPy not available: {e}")
        return False
    
    # Test model loading
    try:
        print("\n🔄 Testing model loading...")
        solar_session = ort.InferenceSession("models/best-solar-panel.onnx")
        pool_session = ort.InferenceSession("models/pool-best.onnx")
        print("✅ Models loaded successfully")
        
        # Test inference with dummy data
        print("🔄 Testing inference...")
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
        
        solar_output = solar_session.run(None, {solar_session.get_inputs()[0].name: dummy_input})
        pool_output = pool_session.run(None, {pool_session.get_inputs()[0].name: dummy_input})
        
        print(f"✅ Solar model inference: {solar_output[0].shape}")
        print(f"✅ Pool model inference: {pool_output[0].shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Improved detection should work.")
    return True

def main():
    """Main test function"""
    success = test_improved_detection()
    
    result = {
        "success": success,
        "message": "Improved ONNX detection test completed",
        "ready_for_deployment": success
    }
    
    print(f"\n📋 Test Result: {json.dumps(result, indent=2)}")
    
    if success:
        print("\n🚀 Ready to deploy to Google Cloud!")
        print("   Push to google-cloud branch or run manual deployment")
    else:
        print("\n❌ Fix issues before deployment")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 