#!/usr/bin/env python3
"""
Simple ONNX test script to verify models can be loaded
"""

import os
import sys

def test_onnx_models():
    """Test if ONNX models can be loaded"""
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime imported successfully")
        
        # Check if models exist
        solar_model = "models/best-solar-panel.onnx"
        pool_model = "models/pool-best.onnx"
        
        if not os.path.exists(solar_model):
            print(f"❌ Solar model not found: {solar_model}")
            return False
            
        if not os.path.exists(pool_model):
            print(f"❌ Pool model not found: {pool_model}")
            return False
            
        print("✅ ONNX models found")
        
        # Try to load models
        print("🔄 Loading solar panel model...")
        solar_session = ort.InferenceSession(solar_model, providers=['CPUExecutionProvider'])
        print("✅ Solar panel model loaded successfully")
        
        print("🔄 Loading pool model...")
        pool_session = ort.InferenceSession(pool_model, providers=['CPUExecutionProvider'])
        print("✅ Pool model loaded successfully")
        
        print("🎉 All ONNX models loaded successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ONNX Models")
    print("=" * 30)
    
    success = test_onnx_models()
    
    if success:
        print("\n🎯 ONNX test PASSED - Ready for deployment!")
        sys.exit(0)
    else:
        print("\n❌ ONNX test FAILED - Check dependencies!")
        sys.exit(1) 