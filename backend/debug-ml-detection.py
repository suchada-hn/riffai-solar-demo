#!/usr/bin/env python3
"""
Debug script for ML detection - tests each component step by step
"""

import os
import sys
import json

def test_environment():
    """Test basic environment"""
    print("🔍 Testing Environment...")
    
    # Check Python version
    print(f"   Python version: {sys.version}")
    
    # Check current directory
    print(f"   Current directory: {os.getcwd()}")
    
    # Check if we're in the right place
    if not os.path.exists("package.json"):
        print("   ❌ Not in backend directory")
        return False
    
    print("   ✅ In backend directory")
    return True

def test_onnx_runtime():
    """Test ONNX Runtime availability"""
    print("\n🔍 Testing ONNX Runtime...")
    
    try:
        import onnxruntime as ort
        print(f"   ✅ ONNX Runtime available: {ort.__version__}")
        return True
    except ImportError as e:
        print(f"   ❌ ONNX Runtime not available: {e}")
        return False

def test_models():
    """Test if ONNX models exist and can be loaded"""
    print("\n🔍 Testing ONNX Models...")
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("   ❌ Models directory not found")
        return False
    
    # Check for specific models
    solar_model = "models/best-solar-panel.onnx"
    pool_model = "models/pool-best.onnx"
    
    if not os.path.exists(solar_model):
        print(f"   ❌ Solar model not found: {solar_model}")
        return False
    
    if not os.path.exists(pool_model):
        print(f"   ❌ Pool model not found: {pool_model}")
        return False
    
    print("   ✅ ONNX models found")
    
    # Try to load models
    try:
        import onnxruntime as ort
        
        print("   🔄 Loading solar panel model...")
        solar_session = ort.InferenceSession(solar_model)
        print("   ✅ Solar panel model loaded")
        
        print("   🔄 Loading pool model...")
        pool_session = ort.InferenceSession(pool_model)
        print("   ✅ Pool model loaded")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return False

def test_detection_script():
    """Test if the detection script exists and works"""
    print("\n🔍 Testing Detection Script...")
    
    script_path = "run-solar-panel-and-pool-detection-onnx-only.py"
    
    if not os.path.exists(script_path):
        print(f"   ❌ Detection script not found: {script_path}")
        return False
    
    print(f"   ✅ Detection script found: {script_path}")
    
    # Check file permissions
    if os.access(script_path, os.R_OK):
        print("   ✅ Script is readable")
    else:
        print("   ❌ Script is not readable")
        return False
    
    if os.access(script_path, os.X_OK):
        print("   ✅ Script is executable")
    else:
        print("   ❌ Script is not executable")
    
    return True

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\n🔍 Testing Dependencies...")
    
    try:
        import cv2
        print("   ✅ OpenCV available")
    except ImportError as e:
        print(f"   ❌ OpenCV not available: {e}")
        return False
    
    try:
        import numpy as np
        print("   ✅ NumPy available")
    except ImportError as e:
        print(f"   ❌ NumPy not available: {e}")
        return False
    
    try:
        from PIL import Image
        print("   ✅ Pillow available")
    except ImportError as e:
        print(f"   ❌ Pillow not available: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 ML Detection Debug Test")
    print("=" * 40)
    
    tests = [
        ("Environment", test_environment),
        ("ONNX Runtime", test_onnx_runtime),
        ("ONNX Models", test_models),
        ("Detection Script", test_detection_script),
        ("Dependencies", test_dependencies)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"   ❌ {test_name} test failed with error: {e}")
            results[test_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests PASSED! ML detection should work.")
        print("   If you're still getting errors, check the deployment.")
    else:
        print("❌ Some tests FAILED! Fix these issues first.")
        print("   Check the errors above for specific problems.")
    
    # Return JSON result for the server
    result = {
        "success": all_passed,
        "tests": results,
        "message": "ML detection debug test completed"
    }
    
    print(f"\n📋 JSON Result: {json.dumps(result, indent=2)}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 