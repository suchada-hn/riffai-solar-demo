# 🚀 Solar Panel Detection Model Training Guide

## 🎯 **Overview**

This guide will help you create **better solar panel detection models** using YOLO11 that should significantly improve the detection quality in your deployed backend.

## 🔍 **Current Problem**

Your current ONNX models are detecting:
- **8 detections** vs **30+ detections** from PyTorch
- **False positives** (pools where there are none)
- **Lower confidence** values
- **Missing solar panels** due to sensitivity issues

## 🛠️ **Training Solutions**

### **Option 1: Simple Training (Recommended for Starters)**
```bash
# Run the simplified training script
python3 train_simple_solar.py
```

**Features:**
- ✅ **50 epochs** (faster training)
- ✅ **AdamW optimizer** (better convergence)
- ✅ **Early stopping** (prevents overfitting)
- ✅ **Automatic ONNX export**

### **Option 2: Advanced Training (For Best Results)**
```bash
# Run the comprehensive training script
python3 train_improved_solar_detection.py
```

**Features:**
- ✅ **100 epochs** (thorough training)
- ✅ **Hyperparameter tuning** (optimized settings)
- ✅ **Advanced data augmentation** (satellite-specific)
- ✅ **Comprehensive validation**

## 📋 **Prerequisites**

### **System Requirements**
- **Python 3.8+**
- **8GB+ RAM** (16GB recommended)
- **GPU** (optional but recommended)
- **50GB+ free disk space**

### **Dependencies**
```bash
# Install required packages
pip install ultralytics opencv-python matplotlib pillow numpy torch torchvision
```

## 🚀 **Training Steps**

### **Step 1: Prepare Environment**
```bash
cd backend
mkdir -p models
mkdir -p training_data
cd training_data
```

### **Step 2: Download Dataset**
The script automatically downloads the Roboflow solar panel dataset:
- **Source**: Satellite imagery of solar farms
- **Classes**: Solar panels, pools
- **Format**: YOLO format with annotations
- **Size**: ~1000+ images

### **Step 3: Start Training**
```bash
# Simple training (recommended first)
python3 ../train_simple_solar.py

# OR Advanced training
python3 ../train_improved_solar_detection.py
```

### **Step 4: Monitor Progress**
Training will show:
- 📊 **Loss curves** (should decrease over time)
- 📈 **mAP metrics** (should increase)
- 🎯 **Validation results** (every 10 epochs)
- 💾 **Model checkpoints** (saved automatically)

## 📊 **Expected Results**

### **Training Metrics**
- **mAP50**: 0.85+ (85% accuracy at 50% IoU)
- **mAP50-95**: 0.65+ (65% accuracy across IoU thresholds)
- **Precision**: 0.80+ (80% of detections are correct)
- **Recall**: 0.85+ (85% of objects are detected)

### **Detection Quality**
- **Solar Panels**: 25-35 detections (vs current 3-8)
- **Confidence**: 0.40-0.90 range (vs current 0.25-0.39)
- **False Positives**: <5% (vs current high rate)
- **Missing Detections**: <10% (vs current 70%+)

## 🔧 **Model Integration**

### **Step 1: Test New Models**
```bash
# Test the new ONNX model
python3 test_new_model.py
```

### **Step 2: Update Backend**
```python
# In server.prod.fixed.js, update model paths:
const solarModelPath = './models/best-solar-panel-improved.onnx';
const poolModelPath = './models/pool-best-improved.onnx';
```

### **Step 3: Deploy and Test**
- Deploy updated backend
- Test with same images
- Compare detection results
- Monitor performance metrics

## 📁 **Output Structure**

```
solar_detection_simple/
├── yolo11n_solar/
│   ├── weights/
│   │   ├── best.pt          # Best PyTorch model
│   │   └── last.pt          # Last checkpoint
│   ├── results.png           # Training curves
│   ├── confusion_matrix.png  # Confusion matrix
│   └── val_batch*.jpg       # Validation examples

models/
├── best-solar-panel-improved.onnx  # New ONNX model
└── pool-best-improved.onnx         # New pool model
```

## 🎯 **Training Tips**

### **For Better Results**
1. **Use GPU** if available (10x faster)
2. **Increase epochs** to 100+ for thorough training
3. **Adjust batch size** based on memory
4. **Monitor validation** to prevent overfitting
5. **Use data augmentation** for robustness

### **Common Issues**
- **Low mAP**: Increase training epochs
- **Overfitting**: Reduce model complexity or increase data
- **Memory errors**: Reduce batch size
- **Slow training**: Use GPU or reduce image size

## 🔄 **Iterative Improvement**

### **Cycle 1: Baseline Model**
- Train with default settings
- Evaluate performance
- Identify weaknesses

### **Cycle 2: Optimized Model**
- Adjust hyperparameters
- Add data augmentation
- Retrain with improvements

### **Cycle 3: Production Model**
- Fine-tune on specific data
- Optimize for deployment
- Validate thoroughly

## 📞 **Support & Troubleshooting**

### **Training Issues**
- **Out of memory**: Reduce batch size or image size
- **Low accuracy**: Check dataset quality and increase epochs
- **Slow convergence**: Adjust learning rate and optimizer

### **Integration Issues**
- **Model not loading**: Check file paths and ONNX compatibility
- **Poor performance**: Verify model format and preprocessing
- **Runtime errors**: Check ONNX Runtime version

## 🎉 **Success Metrics**

Your new model should achieve:
- ✅ **2-3x more detections** than current ONNX
- ✅ **Higher confidence** values (0.40-0.90)
- ✅ **Better accuracy** (mAP50 > 0.85)
- ✅ **Fewer false positives** (<5%)
- ✅ **Faster inference** (optimized ONNX)

## 🚀 **Next Steps**

1. **Start with simple training** (`train_simple_solar.py`)
2. **Evaluate results** and compare with current models
3. **Iterate and improve** using advanced training
4. **Integrate into backend** and test thoroughly
5. **Deploy and monitor** performance in production

---

**🎯 Goal**: Create models that match or exceed PyTorch performance while maintaining ONNX deployment compatibility.

**⏱️ Timeline**: 2-4 hours for training, 1-2 hours for testing and integration.

**💡 Pro Tip**: Start with the simple training script to get familiar with the process, then move to advanced training for best results. 