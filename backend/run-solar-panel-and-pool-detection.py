import sys
import json
from ultralytics import YOLO
import cv2
import numpy as np
import os

# Model paths
POOL_MODEL_PATH = "./pool-best.pt"
SOLAR_PANEL_MODEL_PATH = "best-solar-panel.pt"

# Load models with proper PyTorch compatibility
try:
    import torch
    from torch.serialization import safe_globals
    import ultralytics.nn.tasks
    
    # Set models directory
    torch.hub.set_dir('./models')
    
    # Use safe_globals context manager for trusted models
    with safe_globals([ultralytics.nn.tasks.DetectionModel]):
        pool_model = YOLO(POOL_MODEL_PATH)
        solar_panel_model = YOLO(SOLAR_PANEL_MODEL_PATH)
    
    models_loaded = True
    print("AI models loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"Error loading models: {e}", file=sys.stderr)
    print("Trying alternative loading method...", file=sys.stderr)
    
    try:
        # Alternative: try with weights_only=False for trusted models
        import torch
        torch.hub.set_dir('./models')
        
        # Try to patch torch.load to use weights_only=False
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load
        
        pool_model = YOLO(POOL_MODEL_PATH)
        solar_panel_model = YOLO(SOLAR_PANEL_MODEL_PATH)
        
        # Restore original torch.load
        torch.load = original_load
        
        models_loaded = True
        print("AI models loaded with alternative method", file=sys.stderr)
    except Exception as e2:
        print(f"Failed to load models: {e2}", file=sys.stderr)
        models_loaded = False
        pool_model = None
        solar_panel_model = None

# Check if required modules are available
def check_dependencies():
    missing_modules = []
    
    try:
        import torch
    except ImportError:
        missing_modules.append("torch")
    
    try:
        import ultralytics
    except ImportError:
        missing_modules.append("ultralytics")
    
    try:
        import cv2
    except ImportError:
        missing_modules.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_modules.append("numpy")
    
    if missing_modules:
        print(f"Missing required modules: {', '.join(missing_modules)}", file=sys.stderr)
        print("Please install missing dependencies:", file=sys.stderr)
        print(f"pip install {' '.join(missing_modules)}", file=sys.stderr)
        return False
    
    return True

# Check dependencies at startup
if not check_dependencies():
    print("Critical dependencies missing. Exiting.", file=sys.stderr)
    sys.exit(1)

def draw_detections(image_path, detections, output_path):
    # read image
    image = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if image is None:
        print(f"Error: Could not load image {image_path}", file=sys.stderr)
        return None
    
    # draw bounding boxes
    for det in detections:
        bbox = det["bbox"]
        conf = det["confidence"]
        name = det["name"]
        
        # convert bbox to integers
        x1, y1, x2, y2 = map(int, bbox)
        
        # define colors
        color = (0, 0, 255) if name == "pool" else (255, 0, 0)
        
        # draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # save image with detections
    success = cv2.imwrite(output_path, image)
    if not success:
        print(f"Error: Could not save annotated image to {output_path}", file=sys.stderr)
        return None
    
    return output_path

def run_detection(image_path, model, label, name, latitude, longitude):
    if model is None or not models_loaded:
        print(f"Error: {name} model not loaded", file=sys.stderr)
        return []
    
    try:
        print(f"Running {name} detection on {image_path}", file=sys.stderr)
        # Use reasonable confidence threshold for production
        results = model.predict(image_path, conf=0.3, verbose=False)
        
        print(f"Raw results for {name}: {len(results)} result sets", file=sys.stderr)
        
        if not results or len(results) == 0:
            print(f"No detections found for {name}", file=sys.stderr)
            return []
        
        detections = []
        for i, result in enumerate(results):
            print(f"Result {i} has {len(result.boxes) if result.boxes else 0} boxes", file=sys.stderr)
            if result.boxes:
                for j, box in enumerate(result.boxes):
                    confidence = float(box.conf[0])
                    bbox = [int(x) for x in box.xyxy[0].tolist()]
                    
                    print(f"Box {j}: confidence={confidence:.3f}, bbox={bbox}", file=sys.stderr)
                    
                    detection = {
                        "class": label,
                        "name": name,
                        "confidence": confidence,
                        "bbox": bbox,
                        "latitude": latitude if latitude is not None else 0.0,
                        "longitude": longitude if longitude is not None else 0.0
                    }
                    detections.append(detection)
                    print(f"Found {name} with confidence {confidence:.3f}", file=sys.stderr)
        
        return detections
    except Exception as e:
        print(f"Error running detection with {name} model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run-solar-panel-and-pool-detection.py <image_path> [latitude] [longitude]")
        sys.exit(1)

    image_path = sys.argv[1]
    latitude = float(sys.argv[2]) if len(sys.argv) > 2 else None
    longitude = float(sys.argv[3]) if len(sys.argv) > 3 else None

    # Check if image file exists and is valid
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Error: {image_path} is not a valid image file", file=sys.stderr)
        sys.exit(1)

    # Run detections for both models
    pool_detections = run_detection(image_path, pool_model, 1, "pool", latitude, longitude)
    solar_panel_detections = run_detection(image_path, solar_panel_model, 2, "solar_panel", latitude, longitude)

    # Combine results
    combined_detections = pool_detections + solar_panel_detections

    if not combined_detections:
        print("No detections found", file=sys.stderr)
        # Return empty result but don't fail
        output = {
            "detections": [],
            "location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "detection_image": None
        }
        print(json.dumps(output))
        sys.exit(0)

    # Create output path
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_detections.jpg")

    # Draw detections on image and save
    detection_image_path = draw_detections(image_path, combined_detections, output_path)
    
    if detection_image_path is None:
        print("Warning: Could not create annotated image", file=sys.stderr)
        output_path = None

    # Output results as JSON
    output = {
        "detections": combined_detections,
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "detection_image": output_path
        }
    
    print(json.dumps(output))