#!/usr/bin/env python3
import sys
import json
import os

def test_detection(image_path, latitude, longitude):
    """Simple test function that returns mock detections"""
    
    # Mock detection results for testing
    mock_detections = [
        {
            "class": 1,
            "name": "pool",
            "confidence": 0.85,
            "bbox": [100, 100, 300, 400],
            "latitude": float(latitude) if latitude else 0.0,
            "longitude": float(longitude) if longitude else 0.0
        },
        {
            "class": 2,
            "name": "solar_panel",
            "confidence": 0.92,
            "bbox": [400, 200, 600, 350],
            "latitude": float(latitude) if latitude else 0.0,
            "longitude": float(longitude) if longitude else 0.0
        }
    ]
    
    # Create output path
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_detections.jpg")
    
    # Output results as JSON
    output = {
        "detections": mock_detections,
        "location": {
            "latitude": float(latitude) if latitude else 0.0,
            "longitude": float(longitude) if longitude else 0.0
        },
        "detection_image": output_path
    }
    
    print(json.dumps(output))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_detection.py <image_path> [latitude] [longitude]")
        sys.exit(1)

    image_path = sys.argv[1]
    latitude = sys.argv[2] if len(sys.argv) > 2 else None
    longitude = sys.argv[3] if len(sys.argv) > 3 else None
    
    test_detection(image_path, latitude, longitude) 