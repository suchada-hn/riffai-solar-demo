import torch
from ultralytics import YOLO

# Load and convert model
model = YOLO('best-solar-panel.pt')
model.export(format='onnx', dynamic=True)