from ultralytics import YOLO

# Load Model
model = YOLO('modelos/best.pt')

# Export Model
model.export(format='onnx')