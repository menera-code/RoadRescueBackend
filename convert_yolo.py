# convert_yolo.py — run once, then delete
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False
)
print("✅ Exported to yolov8n.onnx")