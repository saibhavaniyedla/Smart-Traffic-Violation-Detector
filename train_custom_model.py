from ultralytics import YOLO

# Train your own all-9 violation detector after preparing annotations.
# Put dataset paths/classes in custom_data.yaml.
model = YOLO('yolov8n.pt')
model.train(
    data='custom_data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='traffic_violation_yolo_all9'
)

# After training, copy:
# runs/detect/traffic_violation_yolo_all9/weights/best.pt
# to:
# models/traffic_violation_yolo.pt
