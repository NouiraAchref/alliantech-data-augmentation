from ultralytics import YOLO

# Load the YOLOv8 model (e.g., YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, or YOLOv8x)
model = YOLO('yolov8n.yaml')  # You can also use a pretrained model like 'yolov8n.pt'

# Train the model
model.train(
    data='yolov8_data/data.yaml',  # Path to your data.yaml file
    epochs=100,                    # Number of training epochs
    batch=16,                      # Batch size (adjust based on your GPU/CPU capacity)
    imgsz=640,                     # Image size for training
    name='yolov8_custom_model',     # Custom name for your model
    batch=8,
)
