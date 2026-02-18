import os
import yaml
from ultralytics import YOLO
from google.colab import drive

ZIP_PATH = '/content/drive/MyDrive/dataset_yolo_lite.zip'
DATASET_DIR = '/content/datasets'
OUTPUT_MODEL_PATH = '/content/drive/MyDrive/yolo_medium_best.pt'

drive.mount('/content/drive')

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"Archive not found: {ZIP_PATH}")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.system(f"unzip -q {ZIP_PATH} -d /content/")

data_config = {
    'path': '/content',
    'train': 'images/train',
    'val': 'images/train',
    'names': {
        0: 'Impacted',
        1: 'Caries',
        2: 'Periapical Lesion',
        3: 'Deep Caries'
    },
    'nc': 4
}

with open('/content/data_colab.yaml', 'w') as f:
    yaml.dump(data_config, f)

model = YOLO('yolov8m.pt') 

print("Starting YOLOv8 Medium training...")
model.train(
    data='/content/data_colab.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='yolo_medium_m3',
    patience=10,
    verbose=True
)

src_model = '/content/runs/detect/yolo_medium_m3/weights/best.pt'
if os.path.exists(src_model):
    os.system(f"cp {src_model} {OUTPUT_MODEL_PATH}")
    print(f"Model saved successfully as: {OUTPUT_MODEL_PATH}")
else:
    print("Error: Could not find trained model weights.")