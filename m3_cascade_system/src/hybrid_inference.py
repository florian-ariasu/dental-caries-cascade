import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import glob
import shutil

YOLO_PATH = 'yolo_medium_best.pt'
RESNET_PATH = 'resnet50_cropped_target_recall.pth'
TEST_IMAGES_DIR = 'training_data/unlabelled/xrays'
OUTPUT_DIR = 'rezultate_hibrid_M3'

CONF_YOLO = 0.15   
CONF_RESNET = 0.50 
PAD = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f"Model not found: {YOLO_PATH}")
yolo_model = YOLO(YOLO_PATH)

if not os.path.exists(RESNET_PATH):
    raise FileNotFoundError(f"Model not found: {RESNET_PATH}")

resnet_model = models.resnet50(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
resnet_model.to(DEVICE)
resnet_model.eval()

resnet_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_image(img_path):
    filename = os.path.basename(img_path)
    img_cv = cv2.imread(img_path)
    if img_cv is None: return

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    results = yolo_model.predict(img_path, conf=CONF_YOLO, classes=[1, 3], verbose=False)
    detected_boxes = []
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_yolo = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = "Caries" if cls_id == 1 else "Deep Caries"
            
            crop = pil_img.crop((
                max(0, x1 - PAD), 
                max(0, y1 - PAD), 
                min(pil_img.width, x2 + PAD), 
                min(pil_img.height, y2 + PAD)
            ))
            
            input_tensor = resnet_transforms(crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = resnet_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                prob_caries = probs[0][0].item() 
            
            is_confirmed = False
            
            if conf_yolo > 0.80:
                is_confirmed = True 
            elif prob_caries > CONF_RESNET:
                is_confirmed = True
                
            if is_confirmed:
                detected_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
                text = f"{label} Y:{conf_yolo:.2f} R:{prob_caries:.2f}"
                cv2.putText(img_cv, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(detected_boxes) > 0:
        save_path = os.path.join(OUTPUT_DIR, f"HYBRID_{filename}")
        cv2.imwrite(save_path, img_cv)

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
test_images.sort()

if not test_images:
    fallback_dir = 'training_data/quadrant-enumeration-disease/xrays'
    if os.path.exists(fallback_dir):
        test_images = glob.glob(os.path.join(fallback_dir, "*.png"))

print(f"Processing {len(test_images)} images...")

for img in test_images[:20]:
    analyze_image(img)

print(f"Process complete. Results in {OUTPUT_DIR}")