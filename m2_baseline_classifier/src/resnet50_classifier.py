import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import random 
import matplotlib.pyplot as plt

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.mps.is_available():
        torch.mps.manual_seed(seed_value)

set_seed(42)
DATA_DIR = "data_cropped"                     
IMG_SIZE = 512                                
IMG_H = 256                              
BATCH_SIZE = 8
LR = 1e-4                                     
EPOCHS = 30
PATIENCE_ES = 5                               
PATIENCE_LR = 2                               
FACTOR_LR = 0.5 

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

train_losses = []
val_losses = []
train_accuracies = [] 
val_accuracies = []

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),       
    transforms.RandomRotation(10),                
    transforms.ColorJitter(brightness=0.1, contrast=0.1), 
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_transforms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   val_test_transforms)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  val_test_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes:", train_ds.classes)
print("Train distribution:", np.bincount(train_ds.targets))

weights = torch.tensor([0.60, 0.40], dtype=torch.float32).to(DEVICE) 
print("Manual Class weights (60/40):", weights)

criterion = nn.CrossEntropyLoss(weight=weights)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=PATIENCE_LR, factor=FACTOR_LR
)

def calculate_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return correct / total

best_loss = float("inf")
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - TRAIN"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * x.size(0)

    train_loss = total_train_loss / len(train_ds)
    train_losses.append(train_loss) 

    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            total_val_loss += loss.item() * x.size(0)

    val_loss = total_val_loss / len(val_ds)
    val_losses.append(val_loss) 
    
    train_acc = calculate_accuracy(train_loader, model, DEVICE)
    val_acc = calculate_accuracy(val_loader, model, DEVICE)
    train_accuracies.append(train_acc) 
    val_accuracies.append(val_acc) 

    scheduler.step(val_loss)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "resnet50_cropped_target_recall.pth")
        print(" -> Best model saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE_ES:
            print("EARLY STOPPING.")
            break

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs_run):
    epochs = range(1, epochs_run + 1)
    
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses[:epochs_run], label='Train Loss', marker='o', color='tab:blue')
    plt.plot(epochs, val_losses[:epochs_run], label='Validation Loss', marker='o', color='tab:orange')
    plt.title('Loss Curve (Training vs. Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    plt.show()
    print("Loss Graph saved as training_loss_curve.png")

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_accuracies[:epochs_run], label='Train Accuracy', marker='o', color='tab:red')
    plt.plot(epochs, val_accuracies[:epochs_run], label='Validation Accuracy', marker='o', color='tab:green')
    plt.title('Accuracy Curve (Training vs. Validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_accuracy_curve.png')
    plt.show()
    print("Accuracy Graph saved as validation_accuracy_curve.png")

print("\nLoading best model...")
model.load_state_dict(torch.load("resnet50_cropped_target_recall.pth", map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="TEST"):
        x = x.to(DEVICE)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
epochs_run = epoch - (1 if epochs_no_improve > 0 else 0)

print("\n=== FINAL TEST RESULTS ===")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("Confusion matrix:\n", cm)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_labels, all_preds, target_names=train_ds.classes))

plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs_run)
