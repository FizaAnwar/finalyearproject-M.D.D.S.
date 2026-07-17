# vit_train.py
# Beginner-friendly ViT-B/16 training for stroke vs non-stroke classification

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# =============================
# Device
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =============================
# Paths
# =============================
train_dir = r"dataset/data/train"
test_dir  = r"dataset/data/test"

# =============================
# Image Transforms
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # ViT-B/16 input size
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =============================
# Load Dataset
# =============================
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Classes:", train_dataset.classes)
print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

# =============================
# Load Pretrained ViT-B/16
# =============================
# Option 1: Download automatically from timm/HF (requires internet)
# model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

# Option 2: Offline (if you have downloaded weights manually)
# model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
# model.load_state_dict(torch.load(r"C:\path_to_downloaded_weights\vit_base_patch16_224.pth"))

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
model = model.to(device)

# =============================
# Loss & Optimizer
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# =============================
# Training Loop
# =============================
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f}")

# =============================
# Evaluation
# =============================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print("\nTest Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

torch.save(model.state_dict(), "vit_stroke_model.pth")
import json
with open("vit_class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)

# =============================
# Optional: Plot confusion matrix
# =============================
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

