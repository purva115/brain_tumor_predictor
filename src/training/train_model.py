#This file will train your CNN on the dataset and save it.

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
# from src.utils.dataset_loader import BrainMRIDataset
from src.utils.dataloader import BrainMRIDataset
from src.models.cnn_classifier import BrainMRIClassifier
from sklearn.metrics import accuracy_score

# Paths
DATASET_PATH = "data/BrainMRI"
MODEL_SAVE_PATH = "models/brain_tumor_model.pth"

# 1. Load dataset
dataset = BrainMRIDataset(DATASET_PATH)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# 2. Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BrainMRIClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# 3. Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Training loss = {avg_loss:.4f}")

# 4. Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# 5. Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 Model saved to {MODEL_SAVE_PATH}")
