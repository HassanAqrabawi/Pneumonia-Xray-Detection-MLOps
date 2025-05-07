import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import numpy as np
import mlflow
import mlflow.pytorch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
base_dir = 'chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Transforms
IMG_SIZE = 150

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Grayscale normalization
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class mapping
class_names = train_dataset.classes
print("Classes:", class_names)

# Load ResNet18 and modify classifier
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# MLflow start
mlflow.set_experiment("pneumonia_detection")
with mlflow.start_run():
    for epoch in range(5):  # You can increase this
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        y_true, y_pred, y_scores = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(inputs)
                scores = torch.sigmoid(outputs)

                preds = scores > 0.5

                y_true.extend(labels.cpu().numpy().squeeze().flatten())
                y_pred.extend(preds.cpu().numpy().squeeze().flatten())
                y_scores.extend(scores.cpu().numpy().squeeze().flatten())


        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        # Compute metrics
        val_acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_scores)

        # Optional: Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)

        print(f"Epoch {epoch+1}:")
        print(f"  Loss = {avg_train_loss:.4f}")
        print(f"  Validation Metrics:")
        print(f"    Accuracy  = {val_acc:.4f}")
        print(f"    Precision = {precision:.4f}")
        print(f"    Recall    = {recall:.4f}")
        print(f"    F1 Score  = {f1:.4f}")
        print(f"    ROC AUC   = {roc_auc:.4f}")

        # Log to MLflow
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_precision", precision, step=epoch)
        mlflow.log_metric("val_recall", recall, step=epoch)
        mlflow.log_metric("val_f1", f1, step=epoch)
        mlflow.log_metric("val_roc_auc", roc_auc, step=epoch)

    # Save model to MLflow
    mlflow.pytorch.log_model(model, "model")
