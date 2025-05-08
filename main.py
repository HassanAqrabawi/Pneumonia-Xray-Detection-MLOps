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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to configure MLflow, but continue if it fails
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow_available = True
    logger.info("MLflow tracking server configured successfully")
except Exception as e:
    mlflow_available = False
    logger.warning(f"MLflow tracking server not available: {str(e)}")
    logger.warning("Continuing without MLflow tracking")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Paths
base_dir = 'chest_xray_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Verify directories exist
for dir_path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

# Transforms
IMG_SIZE = 255

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
logger.info(f"Classes: {class_names}")

# Load ResNet18 and modify classifier
model = models.resnet18(weights='DEFAULT')  # Updated to use new weights parameter
# Add dropout before the final fully connected layer
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # 50% dropout
    nn.Linear(model.fc.in_features, 1)
)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)  # Add weight decay

def main():
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        # Log parameters
        mlflow.log_params({
            "learning_rate": 1e-4,
            "batch_size": BATCH_SIZE,
            "image_size": IMG_SIZE,
            "model": "ResNet18",
            "optimizer": "Adam",
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "classes": class_names
        })

        num_epochs = 5
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
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
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_scores = np.array(y_scores)
            val_acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_scores)
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:\n", cm)
            print(f"  Loss = {avg_train_loss:.4f}")
            print(f"  Validation Metrics:")
            print(f"    Accuracy  = {val_acc:.4f}")
            print(f"    Precision = {precision:.4f}")
            print(f"    Recall    = {recall:.4f}")
            print(f"    F1 Score  = {f1:.4f}")
            print(f"    ROC AUC   = {roc_auc:.4f}")
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_accuracy": val_acc,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "val_roc_auc": roc_auc
            }, step=epoch)
            print(f"[MLflow] Metrics logged for epoch {epoch+1}")

        # Save model to MLflow
        mlflow.pytorch.log_model(model, "model")
        print("[MLflow] Model saved.")

        # Test set evaluation
        model.eval()
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                scores = torch.sigmoid(outputs)
                preds = scores > 0.5
                y_true.extend(labels.cpu().numpy().squeeze().flatten())
                y_pred.extend(preds.cpu().numpy().squeeze().flatten())
                y_scores.extend(scores.cpu().numpy().squeeze().flatten())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        test_acc = accuracy_score(y_true, y_pred)
        test_precision = precision_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)
        test_roc_auc = roc_auc_score(y_true, y_scores)
        print("\nTest Set Evaluation:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test ROC AUC: {test_roc_auc:.4f}")
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_roc_auc": test_roc_auc
        })
        print("[MLflow] Test metrics logged.")
        print(f"[MLflow] Run complete. All metrics logged. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    print("Starting training script...")
    main()
