import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
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

def train_and_log_model(model, model_name, device, train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset, num_epochs=5, lr=1e-4, batch_size=32, img_size=255):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    mlflow.set_experiment("pneumonia_detection")
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\n[MLflow] Run ID: {run.info.run_id} for {model_name}")
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "image_size": img_size,
            "model": model_name,
            "optimizer": "Adam",
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "classes": class_names
        })
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} [{model_name}]")
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

def get_dataloaders_and_classes():
    base_dir = 'chest_xray_split'
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    IMG_SIZE = 255
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    print("Starting training script...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset = get_dataloaders_and_classes()

    # ResNet18
    resnet = models.resnet18(weights='DEFAULT')
    resnet.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(resnet.fc.in_features, 1))
    resnet = resnet.to(device)
    train_and_log_model(resnet, "ResNet18", device, train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset)

    # MobileNetV2
    mobilenet = models.mobilenet_v2(weights='DEFAULT')
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 1)
    mobilenet = mobilenet.to(device)
    train_and_log_model(mobilenet, "MobileNetV2", device, train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset)

    # EfficientNetB0
    efficientnet = models.efficientnet_b0(weights='DEFAULT')
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 1)
    efficientnet = efficientnet.to(device)
    train_and_log_model(efficientnet, "EfficientNetB0", device, train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset)
