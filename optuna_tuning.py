import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import mlflow
import mlflow.pytorch
import optuna

# Use SQLite storage for persistent studies
storage = "sqlite:///optuna_study.db"

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

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset = get_dataloaders_and_classes()

    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    num_epochs = 3  # Keep small for tuning speed

    # Model
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier[0] = nn.Dropout(p=dropout)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    mlflow.set_experiment("mobilenetv2_optuna_tuning")
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            'lr': lr,
            'dropout': dropout,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs
        })
        for epoch in range(num_epochs):
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
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_accuracy': val_acc,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'val_roc_auc': roc_auc
            }, step=epoch)
        # Return the main metric for Optuna to optimize
        return f1

def tune_hyperparameters(n_trials=10):
    mlflow.set_tracking_uri("http://localhost:5000")
    study = optuna.load_study(study_name='mobilenetv2_f1_tuning', storage=storage)
    with mlflow.start_run(run_name="Optuna_Tuning_Parent"):
        study.optimize(objective, n_trials=n_trials)
        print("Best trial:")
        print(study.best_trial)
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric('best_f1', study.best_trial.value)

if __name__ == "__main__":
    print("Starting Optuna hyperparameter tuning for MobileNetV2...")
    tune_hyperparameters(n_trials=10) 