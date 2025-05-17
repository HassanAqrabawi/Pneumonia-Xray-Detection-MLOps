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
import json
import time
from datetime import datetime
import logging
import pandas as pd

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_monitoring.log'),
        logging.StreamHandler()
    ]
)

# Use SQLite storage for persistent studies
storage = "sqlite:///optuna_study.db"

# Data pipeline setup (same as main.py)
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

def load_best_params(study_name='mobilenetv2_f1_tuning', storage=storage):
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study.best_trial.params

class ModelMonitor:
    def __init__(self, model_name='mobilenetv2_best', model_stage='Production'):
        self.model_name = model_name
        self.model_stage = model_stage
        self.metrics_history = []
        self.drift_threshold = 0.1  # 10% degradation threshold
        self.metrics_file = 'model_metrics_history.json'
        self.load_metrics_history()
        
    def load_metrics_history(self):
        """Load historical metrics if they exist"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics_history = json.load(f)
    
    def save_metrics_history(self):
        """Save metrics history to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f)
    
    def check_for_drift(self, current_metrics):
        """Check for model drift by comparing with historical performance"""
        if len(self.metrics_history) < 2:
            return
        
        # Get baseline metrics (average of first 10% of history)
        baseline_size = max(1, len(self.metrics_history) // 10)
        baseline_metrics = {}
        
        # Calculate mean for each metric separately
        for metric in ['accuracy', 'f1', 'roc_auc']:
            values = [m[metric] for m in self.metrics_history[:baseline_size]]
            baseline_metrics[metric] = np.mean(values)
        
        # Calculate drift
        drift = {}
        for metric in ['accuracy', 'f1', 'roc_auc']:
            current_value = current_metrics[metric]
            baseline_value = baseline_metrics[metric]
            relative_change = (current_value - baseline_value) / baseline_value
            
            if abs(relative_change) > self.drift_threshold:
                drift[metric] = relative_change
                logging.warning(f"Drift detected in {metric}: {relative_change:.2%} change")
        
        if drift:
            self.alert_drift(drift)
    
    def alert_drift(self, drift_metrics):
        """Alert when drift is detected"""
        alert_message = f"Model drift detected!\n"
        for metric, change in drift_metrics.items():
            alert_message += f"{metric}: {change:.2%} change\n"
        
        logging.warning(alert_message)
        # Here you could add additional alert mechanisms:
        # - Send email
        # - Create JIRA ticket
        # - Send Slack notification
        # - etc.
    
    def generate_performance_report(self):
        """Generate a performance report from metrics history"""
        if not self.metrics_history:
            return "No metrics history available"
        
        # Convert metrics history to DataFrame
        metrics_df = pd.DataFrame(self.metrics_history)
        
        # Remove timestamp column for calculations
        if 'timestamp' in metrics_df.columns:
            metrics_df = metrics_df.drop('timestamp', axis=1)
        
        report = {
            'current_performance': metrics_df.iloc[-1].to_dict() if not metrics_df.empty else {},
            'average_performance': metrics_df.mean().to_dict() if not metrics_df.empty else {},
            'performance_trend': {
                'accuracy': metrics_df['accuracy'].tolist() if not metrics_df.empty else [],
                'f1': metrics_df['f1'].tolist() if not metrics_df.empty else [],
                'roc_auc': metrics_df['roc_auc'].tolist() if not metrics_df.empty else []
            }
        }
        
        return report

def train_and_register_best_model(params, num_epochs=5, model_name='mobilenetv2_best'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset = get_dataloaders_and_classes()
    
    # Model
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier[0] = nn.Dropout(p=float(params['dropout']))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(params['lr']), weight_decay=float(params['weight_decay']))
    
    mlflow.set_experiment("mobilenetv2_optuna_tuning")
    with mlflow.start_run(run_name="Register_Best_Model") as run:
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
        
        # Evaluate on validation set
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
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_scores)
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Initialize monitoring and check for drift
        monitor = ModelMonitor(model_name=model_name)
        # Add timestamp to metrics history but not to MLflow metrics
        metrics_with_timestamp = {**metrics, 'timestamp': datetime.now().isoformat()}
        monitor.metrics_history.append(metrics_with_timestamp)
        monitor.save_metrics_history()
        monitor.check_for_drift(metrics_with_timestamp)
        
        # Log and register model
        mlflow.pytorch.log_model(model, "model", registered_model_name=model_name)
        
        # Transition model to Production stage
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_versions[0].version,
                stage="Production"
            )
        
        logging.info(f"Model registered as '{model_name}' in MLflow Model Registry.")
        logging.info(f"Validation Metrics: {metrics}")
        
        # Generate and log performance report
        report = monitor.generate_performance_report()
        logging.info(f"Performance report: {json.dumps(report, indent=2)}")
        
        print(f"Model registered as '{model_name}' in MLflow Model Registry.")
        print(f"Validation Metrics: {metrics}")
        print("\nTo serve the model, run:")
        print(f"mlflow models serve -m 'models:/{model_name}/Production' --host 0.0.0.0 --port 1234")
        print("\nTo monitor the model, check the model_monitoring.log file")

def main():
    print("Registering and training best MobileNetV2 model from Optuna study...")
    best_params = load_best_params(study_name='mobilenetv2_f1_tuning', storage=storage)
    print("Best hyperparameters:", best_params)
    train_and_register_best_model(best_params, num_epochs=5, model_name='mobilenetv2_best')

if __name__ == "__main__":
    main() 