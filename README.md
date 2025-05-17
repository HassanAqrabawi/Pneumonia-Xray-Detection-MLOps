# Pneumonia Detection using Deep Learning

This project implements a deep learning-based system for detecting pneumonia from chest X-ray images. It uses multiple state-of-the-art CNN architectures and includes experiment tracking, model serving, and web interfaces for easy interaction.

## Features

- Multiple model architectures (ResNet18, MobileNetV2, EfficientNetB0)
- MLflow integration for experiment tracking and model management
- Model serving capabilities
- Web interfaces using both Gradio and Streamlit
- Hyperparameter tuning using Optuna
- Comprehensive model evaluation metrics

## Project Structure

- `main.py`: Core training script with model implementations
- `optuna_tuning.py`: Hyperparameter optimization using Optuna
- `serve_model.py`: Model serving implementation
- `gradio_interface.py`: Web interface using Gradio
- `mlflow_server.py`: MLflow server configuration
- `register_and_serve.py`: Model registration and serving utilities

## Prerequisites

- Python 3.8+
- PyTorch
- MLflow
- Gradio
- Streamlit
- Optuna
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:

```bash
git clone https://github.com/HassanAqrabawi/Pneumonia-Xray-Detection-MLOps.git
cd Pneumonia-Xray-Detection-MLOps
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To train the models:

```bash
python main.py
```

### Hyperparameter Tuning

To perform hyperparameter optimization:

```bash
python optuna_tuning.py
```

### Running Web Interfaces

Gradio interface:

```bash
python gradio_interface.py
```

### Model Serving

To serve the model:

```bash
python serve_model.py
```

## Model Performance

The project implements three different architectures:

- ResNet18
- MobileNetV2
- EfficientNetB0

Each model is evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## MLflow Integration

The project uses MLflow for:

- Experiment tracking
- Model versioning
- Metric logging
- Model serving

To start the MLflow server:

```bash
python mlflow_server.py
```

### Model Registration and Serving

The `register_and_serve.py` script provides functionality to:
- Register the best performing model from MLflow experiments
- Serve the registered model as a REST API endpoint
- Monitor model performance in production

To register and serve the best model:
```bash
python register_and_serve.py
```

This will:
1. Query MLflow for the best model based on validation metrics
2. Register the model in the MLflow Model Registry
3. Start a local server to serve the model
4. The model will be available at `http://localhost:5000/invocations`

You can also serve a specific model version using MLflow directly:
```bash
mlflow models serve -m "models:/pneumonia_detection/Production" -p 5000
```

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle. The dataset contains:

- 5,863 X-Ray images (JPEG) in two categories:
  - Normal: 1,583 images
  - Pneumonia: 4,280 images
- Images are split into:
  - Training set: 70%
  - Validation set: 15%
  - Test set: 15%

The dataset is organized in the following structure:
```
chest_xray_split/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

To use the dataset:
1. Download it from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract the contents to the project root directory




