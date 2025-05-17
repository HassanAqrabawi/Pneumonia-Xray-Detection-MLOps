import mlflow
import subprocess
import sys
import numpy as np
from PIL import Image
import logging
import requests
import os
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow settings
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "mobilenetv2_best"
MODEL_STAGE = "Production"
SERVE_PORT = 1234

def preprocess_image(image_path):
    """Preprocess an image for model prediction"""
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size
        image = image.resize((255, 255))
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Change to [C, H, W] format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict(image_path):
    """Make a prediction using the served model"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Prepare the request payload
        payload = {
            "inputs": processed_image.tolist()
        }
        
        # Make request to MLflow model server
        response = requests.post(
            f"http://localhost:{SERVE_PORT}/invocations",
            json=payload
        )
        
        if response.status_code == 200:
            prediction = response.json()
            probability = prediction[0][0]
            result = "PNEUMONIA" if probability > 0.5 else "NORMAL"
            confidence = probability if result == "PNEUMONIA" else (1 - probability)
            
            return {
                "diagnosis": result,
                "confidence": f"{confidence * 100:.2f}%"
            }
        else:
            logger.error(f"Error from model server: {response.text}")
            return {"error": f"Model server returned status {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {"error": str(e)}

def start_model_server():
    """Start the MLflow model server"""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Build the serve command
        cmd = [
            "mlflow", "models", "serve",
            "-m", f"models:/{MODEL_NAME}/{MODEL_STAGE}",
            "--host", "0.0.0.0",
            "--port", str(SERVE_PORT),
            "--no-conda"
        ]
        
        # Start the server
        logger.info(f"Starting model server on port {SERVE_PORT}...")
        process = subprocess.Popen(cmd)
        
        # Wait for server to start
        time.sleep(5)
        
        return process
        
    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        raise

def main():
    """Main function to start the model server"""
    try:
        # Start the model server
        server_process = start_model_server()
        logger.info("Server started successfully")
        
        # Keep the server running
        logger.info("Server is running. Press Ctrl+C to stop.")
        server_process.wait()
        
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server_process.terminate()
        server_process.wait()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        if 'server_process' in locals():
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main() 