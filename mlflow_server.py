import os
import subprocess
import time
import webbrowser
from pathlib import Path

def start_mlflow_server():
    # Create directories for MLflow
    mlflow_dir = Path("mlflow_data")
    mlflow_dir.mkdir(exist_ok=True)
    
    # Set environment variables for MLflow
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    
    # Start MLflow server
    server_process = subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", "sqlite:///mlflow_data/mlflow.db",
            "--default-artifact-root", "./mlflow_data/artifacts",
            "--host", "0.0.0.0",
            "--port", "5000"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Open MLflow UI in browser
    webbrowser.open("http://localhost:5000")
    
    print("MLflow server started at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    start_mlflow_server() 