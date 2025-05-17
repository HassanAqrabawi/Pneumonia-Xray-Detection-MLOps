import os
import subprocess
import time
import webbrowser
from pathlib import Path

def start_mlflow_server():
    # Start MLflow server with default backend (mlruns)
    server_process = subprocess.Popen(
        [
            "mlflow", "server",
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