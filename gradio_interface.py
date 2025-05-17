import gradio as gr
import numpy as np
from PIL import Image
import requests
import torch
import mlflow.pyfunc

SERVE_PORT = 1234

class PneumoniaModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from torchvision import models
        import torch.nn as nn
        
        # Load the model
        self.model = models.mobilenet_v2(weights='DEFAULT')
        self.model.classifier[0] = nn.Dropout(p=0.2)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        self.model.eval()
        
        # Move to CPU to avoid CUDA issues
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
    
    def predict(self, context, model_input):
        import torch
        import pandas as pd
        
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values
        
        input_tensor = torch.tensor(model_input, dtype=torch.float32)
        
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.view(-1, 3, 255, 255)
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
            return [[prob]]

def preprocess_image(image: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((255, 255))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
    img_array = np.transpose(img_array, (2, 0, 1))  # [C, H, W]
    img_array = np.expand_dims(img_array, axis=0)   # [1, C, H, W]
    return img_array

def predict_gradio(image: Image.Image):
    try:
        processed = preprocess_image(image)
        payload = {"inputs": processed.tolist()}
        response = requests.post(
            f"http://localhost:{SERVE_PORT}/invocations",
            json=payload
        )

        if response.status_code == 200:
            prediction = response.json()
            prob = prediction['predictions'][0][0]
            result = "🦠 <b>PNEUMONIA</b>" if prob > 0.5 else "✅ <b>NORMAL</b>"
            confidence = prob if prob > 0.5 else 1 - prob
            return f"<div style='font-size:1.5em; text-align:center;'>{result}<br><span style='font-size:1em; color:#888;'>Confidence: {confidence*100:.2f}%</span></div>"
        else:
            return f"<span style='color:red;'>Server error: {response.text}</span>"
    except Exception as e:
        return f"<span style='color:red;'>Prediction failed: {str(e)}</span>"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🩻 Pneumonia Detection from Chest X-Ray
    <span style='font-size:1.2em;'>Upload a chest X-ray image to predict if <b>pneumonia</b> is present.</span>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Chest X-ray", elem_id="input-img")
        with gr.Column(scale=1):
            output = gr.HTML(label="Prediction Result", elem_id="output-box")
    submit_btn = gr.Button("Analyze X-ray", elem_id="analyze-btn")
    example_imgs = gr.Examples([
        # You can add example image paths here if you want
    ], inputs=[image_input])

    submit_btn.click(fn=predict_gradio, inputs=image_input, outputs=output)
    image_input.change(fn=lambda _: "", inputs=image_input, outputs=output)

demo.launch()
