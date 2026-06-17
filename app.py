"""
Gradio Web Interface for Satellite Imagination.
Provides a local UI to interact with the trained Pix2Pix model.
"""

import gradio as gr
import torch
from PIL import Image
import os
import logging
from inference import Pix2PixInference 

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
logging.getLogger("httpx").setLevel(logging.WARNING)

# Force hardware acceleration via Apple Silicon (MPS) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Initializing inference engine on: {device.upper()}")

# Path to the pre-trained model weights
CHECKPOINT_PATH = "generator_epoch_208.pth.tar" 

print("Loading neural network weights...")
inference_model = Pix2PixInference(checkpoint_path=CHECKPOINT_PATH, device=device)
print("✅ Model is successfully loaded and ready!")

def generate_satellite(input_image: Image.Image) -> Image.Image:
    """
    Processes the input sketch and generates the corresponding satellite image.
    """
    if input_image is None:
        return None
    
    # CRITICAL STEP: Force a perfect 256x256 square to prevent geometry distortion 
    # before feeding it into the U-Net architecture.
    resized_input = input_image.resize((256, 256), Image.Resampling.LANCZOS)
    
    # Save the preprocessed image temporarily for the inference module
    temp_path = "temp_input_gradio.png"
    resized_input.save(temp_path)
    
    # Run the model inference
    output_image = inference_model.predict(temp_path)
    
    # Upscale the output for better UI visibility, strictly maintaining the 1:1 aspect ratio
    output_image = output_image.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Clean up the temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return output_image

# Build a symmetric GUI (Forcing 512x512 squares for both input and output)
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #00b4d8;'>🛰️ Satellite Imagination</h1>
        <p style='text-align: center;'>Wgraj wektorowy szkic mapy. Obraz zostanie automatycznie dopasowany do geometrii sieci.</p>
        """
    )
    
    with gr.Row():
        with gr.Column(): 
            # Fixed square container for the input map
            input_img = gr.Image(type="pil", label="Szkic Mapy (Wejście)", height=512, width=512)
            submit_btn = gr.Button("Generuj Satelitę", variant="primary")
            
        with gr.Column(): 
            # Identical fixed square container for the generated output - prevents aspect ratio stretching
            output_img = gr.Image(type="pil", label="Wygenerowany Obraz (Wyjście)", height=512, width=512, interactive=False)
            
    # Bind the function to the button click event
    submit_btn.click(fn=generate_satellite, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()