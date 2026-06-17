import gradio as gr
import torch
from PIL import Image
import os
from inference import Pix2PixInference 

# Wymuszenie użycia układu scalonego Apple Silicon (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Uruchamianie silnika graficznego na: {device.upper()}")

# Dokładna nazwa Twojego 448 MB pliku
CHECKPOINT_PATH = "generator_epoch_500.pth.tar" 

print("Wczytywanie sieci neuronowej (to może zająć kilka sekund)...")
inference_model = Pix2PixInference(checkpoint_path=CHECKPOINT_PATH, device=device)
print("✅ Model gotowy do akcji!")

def generate_satellite(input_image: Image.Image) -> Image.Image:
    if input_image is None:
        return None
    
    # Tymczasowy zapis obrazka z interfejsu
    temp_path = "temp_input_gradio.png"
    input_image.save(temp_path)
    
    # Odpalenie inferencji
    output_image = inference_model.predict(temp_path)
    
    # Sprzątanie dysku
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return output_image

# Budowa ciemnego, nowoczesnego interfejsu
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #00b4d8;'>🛰️ Satellite Imagination</h1>
        <p style='text-align: center;'>Wgraj wektorowy szkic mapy po lewej stronie, a sieć wygeneruje fotorealistyczny obraz satelitarny po prawej.</p>
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Szkic Mapy (Wejście)")
            submit_btn = gr.Button("Generuj Satelitę", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(type="pil", label="Wygenerowany Obraz (Wyjście)", interactive=False)
            
    submit_btn.click(fn=generate_satellite, inputs=input_img, outputs=output_img)

if __name__ == "__main__":
    demo.launch()