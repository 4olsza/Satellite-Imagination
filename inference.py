"""
Pix2Pix Inference Script.
Loads a trained generator model and generates satellite images from input maps.
"""

import sys
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ensure the source directory is in the system path for internal imports
sys.path.insert(0, str(Path(__file__).parent))
from src.models.generator import Generator


class Pix2PixInference:
    """
    A pipeline class to handle single and batch inference using a trained Pix2Pix Generator.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        features: int = 64,
    ):
        """
        Initializes the generator model architecture and loads the trained weights.

        Args:
            checkpoint_path (str): Path to the saved model checkpoint (.pth or .pth.tar).
            device (str): Device to run the inference on ('cuda' or 'cpu').
            features (int): Base number of channels/filters in the generator layers.
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Initialize the generator structure and transfer to target device
        self.generator = Generator(in_channels=3, features=features).to(self.device)

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load model weights safely, handling both standalone state dicts and training checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["state_dict"])
        else:
            self.generator.load_state_dict(checkpoint)

        self.generator.eval()
        logger.info(f"✓ Model successfully loaded from: {checkpoint_path}")

        # Input transformation: Convert to tensor and normalize to [-1, 1] range
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Preprocesses an input image, preparing it for the generator network.
        Automatically handles and crops side-by-side paired images (Satellite + Map).

        Args:
            image_path (str): Path to the input image file.

        Returns:
            torch.Tensor: Preprocessed image tensor with shape [1, 3, 256, 256].
        """
        img = Image.open(image_path).convert("RGB")
        
        # Fallback mechanism for raw dataset images where the map is on the right side
        width, height = img.size
        if width > height:
            img = img.crop((width // 2, 0, width, height))

        # Resize to standard network resolution (256x256) using default bilinear interpolation
        img = img.resize((256, 256))
        img_tensor = self.transform(img).unsqueeze(0) # type: ignore
        return img_tensor.to(self.device)

    def _postprocess(self, output_tensor: torch.Tensor) -> Image.Image:
        """
        Converts the normalized output tensor back into a viewable PIL Image.

        Args:
            output_tensor (torch.Tensor): Generator raw output tensor with shape [1, 3, 256, 256].

        Returns:
            Image.Image: Postprocessed RGB PIL Image.
        """
        output_tensor = output_tensor.squeeze(0)
        
        # Denormalize from [-1, 1] back to standard [0, 1] range
        output_tensor = (output_tensor + 1) / 2
        output_tensor = output_tensor.clamp(0, 1)

        # Move tensor to CPU memory and convert to a standard NumPy uint8 array
        output_tensor = output_tensor.cpu().detach()
        output_array = (output_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        output_image = Image.fromarray(output_array)
        return output_image

    def predict(self, image_path: str) -> Image.Image:
        """
        Generates a synthetic satellite view for a single input map image.

        Args:
            image_path (str): Path to the input map image.

        Returns:
            Image.Image: Generated satellite imagery as a PIL Image.
        """
        with torch.no_grad():
            input_tensor = self._preprocess(image_path)
            output_tensor = self.generator(input_tensor)
            output_image = self._postprocess(output_tensor)

        return output_image

    def predict_batch(self, image_paths: List[str], output_dir: str) -> None:
        """
        Processes a batch of map images, saving all generated results to a directory.

        Args:
            image_paths (List[str]): List of file paths to process.
            output_dir (str): Target directory to save the generated images.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Generating imagery for {len(image_paths)} files...")
        for img_path in tqdm(image_paths):
            try:
                output_image = self.predict(img_path)
                img_name = Path(img_path).stem
                output_file = output_path / f"{img_name}_generated.png"
                output_image.save(output_file)
                logger.debug(f"  ✓ Saved: {output_file}")
            except Exception as e:
                logger.error(f"  ❌ Error processing {img_path}: {e}")

        logger.info(f"✓ Batch processing completed! Results saved to: {output_dir}")

    def generate_comparison(self, image_path: str, output_path: str) -> None:
        """
        Creates a side-by-side comparison image: Input Map vs. Generated Satellite.

        Args:
            image_path (str): Path to the input map image.
            output_path (str): Target path to save the comparison image.
        """
        from PIL import ImageDraw

        map_img = Image.open(image_path).convert("RGB")
        
        # Handle side-by-side paired dataset images to avoid aspect ratio squashing
        width, height = map_img.size
        if width > height:
            map_img = map_img.crop((width // 2, 0, width, height))
            
        map_img = map_img.resize((256, 256))
        sat_img = self.predict(image_path)

        # Create a blank double-width canvas and paste both images
        comparison = Image.new("RGB", (512, 256))
        comparison.paste(map_img, (0, 0))
        comparison.paste(sat_img, (256, 0))

        # Add descriptive labels onto the visual canvas
        draw = ImageDraw.Draw(comparison)
        try:
            draw.text((10, 10), "MAP", fill=(255, 255, 255))
            draw.text((266, 10), "GENERATED SATELLITE", fill=(255, 255, 255))
        except Exception:
            pass

        comparison.save(output_path)
        logger.info(f"✓ Comparison layout saved to: {output_path}")


def main():
    """
    Command-line execution entry point for running the Pix2Pix inference pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pix2Pix Model Inference CLI")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the trained model checkpoint file"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to an input image file or a directory of images"
    )
    parser.add_argument(
        "--output", type=str, default="./output", help="Directory path to save the generated output"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device ('cuda' or 'cpu')")

    args = parser.parse_args()
    inference = Pix2PixInference(args.checkpoint, device=args.device)
    input_path = Path(args.input)

    if input_path.is_file():
        logger.info(f"📷 Processing single file mode for: {input_path}")
        output_img = inference.predict(str(input_path))
        output_file = Path(args.output) / f"{input_path.stem}_generated.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_file)
        logger.info(f"✓ Single output saved successfully: {output_file}")
        
    elif input_path.is_dir():
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
        inference.predict_batch([str(f) for f in image_files], args.output)
    else:
        raise FileNotFoundError(f"Provided input path does not exist: {args.input}")


if __name__ == "__main__":
    main()