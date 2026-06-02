"""
Inference script dla Pix2Pix
Wczytuje wytrenowany model i generuje satelity z map
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sys.path.insert(0, str(Path(__file__).parent))

from src.models.generator import Generator


class Pix2PixInference:
    """
    Klasa do inferencji Pix2Pix modelu.
    Wczytuje model i generuje obrazy satelitarne z map.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        features: int = 64,
    ):
        """
        Inicjalizacja modelu do inferencji.

        Args:
            checkpoint_path: Ścieżka do wytrenowanego modelu
            device: cuda lub cpu
            features: Feature maps dla generatora
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Wczytaj model
        self.generator = Generator(in_channels=3, features=features).to(self.device)

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint nie istnieje: {checkpoint_path}")

        # Wczytaj wagi
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            self.generator.load_state_dict(checkpoint)

        self.generator.eval()
        logger.info(f"✓ Model wczytany: {checkpoint_path}")

        # Transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Preprocessuje obraz do formatu modelu.

        Args:
            image_path: Ścieżka do obrazu

        Returns:
            Tensor w kształcie [1, 3, 256, 256]
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256), Image.BILINEAR)
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device)

    def _postprocess(self, output_tensor: torch.Tensor) -> Image.Image:
        """
        Postprocessuje wyjście modelu do obrazu.

        Args:
            output_tensor: Output z modelu [1, 3, 256, 256]

        Returns:
            PIL Image
        """
        output_tensor = output_tensor.squeeze(0)

        # Denormalizacja z [-1, 1] na [0, 1]
        output_tensor = (output_tensor + 1) / 2
        output_tensor = output_tensor.clamp(0, 1)

        # Konwersja na numpy i PIL
        output_tensor = output_tensor.cpu().detach()
        output_array = (output_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        output_image = Image.fromarray(output_array)

        return output_image

    def predict(self, image_path: str) -> Image.Image:
        """
        Generuje satelitę z mapy.

        Args:
            image_path: Ścieżka do mapy

        Returns:
            PIL Image ze wygenerowanym satelitą
        """
        with torch.no_grad():
            # Preprocessowanie
            input_tensor = self._preprocess(image_path)

            # Inference
            output_tensor = self.generator(input_tensor)

            # Postprocessowanie
            output_image = self._postprocess(output_tensor)

        return output_image

    def predict_batch(self, image_paths: List[str], output_dir: str) -> None:
        """
        Generuje satelity dla wielu map.

        Args:
            image_paths: Lista ścieżek do map
            output_dir: Folder na wyjście
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Generowanie {len(image_paths)} satelit...")

        for img_path in tqdm(image_paths):
            try:
                output_image = self.predict(img_path)

                # Nazwa wyjścia
                img_name = Path(img_path).stem
                output_file = output_path / f"{img_name}_generated.png"

                output_image.save(output_file)
                logger.debug(f"  ✓ {output_file}")

            except Exception as e:
                logger.error(f"  ❌ Błąd przy {img_path}: {e}")

        logger.info(f"✓ Generowanie zakończone! Wyniki w: {output_dir}")

    def generate_comparison(self, image_path: str, output_path: str) -> None:
        """
        Generuje porównanie: input | output.

        Args:
            image_path: Ścieżka do mapy
            output_path: Ścieżka do zapisu porównania
        """
        from PIL import ImageDraw, ImageFont

        # Wczytaj mapę
        map_img = Image.open(image_path).convert("RGB")
        map_img = map_img.resize((256, 256), Image.BILINEAR)

        # Generuj satelitę
        sat_img = self.predict(image_path)

        # Stwórz porównanie
        comparison = Image.new("RGB", (512, 256))
        comparison.paste(map_img, (0, 0))
        comparison.paste(sat_img, (256, 0))

        # Dodaj etykiety
        draw = ImageDraw.Draw(comparison)
        try:
            draw.text((10, 10), "MAP", fill=(255, 255, 255))
            draw.text((266, 10), "GENERATED SATELLITE", fill=(255, 255, 255))
        except Exception:
            pass  # Brak fontu - brak etykiet

        comparison.save(output_path)
        logger.info(f"✓ Porównanie zapisane: {output_path}")


def main():
    """
    Przykład użycia inferencji.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pix2Pix Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input image or folder"
    )
    parser.add_argument("--output", type=str, default="./output", help="Output folder")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    args = parser.parse_args()

    # Inicjalizacja
    inference = Pix2PixInference(args.checkpoint, device=args.device)

    input_path = Path(args.input)

    if input_path.is_file():
        # Pojedynczy obraz
        logger.info(f"📷 Generowanie dla: {input_path}")
        output_img = inference.predict(str(input_path))

        output_file = Path(args.output) / f"{input_path.stem}_generated.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_file)

        logger.info(f"✓ Wynik: {output_file}")

    elif input_path.is_dir():
        # Wiele obrazów
        image_files = [
            f
            for f in input_path.iterdir()
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]

        inference.predict_batch([str(f) for f in image_files], args.output)

    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()
    """
    Użycie:
    
    # Pojedynczy obraz
    python inference.py --checkpoint checkpoints/best.pth --input map.jpg --output output/
    
    # Folder
    python inference.py --checkpoint checkpoints/best.pth --input ./test_maps/ --output ./output/
    """
