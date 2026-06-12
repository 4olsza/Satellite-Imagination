"""
Skrypt inferencyjny Pix2Pix.
Wczytuje wytrenowany generator i generuje obrazy satelitarne z map.
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

sys.path.insert(0, str(Path(__file__).parent))
from src.models.generator import Generator


class Pix2PixInference:
    """
    Klasa do inferencji modelu Pix2Pix.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        features: int = 64,
    ):
        """
        Inicjalizuje generator oraz wczytuje checkpoint.
Argumenty:
            checkpoint_path: Ścieżka do wytrenowanego modelu.
            device: Urządzenie do inferencji ('cuda' lub 'cpu').
            features: Bazowa liczba filtrów w generatorze.
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Tworzymy generator i przenosimy go na urządzenie
        self.generator = Generator(in_channels=3, features=features).to(self.device)

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint nie istnieje: {checkpoint_path}")

        # Wczytujemy wagi modelu
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            self.generator.load_state_dict(checkpoint)

        self.generator.eval()
        logger.info(f"✓ Model wczytany: {checkpoint_path}")

        # Normalizacja wejścia do przedziału [-1, 1]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """
        Przygotowuje obraz wejściowy do inferencji.
Argumenty:
            image_path: Ścieżka do pliku obrazu.
Zwraca:
            Tensor w kształcie [1, 3, 256, 256].
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256), Image.BILINEAR)
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device)

    def _postprocess(self, output_tensor: torch.Tensor) -> Image.Image:
        """
        Konwertuje wyjście modelu na obraz PIL.
Argumenty:
            output_tensor: Tensor [1, 3, 256, 256].
Zwraca:
            Obraz PIL w formacie RGB.
        """
        output_tensor = output_tensor.squeeze(0)
        output_tensor = (output_tensor + 1) / 2
        output_tensor = output_tensor.clamp(0, 1)

        output_tensor = output_tensor.cpu().detach()
        output_array = (output_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        output_image = Image.fromarray(output_array)
        return output_image

    def predict(self, image_path: str) -> Image.Image:
        """
        Generuje satelitę dla pojedynczego obrazu mapy.
Argumenty:
            image_path: Ścieżka do mapy.
Zwraca:
            Obraz PIL z wygenerowaną satelitą.
        """
        with torch.no_grad():
            input_tensor = self._preprocess(image_path)
            output_tensor = self.generator(input_tensor)
            output_image = self._postprocess(output_tensor)

        return output_image

    def predict_batch(self, image_paths: List[str], output_dir: str) -> None:
        """
        Generuje satelity dla wielu map i zapisuje wyniki.
Argumenty:
            image_paths: Lista ścieżek do map.
            output_dir: Folder wyjściowy.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"🚀 Generowanie {len(image_paths)} obrazów...")
        for img_path in tqdm(image_paths):
            try:
                output_image = self.predict(img_path)
                img_name = Path(img_path).stem
                output_file = output_path / f"{img_name}_generated.png"
                output_image.save(output_file)
                logger.debug(f"  ✓ {output_file}")
            except Exception as e:
                logger.error(f"  ❌ Błąd przy {img_path}: {e}")

        logger.info(f"✓ Generowanie zakończone! Wyniki w: {output_dir}")

    def generate_comparison(self, image_path: str, output_path: str) -> None:
        """
        Tworzy obraz porównawczy: oryginalna mapa obok wygenerowanej satelity.
Argumenty:
            image_path: Ścieżka do mapy.
            output_path: Ścieżka zapisu pliku.
        """
        from PIL import ImageDraw

        map_img = Image.open(image_path).convert("RGB")
        map_img = map_img.resize((256, 256), Image.BILINEAR)
        sat_img = self.predict(image_path)

        comparison = Image.new("RGB", (512, 256))
        comparison.paste(map_img, (0, 0))
        comparison.paste(sat_img, (256, 0))

        draw = ImageDraw.Draw(comparison)
        try:
            draw.text((10, 10), "MAP", fill=(255, 255, 255))
            draw.text((266, 10), "GENERATED SATELLITE", fill=(255, 255, 255))
        except Exception:
            pass

        comparison.save(output_path)
        logger.info(f"✓ Porównanie zapisane: {output_path}")


def main():
    """
    Przykład uruchomienia inferencji z linii poleceń.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pix2Pix inferencja")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Ścieżka do checkpointu"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Plik lub folder wejściowy"
    )
    parser.add_argument(
        "--output", type=str, default="./output", help="Folder wyjściowy"
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda lub cpu")

    args = parser.parse_args()
    inference = Pix2PixInference(args.checkpoint, device=args.device)
    input_path = Path(args.input)

    if input_path.is_file():
        logger.info(f"📷 Generowanie dla: {input_path}")
        output_img = inference.predict(str(input_path))
        output_file = Path(args.output) / f"{input_path.stem}_generated.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_img.save(output_file)
        logger.info(f"✓ Wynik zapisano: {output_file}")
    elif input_path.is_dir():
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
        inference.predict_batch([str(f) for f in image_files], args.output)
    else:
        raise FileNotFoundError(f"Nie znaleziono ścieżki wejściowej: {args.input}")


if __name__ == "__main__":
    main()
