"""
Preprocessing dla danych satelitarnych
- Resize/normalizacja
- Train/Val/Test split
- Data validation
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor dla danych Pix2Pix.
    Walidacja, resize, i organizacja danych.
    """

    def __init__(self, min_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            min_size: Minimalny rozmiar obrazu (width, height)
        """
        self.min_size = min_size
        self.stats = {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "errors": [],
        }

    def validate_image(self, img_path: str) -> bool:
        """
        Waliduje pojedynczy obraz.

        Args:
            img_path: Ścieżka do obrazu

        Returns:
            True jeśli obraz jest ok
        """
        try:
            img = Image.open(img_path).convert("RGB")
            width, height = img.size

            # Musi mieć co najmniej min_size
            if width < self.min_size[0] or height < self.min_size[1]:
                self.stats["errors"].append(f"{img_path}: Zbyt mały ({width}x{height})")
                return False

            # Musi być podzielny na 2 (mapa i satelita)
            if width % 2 != 0:
                self.stats["errors"].append(f"{img_path}: Nieparzysta szerokość")
                return False

            self.stats["valid_images"] += 1
            return True

        except Exception as e:
            self.stats["errors"].append(f"{img_path}: {str(e)}")
            return False

    def validate_dataset(self, data_dir: str) -> dict:
        """
        Waliduje cały dataset.

        Args:
            data_dir: Folder z obrazami

        Returns:
            Słownik ze statystyką
        """
        data_path = Path(data_dir)
        image_files = [
            f
            for f in data_path.iterdir()
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]

        print(f"📊 Walidacja {len(image_files)} obrazów...")
        for img_file in tqdm(image_files):
            self.stats["total_images"] += 1
            self.validate_image(str(img_file))

        self.stats["invalid_images"] = (
            self.stats["total_images"] - self.stats["valid_images"]
        )

        print(f"\n✅ Walidacja zakończona:")
        print(f"  Ważnych: {self.stats['valid_images']}")
        print(f"  Błędnych: {self.stats['invalid_images']}")

        if self.stats["errors"]:
            print(f"\n❌ Błędy ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:10]:  # Pokaż max 10
                print(f"  - {error}")

        return self.stats

    def split_dataset(
        self,
        data_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> None:
        """
        Dzieli dataset na train/val/test.

        Args:
            data_dir: Folder źródłowy
            output_dir: Folder wyjściowy
            train_ratio: Procent treningowy
            val_ratio: Procent walidacyjny
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)

        # Tworzenie folderów
        train_dir = output_path / "train"
        val_dir = output_path / "val"
        test_dir = output_path / "test"

        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Pobierz listę obrazów
        image_files = sorted(
            [f for f in data_path.iterdir() if f.suffix.lower() in [".jpg", ".png"]]
        )

        # Oblicz split
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Podziel
        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]

        # Kopiuj pliki
        print(f"📁 Splitting dataset...")
        for files, dest_dir, name in [
            (train_files, train_dir, "train"),
            (val_files, val_dir, "val"),
            (test_files, test_dir, "test"),
        ]:
            print(f"\n  {name.upper()}: {len(files)} obrazów")
            for src_file in tqdm(files, desc=f"  Kopiowanie do {name}"):
                dst_file = dest_dir / src_file.name
                if src_file != dst_file:  # Jeśli to nie ten sam folder
                    Image.open(src_file).save(dst_file)

        print(f"\n✅ Split zakończony!")

    def resize_images(
        self, data_dir: str, output_dir: str, target_size: Tuple[int, int] = (512, 256)
    ) -> None:
        """
        Resizuje wszystkie obrazy do docelowego rozmiaru.

        Args:
            data_dir: Folder źródłowy
            output_dir: Folder wyjściowy
            target_size: Docelowy rozmiar (width, height)
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = [
            f for f in data_path.iterdir() if f.suffix.lower() in [".jpg", ".png"]
        ]

        print(f"🖼️  Resizing {len(image_files)} obrazów do {target_size}...")

        for img_file in tqdm(image_files):
            try:
                img = Image.open(img_file).convert("RGB")
                img_resized = img.resize(target_size, Image.BILINEAR)
                output_file = output_path / img_file.name
                img_resized.save(output_file, quality=95)
            except Exception as e:
                print(f"❌ Błąd przy {img_file}: {e}")

        print("✅ Resize zakończony!")

    def get_dataset_stats(self, data_dir: str) -> dict:
        """
        Oblicza statystykę datasetu.

        Args:
            data_dir: Folder z danymi

        Returns:
            Słownik ze statystyką
        """
        data_path = Path(data_dir)
        image_files = [
            f for f in data_path.iterdir() if f.suffix.lower() in [".jpg", ".png"]
        ]

        sizes = []
        channels = []

        for img_file in tqdm(image_files, desc="Obliczanie statystyki"):
            try:
                img = Image.open(img_file).convert("RGB")
                sizes.append(img.size)
                channels.append(len(img.getbands()))
            except Exception:
                pass

        if sizes:
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]

            stats = {
                "total_images": len(image_files),
                "avg_width": np.mean(widths),
                "avg_height": np.mean(heights),
                "min_width": np.min(widths),
                "max_width": np.max(widths),
                "min_height": np.min(heights),
                "max_height": np.max(heights),
                "avg_channels": np.mean(channels),
            }

            print("\n📊 Statystyka datasetu:")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f}")

            return stats

        return {}


if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # Przykład użycia
    print("Validation example:")
    preprocessor.validate_dataset("data/train")

    print("\nDataset stats:")
    preprocessor.get_dataset_stats("data/train")
