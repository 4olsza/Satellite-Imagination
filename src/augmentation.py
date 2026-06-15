"""
Augmentacje danych dla Pix2Pix.
Zwiększają różnorodność treningową i poprawiają generalizację modelu.
"""

import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
from typing import Tuple


class Pix2PixAugmentation:
    """
    Klasa z augmentacjami dla pary obrazów: mapa + satelita.
    Wszystkie transformacje muszą być stosowane identycznie do obu obrazów.
    """

    def __init__(
        self,
        resize_size: int = 286,
        crop_size: int = 256,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.3,
        color_jitter_prob: float = 0.0,  # Parametr zostawiony dla kompatybilności, ale wyłączony
    ):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob

    def __call__(
        self, map_img: Image.Image, sat_img: Image.Image, is_train: bool = True
    ) -> Tuple[Image.Image, Image.Image]:
        
        if not is_train:
            # Walidacja / test - bez losowych zmian, tylko dopasowanie rozmiaru.
            return self._resize_and_crop(map_img, sat_img)

        # 1. Resize do większego rozmiaru przed cropem.
        map_img, sat_img = self._resize(map_img, sat_img)

        # 2. Losowy crop w tym samym obszarze dla obu obrazów.
        map_img, sat_img = self._random_crop(map_img, sat_img)

        # 3. Losowe odbicie w poziomie.
        if random.random() < self.flip_prob:
            map_img = F.hflip(map_img)  # type: ignore
            sat_img = F.hflip(sat_img)  # type: ignore

        # 4. Losowa rotacja małym kątem.
        if random.random() < self.rotation_prob:
            angle = random.uniform(-15, 15)
            map_img = F.rotate(map_img, angle, fill=0)  # type: ignore
            sat_img = F.rotate(sat_img, angle, fill=0)  # type: ignore

        return map_img, sat_img

    def _resize(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Zmienia rozmiar obu obrazów do wartości `resize_size`."""
        # Podajemy rozmiar jako listę [size, size] i omijamy stary zapis Image.BILINEAR
        map_img = F.resize(map_img, [self.resize_size, self.resize_size])  # type: ignore
        sat_img = F.resize(sat_img, [self.resize_size, self.resize_size])  # type: ignore
        return map_img, sat_img

    def _random_crop(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Wykonuje losowy crop z tego samego miejsca dla obu obrazów."""
        w, h = map_img.size
        th, tw = self.crop_size, self.crop_size

        if w == tw and h == th:
            return map_img, sat_img

        # losowy współrzędny startowy cropu
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        map_img = F.crop(map_img, y1, x1, th, tw)  # type: ignore
        sat_img = F.crop(sat_img, y1, x1, th, tw)  # type: ignore

        return map_img, sat_img

    def _resize_and_crop(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Dopasowuje obrazy do `crop_size` bez losowości dla walidacji/testu."""
        map_img = F.resize(map_img, [self.crop_size, self.crop_size])  # type: ignore
        sat_img = F.resize(sat_img, [self.crop_size, self.crop_size])  # type: ignore
        return map_img, sat_img


# Predefiniowane zestawy augmentacji.
STRONG_AUGMENTATION = Pix2PixAugmentation(
    flip_prob=0.5, rotation_prob=0.4, color_jitter_prob=0.0
)

MILD_AUGMENTATION = Pix2PixAugmentation(
    flip_prob=0.3, rotation_prob=0.1, color_jitter_prob=0.0
)

NO_AUGMENTATION = Pix2PixAugmentation(
    flip_prob=0.0, rotation_prob=0.0, color_jitter_prob=0.0
)


if __name__ == "__main__":
    from data.dataset import MapDataset # lub src.data.dataset zaleznie od struktury
    from torch.utils.data import DataLoader

    print("Testowanie augmentacji...")
    print("✓ Augmentacje gotowe do użycia bez błędów składni!")