"""
Augmentacje danych dla Pix2Pix
Zwiększają różnorodność treningową i poprawiają generalizację
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import random
from typing import Tuple


class Pix2PixAugmentation:
    """
    Klasa z augmentacjami dla Pix2Pix.
    Ważne: augmentacje muszą być zastosowane JEDNAKOWO do mapy i satelity!
    """

    def __init__(
        self,
        resize_size: int = 286,
        crop_size: int = 256,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.3,
        color_jitter_prob: float = 0.3,
    ):
        """
        Args:
            resize_size: Rozmiar do resize'u (będziemy cropować z tego)
            crop_size: Final size (256x256)
            flip_prob: Prawdopodobieństwo horizontal flip
            rotation_prob: Prawdopodobieństwo rotacji
            color_jitter_prob: Prawdopodobieństwo color jitter
        """
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.color_jitter_prob = color_jitter_prob

    def __call__(
        self, map_img: Image.Image, sat_img: Image.Image, is_train: bool = True
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Aplikuje augmentacje do obu obrazów razem.

        Args:
            map_img: Obraz mapy
            sat_img: Obraz satelity
            is_train: Czy to training (True) czy val (False)

        Returns:
            Tuple: (augmented_map, augmented_sat)
        """
        if not is_train:
            # Walidacja - bez augmentacji, tylko resize i crop
            return self._resize_and_crop(map_img, sat_img)

        # Training augmentations
        # 1. Resize
        map_img, sat_img = self._resize(map_img, sat_img)

        # 2. Random crop
        map_img, sat_img = self._random_crop(map_img, sat_img)

        # 3. Horizontal flip
        if random.random() < self.flip_prob:
            map_img = F.hflip(map_img)
            sat_img = F.hflip(sat_img)

        # 4. Random rotation (mały kąt)
        if random.random() < self.rotation_prob:
            angle = random.uniform(-15, 15)  # -15 do 15 stopni
            map_img = F.rotate(map_img, angle, fill=0)
            sat_img = F.rotate(sat_img, angle, fill=0)

        # 5. Color jitter - TYLKO do mapy (satelita ma realistyczne kolory)
        if random.random() < self.color_jitter_prob:
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )
            map_img = color_jitter(map_img)

        return map_img, sat_img

    def _resize(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Resize do resize_size"""
        map_img = F.resize(map_img, self.resize_size, Image.BILINEAR)
        sat_img = F.resize(sat_img, self.resize_size, Image.BILINEAR)
        return map_img, sat_img

    def _random_crop(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Random crop z tego samego miejsca obu obrazów"""
        w, h = map_img.size
        th, tw = self.crop_size, self.crop_size

        if w == tw and h == th:
            return map_img, sat_img

        # Losowy punkt startu
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        map_img = F.crop(map_img, y1, x1, th, tw)
        sat_img = F.crop(sat_img, y1, x1, th, tw)

        return map_img, sat_img

    def _resize_and_crop(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Centrowe resize i crop - dla walidacji"""
        map_img = F.resize(map_img, self.crop_size, Image.BILINEAR)
        sat_img = F.resize(sat_img, self.crop_size, Image.BILINEAR)
        return map_img, sat_img


# Predefiniowane augmentacje
STRONG_AUGMENTATION = Pix2PixAugmentation(
    flip_prob=0.5, rotation_prob=0.4, color_jitter_prob=0.5
)

MILD_AUGMENTATION = Pix2PixAugmentation(
    flip_prob=0.3, rotation_prob=0.1, color_jitter_prob=0.1
)

NO_AUGMENTATION = Pix2PixAugmentation(
    flip_prob=0.0, rotation_prob=0.0, color_jitter_prob=0.0
)


if __name__ == "__main__":
    from dataset import MapDataset
    from torch.utils.data import DataLoader

    # Test augmentacji
    print("Testing augmentations...")

    dataset = MapDataset(root_dir="data/train")
    dataloader = DataLoader(dataset, batch_size=1)

    map_img, sat_img = next(iter(dataloader))
    print(f"Original shapes: map={map_img.shape}, sat={sat_img.shape}")

    aug = STRONG_AUGMENTATION
    # Konwertuj tensor na PIL dla augmentacji
    # (w praktyce augmentacje są w dataset.py)
    print("✓ Augmentations ready!")
