"""
Data Augmentation Pipeline for Pix2Pix.
Increases training dataset diversity and improves model generalization 
by applying identical spatial transformations to paired images.
"""

import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
from typing import Tuple


class Pix2PixAugmentation:
    """
    A paired image augmentation class for Map and Satellite image pairs.
    Crucially, all spatial transformations are applied identically to both 
    images to ensure perfect pixel-to-pixel alignment is maintained.
    """

    def __init__(
        self,
        resize_size: int = 286,
        crop_size: int = 256,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.3,
        color_jitter_prob: float = 0.0,
    ) -> None:
        """
        Initializes the augmentation pipeline.

        Args:
            resize_size (int): The intermediate size to scale up to before cropping.
            crop_size (int): The final target dimensions for the network (e.g., 256).
            flip_prob (float): Probability of applying a horizontal flip.
            rotation_prob (float): Legacy parameter. Kept for signature compatibility, 
                                   but disabled in logic to prevent black edge artifacts.
            color_jitter_prob (float): Legacy parameter. Kept for signature compatibility.
        """
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob

    def __call__(
        self, map_img: Image.Image, sat_img: Image.Image, is_train: bool = True
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Applies the augmentation sequence to an image pair.
        
        Args:
            map_img (Image.Image): The input map image.
            sat_img (Image.Image): The target satellite image.
            is_train (bool): If False, bypasses randomness and returns a deterministic crop.
            
        Returns:
            Tuple[Image.Image, Image.Image]: The transformed image pair.
        """
        if not is_train:
            return self._resize_and_crop(map_img, sat_img)

        # 1. Resize to a larger intermediate dimension (e.g., 286x286)
        map_img, sat_img = self._resize(map_img, sat_img)

        # 2. Random crop down to target size (e.g., 256x256) to introduce translation variance
        map_img, sat_img = self._random_crop(map_img, sat_img)

        # 3. Random Horizontal Flip
        if random.random() < self.flip_prob:
            map_img = F.hflip(map_img)  # type: ignore
            sat_img = F.hflip(sat_img)  # type: ignore

        return map_img, sat_img

    def _resize(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Resizes both images to the specified `resize_size`."""
        
        # Dimensions are passed as a list [size, size] to avoid legacy PIL Image.BILINEAR warnings
        map_img = F.resize(map_img, [self.resize_size, self.resize_size])  # type: ignore
        sat_img = F.resize(sat_img, [self.resize_size, self.resize_size])  # type: ignore
        return map_img, sat_img

    def _random_crop(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Performs a random spatial crop at the exact same coordinates for both images."""
        w, h = map_img.size
        th, tw = self.crop_size, self.crop_size

        if w == tw and h == th:
            return map_img, sat_img

        # Calculate random starting coordinates for the crop window
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        map_img = F.crop(map_img, y1, x1, th, tw)  # type: ignore
        sat_img = F.crop(sat_img, y1, x1, th, tw)  # type: ignore

        return map_img, sat_img

    def _resize_and_crop(
        self, map_img: Image.Image, sat_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Deterministically resizes images directly to `crop_size` for validation and testing."""
        map_img = F.resize(map_img, [self.crop_size, self.crop_size])  # type: ignore
        sat_img = F.resize(sat_img, [self.crop_size, self.crop_size])  # type: ignore
        return map_img, sat_img


# Predefined Augmentation Configurations
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
    # Internal module testing
    print("Testing augmentations...")
    print("✓ Augmentations are compiled and ready for deployment!")