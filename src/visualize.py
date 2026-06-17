"""
Dataset Visualization Utility.
Provides a quick sanity check to visualize the input map and target satellite image pairs
before starting the heavy training process.
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Ensure the source directory is in the system path for internal imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.data.dataset import MapDataset


def show_images(map_img: torch.Tensor, sat_img: torch.Tensor) -> None:
    """
    Converts normalized image tensors to viewable arrays and plots them side-by-side.

    Args:
        map_img (torch.Tensor): The input map tensor from the dataset.
        sat_img (torch.Tensor): The target satellite image tensor.
    """
    # PyTorch tensors are natively formatted as [Channels, Height, Width].
    # Matplotlib requires images to be formatted as [Height, Width, Channels].
    # Furthermore, we denormalize the pixel values from the [-1, 1] range back to [0, 1].
    map_img_display = (map_img.squeeze().permute(1, 2, 0) * 0.5) + 0.5
    sat_img_display = (sat_img.squeeze().permute(1, 2, 0) * 0.5) + 0.5

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(map_img_display.numpy())
    ax[0].set_title("Input Sketch (Map)")
    ax[0].axis("off")

    ax[1].imshow(sat_img_display.numpy())
    ax[1].set_title("Target Image (Satellite Ground Truth)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Loading dataset for visualization...")

    # Initialize the dataset. 
    # Note: Ensure the path matches your actual folder structure (e.g., 'data/maps/train')
    dataset = MapDataset(root_dir="data/train")

    # Use a DataLoader to fetch a single random image pair (batch_size=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    map_tensor, sat_tensor = next(iter(dataloader))

    print("Generating visual preview...")
    show_images(map_tensor, sat_tensor)