import sys

sys.path.append(".")
import torch
import matplotlib.pyplot as plt

from src.data.dataset import MapDataset
from torch.utils.data import DataLoader


def show_images(map_img, sat_img):
    """Konwertuje tensory na obrazy i wyświetla je w matplotlib."""
    # PyTorch: [Kanały, Wysokość, Szerokość]
    # Matplotlib: [Wysokość, Szerokość, Kanały]
    # Dodatkowo denormalizujemy z [-1, 1] do [0, 1].
    map_img = (map_img.squeeze().permute(1, 2, 0) * 0.5) + 0.5
    sat_img = (sat_img.squeeze().permute(1, 2, 0) * 0.5) + 0.5

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(map_img.numpy())
    ax[0].set_title("Szkic wejściowy (Mapa)")
    ax[0].axis("off")

    ax[1].imshow(sat_img.numpy())
    ax[1].set_title("Docelowy obraz (Satelita)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Wczytywanie datasetu...")

    dataset = MapDataset(root_dir="data/train")

    # Dataloader pobiera jedną parę obrazów do podglądu
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    map_img, sat_img = next(iter(dataloader))

    print("Generowanie podglądu...")
    show_images(map_img, sat_img)
