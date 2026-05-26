import torch
import matplotlib.pyplot as plt

# Zakładam, że Krzysiu nazwał swoją klasę datasetu "MapDataset"
# Jeśli nazwał ją inaczej, zmień tę nazwę poniżej!
from dataset import MapDataset
from torch.utils.data import DataLoader


def show_images(map_img, sat_img):
    """Funkcja pomocnicza do zamiany tensorów na obrazki matplotlib"""
    # PyTorch używa formatu [Kanały, Wysokość, Szerokość]
    # Matplotlib potrzebuje [Wysokość, Szerokość, Kanały], więc używamy .permute()
    # Dodatkowo normalizujemy wartości z [-1, 1] z powrotem do [0, 1] dla ładnego wyświetlania

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

    dataset = MapDataset(
        root_dir="../data/train"
    )  # Pamiętaj o ścieżce do folderu z obrazkami!

    # Bierzemy tylko jeden zestaw danych do testu
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Pobieramy pierwszą paczkę
    map_img, sat_img = next(iter(dataloader))

    print("Generowanie podglądu...")
    show_images(map_img, sat_img)
