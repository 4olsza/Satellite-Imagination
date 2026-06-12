import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
import logging
from typing import Tuple, Optional, Callable

# Konfiguracja logowania
logger = logging.getLogger(__name__)


class MapDataset(Dataset):
    """
    Dataset dla Pix2Pix - mapy → satelity.
    Zakłada, że każdy obraz zawiera mapę po lewej i satelitę po prawej stronie.
    Obsługuje augmentacje, walidację i obsługę błędów.
    """

    def __init__(
        self,
        root_dir: str,
        valid_extensions: Tuple[str, ...] = (".jpg", ".png", ".jpeg"),
        augmentation: Optional[Callable] = None,
        is_train: bool = True,
    ):
        """
Argumenty:
            root_dir: Ścieżka do folderu z danymi.
            valid_extensions: Dozwolone rozszerzenia plików.
            augmentation: Funkcja augmentacji par obrazów.
            is_train: Czy to zestaw treningowy.
        """
        self.root_dir = Path(root_dir)
        self.augmentation = augmentation
        self.is_train = is_train

        # Wczytanie listy plików, pomijając ukryte pliki
        self.list_files = [
            f
            for f in os.listdir(self.root_dir)
            if not f.startswith(".") and f.lower().endswith(valid_extensions)
        ]

        if not self.list_files:
            raise ValueError(
                f"Brak obrazów w folderze {root_dir} z rozszerzeniami {valid_extensions}"
            )

        logger.info(
            f"✓ Załadowano {len(self.list_files)} obrazów z {root_dir} "
            f"(augmentacja: {'ON' if augmentation else 'OFF'})"
        )

        # Transforms - normalizacja
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        # Zwraca liczbę obrazów w datasetcie
        return len(self.list_files)

    def __getitem__(self, index):
        # Pobranie nazwy pliku po indeksie
        img_file = self.list_files[index]

        # Pełna ścieżka do pliku
        img_path = os.path.join(self.root_dir, img_file)

        try:
            # Wczytanie obrazu jako RGB
            image = Image.open(img_path).convert("RGB")

            # Walidacja wymiarów: szerokość musi być parzysta, a wysokość co najmniej 256
            width, height = image.size
            if width % 2 != 0 or height < 256:
                logger.warning(
                    f"Obraz {img_file} ma niestandartowe wymiary: {width}x{height}"
                )
                width = (width // 2) * 2
                height = max(height, 256)
                image = image.resize((width, height))

            # Obrazy w datasetcie są połączone poziomo, więc dzielimy je na dwie części
            width, height = image.size

            # Wycinamy lewą i prawą połowę
            satellite_img = image.crop((0, 0, width // 2, height))
            map_img = image.crop((width // 2, 0, width, height))

            # Konwertujemy obrazy do tensorów i normalizujemy
            satellite_tensor = self.transform(satellite_img)
            map_tensor = self.transform(map_img)

            # Zwracamy najpierw mapę, potem satelitę
            return map_tensor, satellite_tensor

        except Exception as e:
            logger.error(f"Błąd przy wczytywaniu {img_file}: {str(e)}")
            # Zwracamy czarny obraz zamiast rzucać wyjątek
            black_image = Image.new("RGB", (256, 256), color=(0, 0, 0))
            black_tensor = self.transform(black_image)
            return black_tensor, black_tensor
