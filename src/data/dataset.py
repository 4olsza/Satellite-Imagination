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
        Args:
            root_dir: Ścieżka do folderu z danymi
            valid_extensions: Dozwolone rozszerzenia plików
            augmentation: Callable augmentation function (z augmentation.py)
            is_train: Czy to dataset treningowy (dla augmentacji)
        """
        self.root_dir = Path(root_dir)
        self.augmentation = augmentation
        self.is_train = is_train

        # Wczytanie listy plików z wylądowaniem plików ukrytych
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
        # returning number of the images
        return len(self.list_files)

    def __getitem__(self, index):
        # getting file's name
        img_file = self.list_files[index]

        # getting full path
        img_path = os.path.join(self.root_dir, img_file)

        try:
            # opening the image using PIL (RGB ensures us of having three colour channels)
            image = Image.open(img_path).convert("RGB")

            # Validacja rozmiaru - obraz powinien być parzysty żeby można było go podzielić
            width, height = image.size
            if width % 2 != 0 or height < 256:
                logger.warning(
                    f"Obraz {img_file} ma niestandartowe wymiary: {width}x{height}"
                )
                # Resize do standardu jeśli to konieczne
                width = (width // 2) * 2
                height = max(height, 256)
                image = image.resize((width, height))

            # data from dataset maps is horizontally concatenated so we need to split it excatly in a half
            # checking width and height of the image
            width, height = image.size

            # using crop function to cut the half of the image
            # satellite image - usually left side
            satellite_img = image.crop((0, 0, width // 2, height))
            # map sketch - usually right side
            map_img = image.crop((width // 2, 0, width, height))

            # putting both images through transforms
            satellite_tensor = self.transform(satellite_img)
            map_tensor = self.transform(map_img)

            # returning data in the tuple - input at first (sketch), then goal (satellite)
            return map_tensor, satellite_tensor

        except Exception as e:
            logger.error(f"Błąd przy wczytywaniu {img_file}: {str(e)}")
            # Zwracamy czarny obraz zamiast rzucać wyjątek
            black_image = Image.new("RGB", (256, 256), color=(0, 0, 0))
            black_tensor = self.transform(black_image)
            return black_tensor, black_tensor
