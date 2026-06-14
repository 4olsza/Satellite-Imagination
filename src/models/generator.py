import torch
import torch.nn as nn
from typing import List


# ==========================================
# 1. Klasa pomocnicza Block ("Klocek LEGO")
# ==========================================
class Block(nn.Module):
    """
    Uniwersalny blok dla sieci U-Net.
    Może działać jako warstwa zstępująca (encoder) lub wstępująca (decoder).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down: bool = True,
        act: str = "relu",
        use_dropout: bool = False,
    ):
        super().__init__()
        # Jeśli down=True, używamy zwykłej Conv2d do zmniejszenia wymiaru.
        # Jeśli down=False, wracamy do klasycznego ConvTranspose2d do powiększania obrazu.
        self.conv = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                padding_mode="reflect",
            )
            if down
            else nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.dropout(x) if self.use_dropout else x


# ==========================================
# Główna sieć Generatora (U-Net)
# ==========================================
class Generator(nn.Module):
    """
    Generator dla Pix2Pix w architekturze U-Net.
    Mapuje obraz wejściowy (mapę) na obraz wyjściowy (satelitę).
    """

    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()

        # ------------------------------------------
        # 2. Koder (Encoder) - schodzimy w dół
        # ------------------------------------------
        # Pierwsza warstwa nie ma BatchNorm, bo to poprawia stabilność przy wejściu.
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )  # Wynik: 128x128

        self.down1 = Block(features, features * 2, down=True, act="leaky")  # -> 64x64
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky")  # -> 32x32
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky")  # -> 16x16
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky")  # -> 8x8
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky")  # -> 4x4
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky")  # -> 2x2

        # Wąskie gardło (bottleneck) na samym dole U-Netu
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )  # Wynik: 1x1

        # ------------------------------------------
        # 3. Dekoder (Decoder) - idziemy w górę
        # ------------------------------------------
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)  # -> 2x2
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  # -> 4x4
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)  # -> 8x8
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu")  # -> 16x16
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu")  # -> 32x32
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu")  # -> 64x64
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu")  # -> 128x128

        # Ostatnia warstwa zamienia kanały na RGB i przywraca oryginalny rozmiar 256x256.
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Tanh normalizuje piksele do [-1, 1]
        )

    # ------------------------------------------
    # 4. Skip Connections (metoda forward)
    # ------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass przez U-Net z skip connections.
        Argumenty:
            x: Tensor wejściowy [Batch, in_channels, H, W]
        Zwraca:
            Tensor wyjściowy [Batch, 3, H, W]
        """
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))

        # Doklejamy d1 do ostatniego dekonwolucyjnego bloku i finalnie generujemy obraz.
        return self.final_up(torch.cat([up7, d1], dim=1))


# Szybki test, czy sieć działa i zwraca odpowiedni wymiar
if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    preds = model(x)
    print(f"Kształt wejścia: {x.shape}")
    print(f"Kształt wyjścia: {preds.shape}")  # Powinno wypisać [1, 3, 256, 256]