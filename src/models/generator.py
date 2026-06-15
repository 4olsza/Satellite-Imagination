import torch
import torch.nn as nn
from typing import List

# ==========================================
# 1. Klasa pomocnicza Block ("Klocek LEGO")
# ==========================================
class Block(nn.Module):
    """
    Uniwersalny blok dla sieci U-Net.
    Używa Upsample + Conv2d zamiast ConvTranspose2d, aby wyeliminować szachownicę.
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
        
        # Jeśli down=True, zmniejszamy wymiar (Encoder).
        # Jeśli down=False, używamy Upsample + Conv2d, by uniknąć checkerboard artifacts (Decoder).
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
            else nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    padding_mode="reflect",
                )
            )
        )

        self.bn = nn.InstanceNorm2d(out_channels)
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
    def __init__(self, in_channels: int = 3, features: int = 64):
        super().__init__()

        # Koder (Encoder)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features * 2, down=True, act="leaky")
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky")
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky")
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky")
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky")

        # Wąskie gardło
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        # Dekoder (Decoder)
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu")
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu")
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu")
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu")

        # Ostatnia warstwa również musi zostać zamieniona na Upsample
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(features * 2, 3, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return self.final_up(torch.cat([up7, d1], dim=1))