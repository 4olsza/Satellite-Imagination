"""
Metryki do ewaluacji generatora
PSNR, SSIM, perceptual distance, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ImageMetrics:
    """
    Klasa z metrykami dla obrazów.
    Porównuje rzeczywiste vs wygenerowane obrazy.
    """

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalizuje tensor z [-1, 1] na [0, 1].

        Args:
            tensor: Tensor w formacie [-1, 1]

        Returns:
            Tensor w formacie [0, 1]
        """
        return (tensor + 1) / 2

    @staticmethod
    def psnr(
        real: torch.Tensor, generated: torch.Tensor, max_val: float = 1.0
    ) -> float:
        """
        Peak Signal-to-Noise Ratio.
        Wyżej = lepiej (zazwyczaj > 20 to dobrze).

        Args:
            real: Rzeczywisty obraz
            generated: Wygenerowany obraz
            max_val: Maksimalna wartość piksela (1.0 dla [0,1])

        Returns:
            PSNR value
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        mse = torch.mean((real - generated) ** 2)
        if mse == 0:
            return 100.0

        psnr_val = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
        return psnr_val.item()

    @staticmethod
    def ssim(real: torch.Tensor, generated: torch.Tensor, window_size: int = 11) -> float:
        """
        Structural Similarity Index.
        Mierzy podobieństwo strukturalne (zazwyczaj 0-1, wyżej = lepiej).

        Args:
            real: Rzeczywisty obraz
            generated: Wygenerowany obraz
            window_size: Rozmiar okna Gaussa

        Returns:
            SSIM value (0-1)
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        # Stałe dla stabilności
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        # Średnia
        mean_real = F.avg_pool2d(real, window_size, stride=1)
        mean_gen = F.avg_pool2d(generated, window_size, stride=1)

        # Wariancja
        sq_real = F.avg_pool2d(real ** 2, window_size, stride=1)
        sq_gen = F.avg_pool2d(generated ** 2, window_size, stride=1)
        sigma_real_sq = sq_real - mean_real ** 2
        sigma_gen_sq = sq_gen - mean_gen ** 2
        sigma_real_gen = F.avg_pool2d(real * generated, window_size, stride=1) - (
            mean_real * mean_gen
        )

        # SSIM
        numerator = (2 * mean_real * mean_gen + C1) * (2 * sigma_real_gen + C2)
        denominator = (
            (mean_real ** 2 + mean_gen ** 2 + C1)
            * (sigma_real_sq + sigma_gen_sq + C2)
        )

        ssim_val = numerator / denominator
        return ssim_val.mean().item()

    @staticmethod
    def mae(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Mean Absolute Error - pixel-wise L1 distance.
        Niżej = lepiej.

        Args:
            real: Rzeczywisty obraz
            generated: Wygenerowany obraz

        Returns:
            MAE value
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        mae_val = torch.mean(torch.abs(real - generated))
        return mae_val.item()

    @staticmethod
    def mse(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Mean Squared Error.
        Niżej = lepiej.

        Args:
            real: Rzeczywisty obraz
            generated: Wygenerowany obraz

        Returns:
            MSE value
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        mse_val = torch.mean((real - generated) ** 2)
        return mse_val.item()

    @staticmethod
    def lpips(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Learned Perceptual Image Patch Similarity.
        Wyżej = bardziej różne (zazwyczaj 0-1).
        
        UWAGA: Ta implementacja jest uproszczona!
        Dla pełnej wersji użyj: pip install lpips

        Args:
            real: Rzeczywisty obraz
            generated: Wygenerowany obraz

        Returns:
            LPIPS value (simplified)
        """
        # Uproszczona wersja - porównanie gradientów
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        # Sobel edges
        kernel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Dla każdego kanału
        diff = 0
        for c in range(real.shape[1]):
            channel_real = real[:, c : c + 1, :, :]
            channel_gen = generated[:, c : c + 1, :, :]

            edge_real_x = F.conv2d(channel_real, kernel_x, padding=1)
            edge_gen_x = F.conv2d(channel_gen, kernel_x, padding=1)

            edge_real_y = F.conv2d(channel_real, kernel_y, padding=1)
            edge_gen_y = F.conv2d(channel_gen, kernel_y, padding=1)

            diff += torch.mean(
                torch.abs(edge_real_x - edge_gen_x) + torch.abs(edge_real_y - edge_gen_y)
            )

        return (diff / real.shape[1]).item()

    @staticmethod
    def compute_all_metrics(
        real: torch.Tensor, generated: torch.Tensor
    ) -> dict:
        """
        Oblicza wszystkie metryki naraz.

        Args:
            real: Rzeczywisty obraz
            generated: Wygenerowany obraz

        Returns:
            Słownik z wszystkimi metrykami
        """
        return {
            "psnr": ImageMetrics.psnr(real, generated),
            "ssim": ImageMetrics.ssim(real, generated),
            "mae": ImageMetrics.mae(real, generated),
            "mse": ImageMetrics.mse(real, generated),
        }


class MetricsTracker:
    """
    Śledzi metryki podczas walidacji/testowania.
    """

    def __init__(self):
        self.history = {
            "psnr": [],
            "ssim": [],
            "mae": [],
            "mse": [],
        }

    def add(self, real: torch.Tensor, generated: torch.Tensor) -> None:
        """
        Dodaje nową parę obrazów i oblicza metryki.

        Args:
            real: Rzeczywisty obraz
            generated: Wygenerowany obraz
        """
        metrics = ImageMetrics.compute_all_metrics(real, generated)
        for key, value in metrics.items():
            self.history[key].append(value)

    def get_averages(self) -> dict:
        """
        Zwraca średnie wartości metryk.

        Returns:
            Słownik ze średnimi
        """
        averages = {}
        for key, values in self.history.items():
            if values:
                averages[key] = np.mean(values)
            else:
                averages[key] = 0.0

        return averages

    def print_summary(self) -> None:
        """Wypisuje podsumowanie metryk."""
        averages = self.get_averages()

        print("\n" + "=" * 50)
        print("📊 METRYKI EWALUACJI")
        print("=" * 50)
        print(f"  PSNR (wyżej=lepiej):  {averages['psnr']:.4f}")
        print(f"  SSIM (wyżej=lepiej):  {averages['ssim']:.4f}")
        print(f"  MAE  (niżej=lepiej):  {averages['mae']:.4f}")
        print(f"  MSE  (niżej=lepiej):  {averages['mse']:.4f}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test metryk
    print("Testing metrics...")

    real = torch.randn((1, 3, 256, 256))
    generated = torch.randn((1, 3, 256, 256))

    metrics = ImageMetrics.compute_all_metrics(real, generated)
    print(f"Metrics: {metrics}")

    print("✓ Metrics working!")
