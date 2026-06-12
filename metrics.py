"""
Metryki do ewaluacji generatora.
Zawiera PSNR, SSIM, MAE, MSE oraz uproszczoną wersję LPIPS.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImageMetrics:
    """
    Klasa z metrykami dla porównania obrazów.
    """

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalizuje tensor z zakresu [-1, 1] do [0, 1].
Argumenty:
            tensor: Tensor obrazu.
Zwraca:
            Tensor w zakresie [0, 1].
        """
        return (tensor + 1) / 2

    @staticmethod
    def psnr(
        real: torch.Tensor, generated: torch.Tensor, max_val: float = 1.0
    ) -> float:
        """
        Oblicza PSNR (Peak Signal-to-Noise Ratio).
        Wyższa wartość oznacza lepszą jakość.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        mse = torch.mean((real - generated) ** 2)
        if mse == 0:
            return 100.0

        psnr_val = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
        return psnr_val.item()

    @staticmethod
    def ssim(
        real: torch.Tensor, generated: torch.Tensor, window_size: int = 11
    ) -> float:
        """
        Oblicza SSIM (Structural Similarity Index).
        Wartość w przybliżeniu 0-1, wyżej = lepsze podobieństwo.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        mean_real = F.avg_pool2d(real, window_size, stride=1)
        mean_gen = F.avg_pool2d(generated, window_size, stride=1)

        sq_real = F.avg_pool2d(real**2, window_size, stride=1)
        sq_gen = F.avg_pool2d(generated**2, window_size, stride=1)
        sigma_real_sq = sq_real - mean_real**2
        sigma_gen_sq = sq_gen - mean_gen**2
        sigma_real_gen = F.avg_pool2d(real * generated, window_size, stride=1) - (
            mean_real * mean_gen
        )

        numerator = (2 * mean_real * mean_gen + C1) * (2 * sigma_real_gen + C2)
        denominator = (mean_real**2 + mean_gen**2 + C1) * (
            sigma_real_sq + sigma_gen_sq + C2
        )

        ssim_val = numerator / denominator
        return ssim_val.mean().item()

    @staticmethod
    def mae(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Oblicza MAE (Mean Absolute Error) między obrazami.
        Niższa wartość jest lepsza.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)
        mae_val = torch.mean(torch.abs(real - generated))
        return mae_val.item()

    @staticmethod
    def mse(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Oblicza MSE (Mean Squared Error) między obrazami.
        Niższa wartość jest lepsza.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)
        mse_val = torch.mean((real - generated) ** 2)
        return mse_val.item()

    @staticmethod
    def lpips(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Uproszczona wersja LPIPS (perceptual similarity).
        Ta implementacja wykorzystuje porównanie krawędziowe.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

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

        diff = 0
        for c in range(real.shape[1]):
            channel_real = real[:, c : c + 1, :, :]
            channel_gen = generated[:, c : c + 1, :, :]

            edge_real_x = F.conv2d(channel_real, kernel_x, padding=1)
            edge_gen_x = F.conv2d(channel_gen, kernel_x, padding=1)
            edge_real_y = F.conv2d(channel_real, kernel_y, padding=1)
            edge_gen_y = F.conv2d(channel_gen, kernel_y, padding=1)

            diff += torch.mean(
                torch.abs(edge_real_x - edge_gen_x)
                + torch.abs(edge_real_y - edge_gen_y)
            )

        return (diff / real.shape[1]).item()

    @staticmethod
    def compute_all_metrics(real: torch.Tensor, generated: torch.Tensor) -> dict:
        """
        Oblicza zbiór wszystkich dostępnych metryk.
        """
        return {
            "psnr": ImageMetrics.psnr(real, generated),
            "ssim": ImageMetrics.ssim(real, generated),
            "mae": ImageMetrics.mae(real, generated),
            "mse": ImageMetrics.mse(real, generated),
        }


class MetricsTracker:
    """
    Klasa do śledzenia historii metryk podczas ewaluacji.
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
        Dodaje parę obrazów do ewaluacji.
        """
        metrics = ImageMetrics.compute_all_metrics(real, generated)
        for key, value in metrics.items():
            self.history[key].append(value)

    def get_averages(self) -> dict:
        """
        Zwraca średnie wartości metryk dla zebranej historii.
        """
        averages = {}
        for key, values in self.history.items():
            averages[key] = np.mean(values) if values else 0.0
        return averages

    def print_summary(self) -> None:
        """Wypisuje podsumowanie zebranych metryk."""
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
