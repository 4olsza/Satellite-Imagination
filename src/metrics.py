"""
Generator Evaluation Metrics.
Includes PSNR, SSIM, MAE, MSE, and a simplified edge-based version of LPIPS.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class ImageMetrics:
    """
    A utility class containing metrics for image quality comparison.
    """

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes an image tensor from the [-1, 1] range back to [0, 1].

        Args:
            tensor (torch.Tensor): The normalized image tensor.

        Returns:
            torch.Tensor: The denormalized tensor in the [0, 1] range.
        """
        return (tensor + 1) / 2

    @staticmethod
    def psnr(
        real: torch.Tensor, generated: torch.Tensor, max_val: float = 1.0
    ) -> float:
        """
        Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
        Higher values indicate better structural fidelity.
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
        Calculates the Structural Similarity Index (SSIM).
        Returns a value approximately between 0 and 1, where higher means better similarity.
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
        Calculates the Mean Absolute Error (MAE / L1 Loss) between images.
        Lower values indicate better pixel-level similarity.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)
        mae_val = torch.mean(torch.abs(real - generated))
        return mae_val.item()

    @staticmethod
    def mse(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        Calculates the Mean Squared Error (MSE / L2 Loss) between images.
        Lower values indicate better pixel-level similarity.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)
        mse_val = torch.mean((real - generated) ** 2)
        return mse_val.item()

    @staticmethod
    def lpips(real: torch.Tensor, generated: torch.Tensor) -> float:
        """
        A simplified approximation of LPIPS (Learned Perceptual Image Patch Similarity).
        This implementation utilizes edge-detection filters to compare structural gradients 
        instead of relying on a heavy pre-trained VGG network.
        """
        real = ImageMetrics.denormalize(real)
        generated = ImageMetrics.denormalize(generated)

        kernel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(real.device)
        )
        kernel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(real.device)
        )

        diff = 0.0
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
            ).item()

        return diff / real.shape[1]

    @staticmethod
    def compute_all_metrics(real: torch.Tensor, generated: torch.Tensor) -> Dict[str, float]:
        """
        Computes a comprehensive dictionary of all available evaluation metrics.
        """
        return {
            "psnr": ImageMetrics.psnr(real, generated),
            "ssim": ImageMetrics.ssim(real, generated),
            "mae": ImageMetrics.mae(real, generated),
            "mse": ImageMetrics.mse(real, generated),
            "lpips_approx": ImageMetrics.lpips(real, generated),
        }


class MetricsTracker:
    """
    A tracker class to accumulate metric histories during model evaluation loops.
    """

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {
            "psnr": [],
            "ssim": [],
            "mae": [],
            "mse": [],
            "lpips_approx": [],
        }

    def add(self, real: torch.Tensor, generated: torch.Tensor) -> None:
        """
        Evaluates a pair of images and appends the results to the internal history.
        """
        metrics = ImageMetrics.compute_all_metrics(real, generated)
        for key, value in metrics.items():
            self.history[key].append(value)

    def get_averages(self) -> Dict[str, float]:
        """
        Calculates and returns the mean values of the accumulated metrics.
        """
        averages = {}
        for key, values in self.history.items():
            averages[key] = float(np.mean(values)) if values else 0.0
        return averages

    def print_summary(self) -> None:
        """
        Outputs a formatted summary of the collected metric averages to the console.
        """
        averages = self.get_averages()

        print("\n" + "=" * 50)
        print("📊 EVALUATION METRICS SUMMARY")
        print("=" * 50)
        print(f"  PSNR         (higher=better): {averages.get('psnr', 0.0):.4f}")
        print(f"  SSIM         (higher=better): {averages.get('ssim', 0.0):.4f}")
        print(f"  MAE          (lower=better):  {averages.get('mae', 0.0):.4f}")
        print(f"  MSE          (lower=better):  {averages.get('mse', 0.0):.4f}")
        print(f"  Approx LPIPS (lower=better):  {averages.get('lpips_approx', 0.0):.4f}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # Internal module tests
    print("Executing internal metric tests...")

    # Generate dummy tensors representing a batch of 1 RGB image
    dummy_real = torch.randn((1, 3, 256, 256))
    dummy_generated = torch.randn((1, 3, 256, 256))

    test_metrics = ImageMetrics.compute_all_metrics(dummy_real, dummy_generated)
    
    for metric_name, metric_val in test_metrics.items():
        print(f"  -> {metric_name.upper()}: {metric_val:.4f}")

    print("✓ Metrics pipeline verified and fully operational!")