"""Max-Lloyd codebook computation for TurboQuant.

Computes optimal scalar quantizers for the Beta distribution that arises
from projecting unit-sphere vectors through a random orthogonal rotation.

Reference: TurboQuant paper (arXiv:2504.19874), Section 3.1, Eq. (4)
"""

import math
from functools import lru_cache

import numpy as np
import torch


def _beta_pdf(grid: np.ndarray, dim: int) -> np.ndarray:
    """Compute the Beta distribution PDF on [-1, 1] for d-dimensional unit sphere.

    After random rotation, each coordinate follows:
        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
    """
    if dim <= 1:
        return np.ones_like(grid) / len(grid)

    coeff = math.gamma(dim / 2) / (math.sqrt(math.pi) * math.gamma((dim - 1) / 2))
    pdf = coeff * np.power(np.clip(1.0 - grid**2, 0.0, None), (dim - 3) / 2)
    pdf_sum = pdf.sum()
    if pdf_sum == 0:
        return np.full_like(grid, 1.0 / len(grid))
    return pdf / pdf_sum


@lru_cache(maxsize=None)
def compute_codebook(dim: int, bits: int) -> torch.Tensor:
    """Compute Max-Lloyd optimal codebook for the coordinate distribution.

    Args:
        dim: Vector dimension (e.g., 256 for Qwen3.5 head_dim)
        bits: Number of quantization bits per coordinate

    Returns:
        Tensor of shape [2^bits] containing the codebook centroids
    """
    if bits <= 0:
        return torch.zeros(0, dtype=torch.float32)

    levels = 1 << bits

    # Analytical fast-path for b=1
    if bits == 1 and dim > 1:
        c = math.sqrt(2.0 / math.pi) / math.sqrt(dim)
        return torch.tensor([-c, c], dtype=torch.float32)

    if dim <= 1:
        return torch.linspace(-1.0, 1.0, levels, dtype=torch.float32)

    # Fine grid for numerical integration
    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf(grid, dim)

    # Initialize centroids at CDF quantiles
    cdf = np.cumsum(weights)
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.interp(quantiles, cdf, grid).astype(np.float32)

    # Max-Lloyd iteration
    for _ in range(100):
        boundaries = np.empty(levels + 1, dtype=np.float32)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        new_centroids = centroids.copy()
        for i in range(levels):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            if i == levels - 1:
                mask = (grid >= lo) & (grid <= hi)
            else:
                mask = (grid >= lo) & (grid < hi)
            bucket_weights = weights[mask]
            if bucket_weights.size == 0:
                continue
            total_weight = bucket_weights.sum()
            if total_weight > 0:
                new_centroids[i] = np.sum(bucket_weights * grid[mask]) / total_weight

        if np.max(np.abs(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return torch.tensor(centroids.astype(np.float32))


def packed_width(length: int, bits: int) -> int:
    """Compute number of uint8 bytes needed to pack `length` elements of `bits` width."""
    if length == 0 or bits == 0:
        return 0
    return (length * bits + 7) // 8
