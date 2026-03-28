"""
TurboQuant Validation Benchmark — Empirical Distortion (Section 4.1)

Standalone GPU script (no server needed). Validates:
  1. MSE distortion: ||x - x̂||² / ||x||² vs theoretical bound
  2. Inner product distortion: <x_i, x_j> vs <x̂_i, x̂_j> bias and variance

Reference: arXiv:2504.19874, Section 4.1

Usage:
    python eval_distortion.py [--dim 128] [--num-vectors 10000] [--device cuda]
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# Add parent to path for common module
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sglang.srt.layers.quantization.turboquant.codebook import compute_codebook
from sglang.srt.layers.quantization.turboquant.quant_ops import (
    mse_dequantize,
    mse_quantize,
)
from sglang.srt.layers.quantization.turboquant.rotation import (
    HadamardTransform,
)


def generate_unit_vectors(n: int, dim: int, seed: int = 42) -> torch.Tensor:
    """Generate n random unit-sphere vectors by normalizing Gaussian vectors.

    Returns: [n, 1, dim] (1 dummy head dim for compatibility with quant ops).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 1, dim, generator=gen)
    norms = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norms


def compute_theoretical_mse(dim: int, bits: int) -> float:
    """Compute theoretical per-vector MSE from codebook quantization error.

    For a unit vector x on S^{d-1}, after random rotation each coordinate
    follows the Beta distribution. The per-coordinate MSE is:
        E[(x_i - Q(x_i))^2]
    and the per-vector MSE (normalized) is d * E[(x_i - Q(x_i))^2].

    We compute this numerically from the Max-Lloyd codebook.
    """
    import numpy as np
    from sglang.srt.layers.quantization.turboquant.codebook import _beta_pdf

    codebook = compute_codebook(dim, bits).numpy()
    grid = np.linspace(-1.0 + 1e-6, 1.0 - 1e-6, 32768, dtype=np.float32)
    weights = _beta_pdf(grid, dim)

    # Per-coordinate MSE: E[(x - Q(x))^2]
    # For each grid point, find nearest centroid and compute squared error
    distances = np.abs(grid[:, None] - codebook[None, :])
    nearest = np.argmin(distances, axis=1)
    sq_errors = (grid - codebook[nearest]) ** 2
    per_coord_mse = np.sum(sq_errors * weights)

    # Per-vector normalized MSE: d * per_coord_mse (since ||x||=1)
    return float(dim * per_coord_mse)


def measure_mse_distortion(
    x: torch.Tensor, dim: int, bits: int, device: torch.device,
) -> dict:
    """Quantize with TQ_mse and measure MSE distortion."""
    hadamard = HadamardTransform(dim, seed=42, device=device)
    codebook = compute_codebook(hadamard.padded_dim, bits).to(device)

    packed, norms = mse_quantize(x, hadamard, codebook, bits)
    x_hat = mse_dequantize(packed, norms, hadamard, codebook, bits, dim)

    # Normalized MSE: ||x - x_hat||^2 / ||x||^2
    diff = (x.float() - x_hat.float())
    mse_per_vec = (diff ** 2).sum(dim=-1) / (x.float() ** 2).sum(dim=-1).clamp(min=1e-12)
    mse_per_vec = mse_per_vec.squeeze(-1)  # [N]

    return {
        "mean": mse_per_vec.mean().item(),
        "std": mse_per_vec.std().item(),
        "x_hat": x_hat,
    }


def measure_inner_product_distortion(
    x: torch.Tensor, x_hat: torch.Tensor, num_pairs: int = 50000, seed: int = 123,
) -> dict:
    """Compare <x_i, x_j> vs <x̂_i, x̂_j> for random pairs.

    Measures bias and variance of the inner product estimator.
    """
    n = x.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx_i = torch.randint(0, n, (num_pairs,), generator=gen)
    idx_j = torch.randint(0, n, (num_pairs,), generator=gen)

    # Squeeze head dim: [N, 1, D] -> [N, D]
    x_flat = x.squeeze(1).float()
    xh_flat = x_hat.squeeze(1).float()

    true_ip = (x_flat[idx_i] * x_flat[idx_j]).sum(dim=-1)
    est_ip = (xh_flat[idx_i] * xh_flat[idx_j]).sum(dim=-1)

    error = est_ip - true_ip
    bias = error.mean().item()
    variance = error.var().item()
    rmse = error.pow(2).mean().sqrt().item()

    return {"bias": bias, "variance": variance, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant empirical distortion validation (Section 4.1)"
    )
    parser.add_argument("--dim", type=int, default=128,
                        help="Vector dimension (default: 128)")
    parser.add_argument("--num-vectors", type=int, default=10000,
                        help="Number of random unit vectors (default: 10000)")
    parser.add_argument("--num-pairs", type=int, default=50000,
                        help="Number of random pairs for inner product test (default: 50000)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4],
                        help="Bit-widths to test (default: 2 3 4)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dim = args.dim
    n = args.num_vectors

    print(f"[distortion] Generating {n} unit vectors of dim={dim} on {device}")
    x = generate_unit_vectors(n, dim).to(device)

    # Results table
    results = []

    for bits in args.bits:
        print(f"\n{'='*70}")
        print(f"  Bit-width: {bits}")
        print(f"{'='*70}")

        # Theoretical MSE bound
        theo_mse = compute_theoretical_mse(dim, bits)

        # TQ_mse
        mse_res = measure_mse_distortion(x, dim, bits, device)
        mse_ip = measure_inner_product_distortion(x, mse_res["x_hat"], args.num_pairs)

        results.append({
            "bits": bits,
            "theoretical_mse": theo_mse,
            "mse_quant": mse_res,
            "mse_ip": mse_ip,
        })

        print(f"\n  MSE Distortion (||x - x̂||² / ||x||²):")
        print(f"    Theoretical bound:  {theo_mse:.6f}")
        print(f"    TQ_mse measured:    {mse_res['mean']:.6f} ± {mse_res['std']:.6f}")

        print(f"\n  Inner Product Distortion (<x̂_i,x̂_j> - <x_i,x_j>):")
        print(f"    TQ_mse  — bias: {mse_ip['bias']:+.6f}  var: {mse_ip['variance']:.6f}  RMSE: {mse_ip['rmse']:.6f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    header = f"{'Bits':>5} {'Theo MSE':>10} {'Meas MSE':>10} {'MSE Std':>10} {'IP Bias':>10} {'IP RMSE':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        bits = r["bits"]
        theo = r["theoretical_mse"]
        meas = r["mse_quant"]["mean"]
        std = r["mse_quant"]["std"]
        bias = r["mse_ip"]["bias"]
        rmse = r["mse_ip"]["rmse"]
        print(f"{bits:>5} {theo:>10.6f} {meas:>10.6f} {std:>10.6f} {bias:>+10.6f} {rmse:>10.6f}")

    print(f"\n[distortion] Done. {n} vectors, dim={dim}, device={device}")


if __name__ == "__main__":
    main()
