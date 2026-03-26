"""Rotation and projection matrix generation for TurboQuant.

Reference: TurboQuant paper (arXiv:2504.19874)
- Rotation: Randomized Hadamard transform (FWHT) — O(d log d), O(d) storage
- Projection matrix S: Algorithm 2, line 3 (i.i.d. Gaussian) — for QJL
"""

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sglang.srt.layers.quantization.turboquant.triton_fwht import (
        triton_fwht_forward,
        triton_fwht_inverse,
    )

    _TRITON_FWHT_AVAILABLE = True
except ImportError:
    _TRITON_FWHT_AVAILABLE = False


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _generate_random_signs(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a deterministic Rademacher vector (+1/-1)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + dim * 7919)
    return (torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1).to(device)


class HadamardTransform:
    """Randomized Hadamard transform for TurboQuant rotation.

    Implements y = (1/sqrt(d)) * H_d * diag(signs) * x where H_d is the
    Walsh-Hadamard matrix and signs are random +/-1 (Rademacher).

    This replaces the QR-based rotation matrix with O(d log d) compute
    and O(d) storage (just a sign vector), matching PR #21419's approach.
    The paper supports any random rotation (Section 3.1).

    Args:
        dim: Original dimension (head_dim)
        seed: Deterministic seed for reproducibility
        device: torch device
    """

    def __init__(self, dim: int, seed: int = 42, device: torch.device = None):
        if device is None:
            device = torch.device("cuda")
        self.dim = dim
        self.padded_dim = _next_power_of_2(dim)
        self.signs = _generate_random_signs(self.padded_dim, seed, device)
        self.scale = 1.0 / math.sqrt(self.padded_dim)
        self.device = device
        self._use_triton = _TRITON_FWHT_AVAILABLE and torch.device(device).type == "cuda"

        # Warm up Triton kernels to trigger JIT compilation before real data
        if self._use_triton:
            dummy = torch.zeros(1, self.padded_dim, device=device)
            triton_fwht_forward(dummy, self.signs, self.padded_dim, self.scale)
            triton_fwht_inverse(dummy, self.signs, self.padded_dim, self.dim, self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomized Hadamard: y = scale * H * diag(signs) * x.

        Args:
            x: (..., dim) tensor
        Returns:
            (..., padded_dim) tensor of rotated coordinates
        """
        if self._use_triton and x.is_cuda:
            return triton_fwht_forward(x, self.signs, self.padded_dim, self.scale)

        orig_shape = x.shape
        d = orig_shape[-1]

        # Pad to power-of-2 if needed
        if d < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - d))

        # Apply random signs
        x = x.float() * self.signs

        # In-place Fast Walsh-Hadamard Transform
        x = self._fwht(x)

        return x * self.scale

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse randomized Hadamard: x = diag(signs) * H * scale * y.

        Since H is symmetric and orthogonal: H^{-1} = H / d.
        Full inverse = diag(signs) * (1/d) * H * (y / scale)
        But scale = 1/sqrt(d), so (1/d) * (1/scale) = 1/sqrt(d) = scale.
        """
        if self._use_triton and y.is_cuda:
            return triton_fwht_inverse(y, self.signs, self.padded_dim, self.dim, self.scale)

        x = self._fwht(y.float()) * self.scale
        x = x * self.signs
        return x[..., :self.dim]

    def to(self, device):
        """Move transform to a new device."""
        self.signs = self.signs.to(device)
        self.device = device
        return self

    @staticmethod
    def _fwht(x: torch.Tensor) -> torch.Tensor:
        """Fast Walsh-Hadamard Transform along the last dimension.

        Uses in-place butterfly operations to avoid per-iteration tensor
        allocations (no torch.stack). Wrapped in torch.compile for fusion.
        """
        return _fwht_impl(x)


def _fwht_impl(x: torch.Tensor) -> torch.Tensor:
    """FWHT fallback (CPU) — in-place butterfly, no torch.stack per iteration."""
    orig_shape = x.shape
    n = orig_shape[-1]
    x = x.reshape(-1, n).float().clone()
    h = 1
    while h < n:
        x_view = x.view(-1, n // (2 * h), 2, h)
        a = x_view[:, :, 0, :].clone()
        b = x_view[:, :, 1, :]
        x_view[:, :, 0, :] = a + b
        x_view[:, :, 1, :] = a - b
        h *= 2
    return x.view(orig_shape)


@lru_cache(maxsize=None)
def projection_matrix(dim: int, seed: int) -> torch.Tensor:
    """Generate a dense Gaussian random projection matrix for QJL.

    NOT orthogonalized — uses raw i.i.d. N(0,1) entries as specified
    in TurboQuant paper Definition 1.

    Args:
        dim: Matrix dimension (head_dim)
        seed: Deterministic seed (seed + dim * 2971 + 17)

    Returns:
        Random matrix of shape [dim, dim]
    """
    if dim <= 0:
        return torch.zeros(0, 0, dtype=torch.float32)

    rng = np.random.default_rng(seed + dim * 2971 + 17)
    matrix = rng.standard_normal((dim, dim)).astype(np.float32)
    return torch.from_numpy(matrix.copy())
