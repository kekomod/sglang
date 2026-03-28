"""Rotation matrix generation for TurboQuant.

Reference: TurboQuant paper (arXiv:2504.19874)
- Rotation: Haar-uniform random orthogonal via QR decomposition (Section 3.1, Algorithm 1)

The rotation matrix Π must be Haar-uniform so that Π·x is uniformly distributed
on S^{d-1}. This guarantees coordinates follow the Beta distribution (Lemma 1),
which the codebook is optimized for.
"""

import numpy as np
import torch


class HadamardTransform:
    """Haar-uniform random orthogonal rotation for TurboQuant.

    Uses QR decomposition of a Gaussian matrix with sign correction,
    matching the paper (Section 3.1, Algorithm 1) and MLX reference.

    Name kept as HadamardTransform for backward compatibility with the
    rest of the codebase (pool, backend, tests).

    Args:
        dim: Original dimension (head_dim or split group dim)
        seed: Deterministic seed for reproducibility
        device: torch device
    """

    def __init__(self, dim: int, seed: int = 42, device: torch.device = None):
        if device is None:
            device = torch.device("cuda")
        self.dim = dim
        self.padded_dim = dim  # No padding needed with dense rotation
        self.device = device

        # Generate Haar-uniform random orthogonal matrix via QR
        # Same seed formula as MLX: seed + dim * 7919
        rng = np.random.default_rng(seed + dim * 7919)
        if dim > 0:
            matrix = rng.standard_normal((dim, dim)).astype(np.float32)
            q, r = np.linalg.qr(matrix)
            q *= np.sign(np.diag(r))  # Sign correction for Haar uniformity
            self.rotation = torch.from_numpy(q.copy()).to(device)
            self.rotation_t = self.rotation.T.contiguous()
        else:
            self.rotation = torch.zeros(0, 0, dtype=torch.float32, device=device)
            self.rotation_t = self.rotation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: y = Π · x = x @ Π^T.

        Args:
            x: (..., dim) tensor
        Returns:
            (..., dim) tensor of rotated coordinates
        """
        return torch.matmul(x.float(), self.rotation_t)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: x = Π^T · y = y @ Π.

        Since Π is orthogonal: Π^{-1} = Π^T.
        """
        result = torch.matmul(y.float(), self.rotation)
        return result[..., :self.dim]

    def to(self, device):
        """Move transform to a new device."""
        self.rotation = self.rotation.to(device)
        self.rotation_t = self.rotation_t.to(device)
        self.device = device
        return self


