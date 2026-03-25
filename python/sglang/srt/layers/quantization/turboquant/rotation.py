"""Rotation and projection matrix generation for TurboQuant.

Reference: TurboQuant paper (arXiv:2504.19874)
- Rotation matrix Pi: Algorithm 1, line 2 (QR decomposition, Haar-uniform)
- Projection matrix S: Algorithm 2, line 3 (i.i.d. Gaussian)
"""

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=None)
def rotation_matrix(dim: int, seed: int) -> torch.Tensor:
    """Generate a Haar-uniform random orthogonal matrix via QR decomposition.

    Uses sign correction Q *= sign(diag(R)) for deterministic Haar-uniform
    distribution, as confirmed by the MLX reference implementation.

    Args:
        dim: Matrix dimension (head_dim)
        seed: Deterministic seed (seed + dim * 7919)

    Returns:
        Orthogonal matrix of shape [dim, dim]
    """
    if dim <= 0:
        return torch.zeros(0, 0, dtype=torch.float32)
    if dim == 1:
        return torch.ones(1, 1, dtype=torch.float32)

    rng = np.random.default_rng(seed + dim * 7919)
    matrix = rng.standard_normal((dim, dim)).astype(np.float32)
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))  # Haar-uniform correction
    return torch.from_numpy(q.copy())


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
