"""Quantization and dequantization operations for TurboQuant.

Implements Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod)
from the paper (arXiv:2504.19874).

All ops work on batched tensors of shape [num_tokens, num_heads, head_dim].
"""

import math

import torch

from sglang.srt.layers.quantization.turboquant.codebook import packed_width

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Bit-packing helpers (pack/unpack arbitrary bit-width into uint8)
# ---------------------------------------------------------------------------

def pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack b-bit integer values into uint8 tensor.

    Args:
        values: Integer tensor with values in [0, 2^bits). Shape [..., length]
        bits: Bits per value (1, 2, 3, or 4)

    Returns:
        Packed uint8 tensor of shape [..., packed_dim]
    """
    if bits == 0:
        return torch.zeros(*values.shape[:-1], 0, dtype=torch.uint8, device=values.device)

    length = values.shape[-1]
    pw = packed_width(length, bits)
    flat = values.reshape(-1, length).to(torch.int32)
    batch = flat.shape[0]

    packed = torch.zeros(batch, pw, dtype=torch.uint8, device=values.device)

    for idx in range(length):
        byte_offset = (idx * bits) // 8
        bit_offset = (idx * bits) % 8
        val = flat[:, idx]

        # Write bits into current byte
        packed[:, byte_offset] |= ((val << bit_offset) & 0xFF).to(torch.uint8)

        # Handle spill into next byte
        spill = bit_offset + bits - 8
        if spill > 0 and byte_offset + 1 < pw:
            packed[:, byte_offset + 1] |= ((val >> (bits - spill)) & 0xFF).to(torch.uint8)

    return packed.reshape(*values.shape[:-1], pw)


def unpack_bits(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    """Unpack uint8 tensor to b-bit integer values.

    Args:
        packed: Packed uint8 tensor of shape [..., packed_dim]
        bits: Bits per value
        length: Number of values to unpack

    Returns:
        Integer tensor of shape [..., length] with values in [0, 2^bits)
    """
    if bits == 0:
        return torch.zeros(*packed.shape[:-1], 0, dtype=torch.int32, device=packed.device)

    flat = packed.reshape(-1, packed.shape[-1]).to(torch.int32)
    batch = flat.shape[0]
    mask = (1 << bits) - 1

    unpacked = torch.zeros(batch, length, dtype=torch.int32, device=packed.device)

    for idx in range(length):
        byte_offset = (idx * bits) // 8
        bit_offset = (idx * bits) % 8
        val = flat[:, byte_offset] >> bit_offset

        spill = bit_offset + bits - 8
        if spill > 0 and byte_offset + 1 < flat.shape[1]:
            val |= flat[:, byte_offset + 1] << (bits - spill)

        unpacked[:, idx] = val & mask

    return unpacked.reshape(*packed.shape[:-1], length)


# ---------------------------------------------------------------------------
# MSE Quantization (Algorithm 1)
# ---------------------------------------------------------------------------

def mse_quantize(
    x: torch.Tensor,
    pi: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
) -> tuple:
    """TurboQuant_mse quantization (Algorithm 1).

    Args:
        x: Input vectors [N, H, D]
        pi: Rotation matrix [D, D]
        codebook: Centroids [2^bits]
        bits: Bits per coordinate

    Returns:
        (packed_indices [N, H, packed_D], norms [N, H])
    """
    x_f = x.float()
    norms = torch.linalg.norm(x_f, dim=-1)
    unit = x_f / norms.unsqueeze(-1).clamp(min=_EPS)

    # Rotate
    rotated = torch.matmul(unit, pi.T)  # [N, H, D]

    # Nearest centroid per coordinate
    distances = (rotated.unsqueeze(-1) - codebook.to(rotated.device)).abs()
    indices = distances.argmin(dim=-1)  # [N, H, D]

    packed = pack_bits(indices, bits)
    return packed, norms.half()


def mse_dequantize(
    packed_indices: torch.Tensor,
    norms: torch.Tensor,
    pi_t: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    dim: int,
) -> torch.Tensor:
    """TurboQuant_mse dequantization (Algorithm 1 inverse).

    Args:
        packed_indices: Packed indices [N, H, packed_D]
        norms: Vector norms [N, H]
        pi_t: Transposed rotation matrix [D, D]
        codebook: Centroids [2^bits]
        bits: Bits per coordinate
        dim: Head dimension

    Returns:
        Reconstructed vectors [N, H, D]
    """
    indices = unpack_bits(packed_indices, bits, dim)
    cb = codebook.to(indices.device)
    rotated = cb[indices.long()]  # [N, H, D]

    # Inverse rotate
    unit = torch.matmul(rotated, pi_t.T)

    return norms.unsqueeze(-1).float() * unit


# ---------------------------------------------------------------------------
# Prod Quantization (Algorithm 2)
# ---------------------------------------------------------------------------

def prod_quantize(
    x: torch.Tensor,
    pi: torch.Tensor,
    s_matrix: torch.Tensor,
    mse_codebook: torch.Tensor,
    bits: int,
) -> tuple:
    """TurboQuant_prod quantization (Algorithm 2).

    Uses (bits-1) bits for MSE component + 1 bit for QJL signs.

    Args:
        x: Input vectors [N, H, D]
        pi: Rotation matrix [D, D]
        s_matrix: QJL projection matrix [D, D]
        mse_codebook: Centroids for (bits-1)-bit MSE [2^(bits-1)]
        bits: Total bits per coordinate

    Returns:
        (mse_packed, qjl_packed, norms, residual_norms)
    """
    mse_bits = max(bits - 1, 0)
    dim = x.shape[-1]
    x_f = x.float()

    norms = torch.linalg.norm(x_f, dim=-1)
    unit = x_f / norms.unsqueeze(-1).clamp(min=_EPS)

    # MSE component at (bits-1) bits
    rotated = torch.matmul(unit, pi.T)

    if mse_bits > 0:
        cb = mse_codebook.to(rotated.device)
        distances = (rotated.unsqueeze(-1) - cb).abs()
        mse_indices = distances.argmin(dim=-1)
        mse_packed = pack_bits(mse_indices, mse_bits)

        # Dequantize MSE to compute residual
        mse_rotated = cb[mse_indices.long()]
        mse_unit = torch.matmul(mse_rotated, pi.contiguous())  # pi^T^T = pi
    else:
        pw = packed_width(dim, mse_bits)
        mse_packed = torch.zeros(
            *x.shape[:-1], pw, dtype=torch.uint8, device=x.device
        )
        mse_unit = torch.zeros_like(unit)

    # Residual
    residual = unit - mse_unit
    residual_norms = torch.linalg.norm(residual, dim=-1)

    # QJL: sign of projection
    projected = torch.matmul(residual, s_matrix.T)
    signs = (projected >= 0).to(torch.uint8)
    qjl_packed = pack_bits(signs, 1)

    return mse_packed, qjl_packed, norms.half(), residual_norms.half()


def prod_dequantize(
    mse_packed: torch.Tensor,
    qjl_packed: torch.Tensor,
    norms: torch.Tensor,
    residual_norms: torch.Tensor,
    pi_t: torch.Tensor,
    s_matrix: torch.Tensor,
    mse_codebook: torch.Tensor,
    bits: int,
    dim: int,
) -> torch.Tensor:
    """TurboQuant_prod dequantization (Algorithm 2 inverse).

    Returns:
        Reconstructed vectors [N, H, D]
    """
    mse_bits = max(bits - 1, 0)

    # MSE component
    if mse_bits > 0:
        mse_indices = unpack_bits(mse_packed, mse_bits, dim)
        cb = mse_codebook.to(mse_indices.device)
        mse_rotated = cb[mse_indices.long()]
        mse_unit = torch.matmul(mse_rotated, pi_t.T)
    else:
        mse_unit = torch.zeros(
            *norms.shape, dim, dtype=torch.float32, device=norms.device
        )

    # QJL component
    signs_01 = unpack_bits(qjl_packed, 1, dim).float()
    signs = signs_01 * 2.0 - 1.0  # {0,1} → {-1,+1}

    scale = math.sqrt(math.pi / 2) / dim if dim > 0 else 0.0
    qjl_unit = (
        scale
        * residual_norms.unsqueeze(-1).float()
        * torch.matmul(signs, s_matrix.to(signs.device))
    )

    return norms.unsqueeze(-1).float() * (mse_unit + qjl_unit)
