"""Quantization and dequantization operations for TurboQuant.

Implements Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod)
from the paper (arXiv:2504.19874).

All ops work on batched tensors of shape [num_tokens, num_heads, head_dim].
"""

import math

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.turboquant.codebook import packed_width

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Bit-packing helpers (pack/unpack arbitrary bit-width into uint8)
# ---------------------------------------------------------------------------

def pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack b-bit integer values into uint8 tensor.

    Uses vectorized tensor-level bitwise ops — no Python per-element loops.
    Supports 1, 2, 3, and 4-bit packing.

    Args:
        values: Integer tensor with values in [0, 2^bits). Shape [..., length]
        bits: Bits per value (1, 2, 3, or 4)

    Returns:
        Packed uint8 tensor of shape [..., packed_dim]
    """
    if bits == 0:
        return torch.zeros(*values.shape[:-1], 0, dtype=torch.uint8, device=values.device)

    batch_shape = values.shape[:-1]
    length = values.shape[-1]
    pw = packed_width(length, bits)

    if bits == 4:
        # 2 values per byte: low nibble = even indices, high nibble = odd indices
        v = values.to(torch.int32)
        even = v[..., 0::2]
        odd = v[..., 1::2]
        packed = ((odd << 4) | (even & 0x0F)).to(torch.uint8)
        # Handle odd length — last value has no pair
        if length % 2 == 1:
            last = v[..., -1:].to(torch.uint8)
            packed = torch.cat([packed, last], dim=-1)
        return packed.reshape(*batch_shape, pw)

    if bits == 2:
        # 4 values per byte
        v = values.to(torch.int32)
        # Pad to multiple of 4
        pad = (4 - length % 4) % 4
        if pad > 0:
            v = F.pad(v, (0, pad))
        groups = v.reshape(*batch_shape, -1, 4)
        packed = (
            (groups[..., 0] & 0x03)
            | ((groups[..., 1] & 0x03) << 2)
            | ((groups[..., 2] & 0x03) << 4)
            | ((groups[..., 3] & 0x03) << 6)
        ).to(torch.uint8)
        return packed[..., :pw].reshape(*batch_shape, pw)

    if bits == 1:
        # 8 values per byte
        v = values.to(torch.int32)
        pad = (8 - length % 8) % 8
        if pad > 0:
            v = F.pad(v, (0, pad))
        groups = v.reshape(*batch_shape, -1, 8)
        packed = groups[..., 0].to(torch.uint8)
        for i in range(1, 8):
            packed = packed | (groups[..., i].to(torch.uint8) << i)
        return packed[..., :pw].reshape(*batch_shape, pw)

    if bits == 3:
        # 8 values → 3 bytes (24 bits)
        v = values.to(torch.int32)
        pad = (8 - length % 8) % 8
        if pad > 0:
            v = F.pad(v, (0, pad))
        groups = v.reshape(*batch_shape, -1, 8)
        # Pack 8 x 3-bit values into a 24-bit integer
        packed_24 = groups[..., 0] & 0x07
        for i in range(1, 8):
            packed_24 = packed_24 | ((groups[..., i] & 0x07) << (i * 3))
        # Split 24-bit integer into 3 bytes
        b0 = (packed_24 & 0xFF).to(torch.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(torch.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(torch.uint8)
        packed = torch.stack([b0, b1, b2], dim=-1).reshape(*batch_shape, -1)
        return packed[..., :pw].reshape(*batch_shape, pw)

    raise ValueError(f"Unsupported bit-width: {bits}")


def unpack_bits(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    """Unpack uint8 tensor to b-bit integer values.

    Uses vectorized tensor-level bitwise ops — no Python per-element loops.
    Supports 1, 2, 3, and 4-bit unpacking.

    Args:
        packed: Packed uint8 tensor of shape [..., packed_dim]
        bits: Bits per value
        length: Number of values to unpack

    Returns:
        Integer tensor of shape [..., length] with values in [0, 2^bits)
    """
    if bits == 0:
        return torch.zeros(*packed.shape[:-1], 0, dtype=torch.int32, device=packed.device)

    batch_shape = packed.shape[:-1]
    mask = (1 << bits) - 1

    if bits == 4:
        # 2 values per byte
        p = packed.to(torch.int32)
        even = p & 0x0F
        odd = (p >> 4) & 0x0F
        unpacked = torch.stack([even, odd], dim=-1).reshape(*batch_shape, -1)
        return unpacked[..., :length]

    if bits == 2:
        # 4 values per byte
        p = packed.to(torch.int32)
        v0 = p & 0x03
        v1 = (p >> 2) & 0x03
        v2 = (p >> 4) & 0x03
        v3 = (p >> 6) & 0x03
        unpacked = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*batch_shape, -1)
        return unpacked[..., :length]

    if bits == 1:
        # 8 values per byte
        p = packed.to(torch.int32)
        vals = [(p >> i) & 1 for i in range(8)]
        unpacked = torch.stack(vals, dim=-1).reshape(*batch_shape, -1)
        return unpacked[..., :length]

    if bits == 3:
        # 3 bytes → 8 values
        p = packed.to(torch.int32)
        # Pad to multiple of 3 bytes
        num_bytes = p.shape[-1]
        pad = (3 - num_bytes % 3) % 3
        if pad > 0:
            p = F.pad(p, (0, pad))
        # Reshape to groups of 3 bytes
        groups = p.reshape(*batch_shape, -1, 3)
        # Reconstruct 24-bit integer
        packed_24 = groups[..., 0] | (groups[..., 1] << 8) | (groups[..., 2] << 16)
        # Extract 8 x 3-bit values
        vals = [(packed_24 >> (i * 3)) & 0x07 for i in range(8)]
        unpacked = torch.stack(vals, dim=-1).reshape(*batch_shape, -1)
        return unpacked[..., :length]

    raise ValueError(f"Unsupported bit-width: {bits}")


# ---------------------------------------------------------------------------
# MSE Quantization (Algorithm 1)
# ---------------------------------------------------------------------------

def mse_quantize(
    x: torch.Tensor,
    hadamard,
    codebook: torch.Tensor,
    bits: int,
) -> tuple:
    """TurboQuant_mse quantization (Algorithm 1).

    Args:
        x: Input vectors [N, H, D]
        hadamard: HadamardTransform instance (or any object with .forward())
        codebook: Centroids [2^bits]
        bits: Bits per coordinate

    Returns:
        (packed_indices [N, H, packed_D], norms [N, H])
    """
    x_f = x.float()
    norms = torch.linalg.norm(x_f, dim=-1)
    unit = x_f / norms.unsqueeze(-1).clamp(min=_EPS)

    # Rotate via Hadamard transform
    rotated = hadamard.forward(unit)  # [N, H, padded_D]

    # Nearest centroid per coordinate
    distances = (rotated.unsqueeze(-1) - codebook).abs()
    indices = distances.argmin(dim=-1)  # [N, H, padded_D]

    packed = pack_bits(indices, bits)
    return packed, norms.half()


def mse_dequantize(
    packed_indices: torch.Tensor,
    norms: torch.Tensor,
    hadamard,
    codebook: torch.Tensor,
    bits: int,
    dim: int,
) -> torch.Tensor:
    """TurboQuant_mse dequantization (Algorithm 1 inverse).

    Args:
        packed_indices: Packed indices [N, H, packed_D]
        norms: Vector norms [N, H]
        hadamard: HadamardTransform instance (or any object with .inverse())
        codebook: Centroids [2^bits]
        bits: Bits per coordinate
        dim: Head dimension (original, pre-padding)

    Returns:
        Reconstructed vectors [N, H, D]
    """
    padded_dim = getattr(hadamard, 'padded_dim', dim)
    indices = unpack_bits(packed_indices, bits, padded_dim)
    cb = codebook
    rotated = cb[indices.long()]  # [N, H, padded_D]

    # Inverse rotate via Hadamard
    unit = hadamard.inverse(rotated)  # [N, H, D]

    return norms.unsqueeze(-1).float() * unit


# ---------------------------------------------------------------------------
# Prod Quantization (Algorithm 2)
# ---------------------------------------------------------------------------

def prod_quantize(
    x: torch.Tensor,
    hadamard,
    s_matrix: torch.Tensor,
    mse_codebook: torch.Tensor,
    bits: int,
) -> tuple:
    """TurboQuant_prod quantization (Algorithm 2).

    Uses (bits-1) bits for MSE component + 1 bit for QJL signs.

    Args:
        x: Input vectors [N, H, D]
        hadamard: HadamardTransform instance
        s_matrix: QJL projection matrix [D, D]
        mse_codebook: Centroids for (bits-1)-bit MSE [2^(bits-1)]
        bits: Total bits per coordinate

    Returns:
        (mse_packed, qjl_packed, norms, residual_norms)
    """
    mse_bits = max(bits - 1, 0)
    dim = x.shape[-1]
    padded_dim = getattr(hadamard, 'padded_dim', dim)
    x_f = x.float()

    norms = torch.linalg.norm(x_f, dim=-1)
    unit = x_f / norms.unsqueeze(-1).clamp(min=_EPS)

    # MSE component at (bits-1) bits — rotate via Hadamard
    rotated = hadamard.forward(unit)  # [N, H, padded_D]

    if mse_bits > 0:
        distances = (rotated.unsqueeze(-1) - mse_codebook).abs()
        mse_indices = distances.argmin(dim=-1)
        mse_packed = pack_bits(mse_indices, mse_bits)

        # Dequantize MSE to compute residual (inverse rotate)
        mse_rotated = mse_codebook[mse_indices.long()]
        mse_unit = hadamard.inverse(mse_rotated)  # [N, H, D]
    else:
        pw = packed_width(padded_dim, mse_bits)
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


# ---------------------------------------------------------------------------
# Split-Channel Codec (fractional bit-widths, e.g. 3.5-bit)
# ---------------------------------------------------------------------------

def select_outlier_indices(
    head_dim: int, avg_bits: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic channel split for fractional bit-widths.

    Splits head_dim channels into a lo-group (floor(bits)) and hi-group
    (ceil(bits)). Rotation randomizes channels, so no calibration needed.

    Paper: Section 3.2, mixed-precision split.

    Args:
        head_dim: Number of channels per head.
        avg_bits: Average bits per coordinate (e.g. 3.5).

    Returns:
        (lo_indices, hi_indices) — 1-D int64 tensors on CPU.
        For integer bits, hi_indices is empty.
    """
    frac = avg_bits - math.floor(avg_bits)
    if abs(frac) < 1e-9:
        # Integer bits — everything goes in the lo group
        return torch.arange(head_dim, dtype=torch.int64), torch.empty(
            0, dtype=torch.int64
        )

    high_count = round(frac * head_dim)
    high_count = max(1, min(head_dim - 1, high_count))
    lo_count = head_dim - high_count
    lo_indices = torch.arange(0, lo_count, dtype=torch.int64)
    hi_indices = torch.arange(lo_count, head_dim, dtype=torch.int64)
    return lo_indices, hi_indices


def split_channel_mse_quantize(
    x: torch.Tensor,
    lo_indices: torch.Tensor,
    hi_indices: torch.Tensor,
    hadamard_lo,
    hadamard_hi,
    cb_lo: torch.Tensor,
    cb_hi: torch.Tensor,
    lo_bits: int,
    hi_bits: int,
) -> tuple:
    """Split-channel TurboQuant_mse quantization.

    Splits x along the last dimension into lo/hi groups, then independently
    applies mse_quantize to each group with its own Hadamard transform and codebook.

    Args:
        x: Input vectors [N, H, D]
        lo_indices, hi_indices: Channel index tensors from select_outlier_indices
        hadamard_lo, hadamard_hi: HadamardTransform instances per group
        cb_lo, cb_hi: Codebooks for each group
        lo_bits, hi_bits: Bit-widths for each group

    Returns:
        (lo_packed, lo_norms, hi_packed, hi_norms)
    """
    dev = x.device
    x_lo = x.index_select(-1, lo_indices.to(dev))
    x_hi = x.index_select(-1, hi_indices.to(dev))

    lo_packed, lo_norms = mse_quantize(x_lo, hadamard_lo, cb_lo, lo_bits)
    hi_packed, hi_norms = mse_quantize(x_hi, hadamard_hi, cb_hi, hi_bits)

    return lo_packed, lo_norms, hi_packed, hi_norms


def split_channel_mse_dequantize(
    lo_packed: torch.Tensor,
    lo_norms: torch.Tensor,
    hi_packed: torch.Tensor,
    hi_norms: torch.Tensor,
    lo_indices: torch.Tensor,
    hi_indices: torch.Tensor,
    restore_order: torch.Tensor,
    hadamard_lo,
    hadamard_hi,
    cb_lo: torch.Tensor,
    cb_hi: torch.Tensor,
    lo_bits: int,
    hi_bits: int,
    dim: int,
) -> torch.Tensor:
    """Split-channel TurboQuant_mse dequantization.

    Dequantizes each group independently, concatenates, then restores
    the original channel order.

    Args:
        lo_packed, lo_norms: Packed indices and norms for lo group
        hi_packed, hi_norms: Packed indices and norms for hi group
        lo_indices, hi_indices: Channel index tensors
        restore_order: argsort(cat(lo_indices, hi_indices)) permutation
        hadamard_lo, hadamard_hi: HadamardTransform instances per group
        cb_lo, cb_hi: Codebooks
        lo_bits, hi_bits: Bit-widths
        dim: Original head dimension (lo + hi)

    Returns:
        Reconstructed vectors [N, H, D]
    """
    d_lo = lo_indices.shape[0]
    d_hi = hi_indices.shape[0]

    lo_recon = mse_dequantize(lo_packed, lo_norms, hadamard_lo, cb_lo, lo_bits, d_lo)
    hi_recon = mse_dequantize(hi_packed, hi_norms, hadamard_hi, cb_hi, hi_bits, d_hi)

    merged = torch.cat([lo_recon, hi_recon], dim=-1)
    return merged.index_select(-1, restore_order.to(merged.device))


def split_channel_prod_quantize(
    x: torch.Tensor,
    lo_indices: torch.Tensor,
    hi_indices: torch.Tensor,
    hadamard_lo,
    hadamard_hi,
    s_lo: torch.Tensor,
    s_hi: torch.Tensor,
    cb_lo: torch.Tensor,
    cb_hi: torch.Tensor,
    lo_bits: int,
    hi_bits: int,
) -> tuple:
    """Split-channel TurboQuant_prod quantization.

    Each group independently runs TurboQuant_prod (MSE at group_bits-1 + QJL at 1-bit).

    Args:
        x: Input vectors [N, H, D]
        lo_indices, hi_indices: Channel splits
        hadamard_lo, hadamard_hi: Per-group HadamardTransform instances
        s_lo, s_hi: Per-group QJL projection matrices
        cb_lo, cb_hi: Per-group MSE codebooks (for bits-1)
        lo_bits, hi_bits: Total bits per group

    Returns:
        (lo_mse_packed, lo_qjl_packed, lo_norms, lo_res_norms,
         hi_mse_packed, hi_qjl_packed, hi_norms, hi_res_norms)
    """
    dev = x.device
    x_lo = x.index_select(-1, lo_indices.to(dev))
    x_hi = x.index_select(-1, hi_indices.to(dev))

    lo_mse_p, lo_qjl_p, lo_n, lo_rn = prod_quantize(
        x_lo, hadamard_lo, s_lo, cb_lo, lo_bits
    )
    hi_mse_p, hi_qjl_p, hi_n, hi_rn = prod_quantize(
        x_hi, hadamard_hi, s_hi, cb_hi, hi_bits
    )

    return lo_mse_p, lo_qjl_p, lo_n, lo_rn, hi_mse_p, hi_qjl_p, hi_n, hi_rn


def split_channel_prod_dequantize(
    lo_mse_packed: torch.Tensor,
    lo_qjl_packed: torch.Tensor,
    lo_norms: torch.Tensor,
    lo_res_norms: torch.Tensor,
    hi_mse_packed: torch.Tensor,
    hi_qjl_packed: torch.Tensor,
    hi_norms: torch.Tensor,
    hi_res_norms: torch.Tensor,
    lo_indices: torch.Tensor,
    hi_indices: torch.Tensor,
    restore_order: torch.Tensor,
    hadamard_lo,
    hadamard_hi,
    s_lo: torch.Tensor,
    s_hi: torch.Tensor,
    cb_lo: torch.Tensor,
    cb_hi: torch.Tensor,
    lo_bits: int,
    hi_bits: int,
    dim: int,
) -> torch.Tensor:
    """Split-channel TurboQuant_prod dequantization.

    Dequantizes each group independently, concatenates, restores channel order.

    Returns:
        Reconstructed vectors [N, H, D]
    """
    d_lo = lo_indices.shape[0]
    d_hi = hi_indices.shape[0]

    lo_recon = prod_dequantize(
        lo_mse_packed, lo_qjl_packed, lo_norms, lo_res_norms,
        hadamard_lo, s_lo, cb_lo, lo_bits, d_lo,
    )
    hi_recon = prod_dequantize(
        hi_mse_packed, hi_qjl_packed, hi_norms, hi_res_norms,
        hadamard_hi, s_hi, cb_hi, hi_bits, d_hi,
    )

    merged = torch.cat([lo_recon, hi_recon], dim=-1)
    return merged.index_select(-1, restore_order.to(merged.device))


def prod_dequantize(
    mse_packed: torch.Tensor,
    qjl_packed: torch.Tensor,
    norms: torch.Tensor,
    residual_norms: torch.Tensor,
    hadamard,
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
    padded_dim = getattr(hadamard, 'padded_dim', dim)

    # MSE component
    if mse_bits > 0:
        mse_indices = unpack_bits(mse_packed, mse_bits, padded_dim)
        mse_rotated = mse_codebook[mse_indices.long()]
        mse_unit = hadamard.inverse(mse_rotated)  # [N, H, D]
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
        * torch.matmul(signs, s_matrix)
    )

    return norms.unsqueeze(-1).float() * (mse_unit + qjl_unit)
