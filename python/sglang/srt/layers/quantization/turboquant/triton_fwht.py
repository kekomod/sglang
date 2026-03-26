"""Triton kernels for Fast Walsh-Hadamard Transform (FWHT).

Replaces torch.compile FWHT with dedicated Triton kernels for forward
(quantization) and inverse (dequantization) randomized Hadamard transforms.

Forward: y = scale * H * diag(signs) * x   (signs BEFORE FWHT)
Inverse: x = diag(signs) * H * scale * y   (signs AFTER FWHT)

Reference: TurboQuant paper (arXiv:2504.19874), Section 3.1
"""

import triton
import triton.language as tl
import torch


@triton.jit
def triton_fwht_forward_kernel(
    in_ptr,
    out_ptr,
    signs_ptr,
    scale,
    stride_row,
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """Forward FWHT: multiply by signs, then butterfly, then scale.

    Each program handles one row of D elements.
    Uses output buffer as scratch space with barriers between stages.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    base = row * stride_row

    # Load input and multiply by signs
    x = tl.load(in_ptr + base + offs).to(tl.float32)
    s = tl.load(signs_ptr + offs)
    x = x * s

    # Store to output buffer (used as scratch)
    tl.store(out_ptr + base + offs, x)
    tl.debug_barrier()

    # Butterfly stages
    for stage in tl.static_range(LOG2_D):
        stride = 1 << stage
        top_mask = (offs & stride) == 0
        partner = offs ^ stride
        x_self = tl.load(out_ptr + base + offs)
        x_partner = tl.load(out_ptr + base + partner)
        x = tl.where(top_mask, x_self + x_partner, x_partner - x_self)
        tl.store(out_ptr + base + offs, x)
        tl.debug_barrier()

    # Scale
    x = tl.load(out_ptr + base + offs) * scale
    tl.store(out_ptr + base + offs, x)


@triton.jit
def triton_fwht_inverse_kernel(
    in_ptr,
    out_ptr,
    signs_ptr,
    scale,
    stride_row,
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
):
    """Inverse FWHT: butterfly, then scale, then multiply by signs.

    Each program handles one row of D elements.
    Uses output buffer as scratch space with barriers between stages.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    base = row * stride_row

    # Load input
    x = tl.load(in_ptr + base + offs).to(tl.float32)

    # Store to output buffer (used as scratch)
    tl.store(out_ptr + base + offs, x)
    tl.debug_barrier()

    # Butterfly stages
    for stage in tl.static_range(LOG2_D):
        stride = 1 << stage
        top_mask = (offs & stride) == 0
        partner = offs ^ stride
        x_self = tl.load(out_ptr + base + offs)
        x_partner = tl.load(out_ptr + base + partner)
        x = tl.where(top_mask, x_self + x_partner, x_partner - x_self)
        tl.store(out_ptr + base + offs, x)
        tl.debug_barrier()

    # Scale then multiply by signs
    x = tl.load(out_ptr + base + offs) * scale
    s = tl.load(signs_ptr + offs)
    x = x * s
    tl.store(out_ptr + base + offs, x)


def _log2(n: int) -> int:
    """Integer log2 for powers of 2."""
    assert n > 0 and (n & (n - 1)) == 0, f"{n} is not a power of 2"
    return n.bit_length() - 1


def triton_fwht_forward(
    x: torch.Tensor,
    signs: torch.Tensor,
    padded_dim: int,
    scale: float,
) -> torch.Tensor:
    """Apply forward randomized Hadamard: y = scale * H * diag(signs) * x.

    Args:
        x: (..., dim) tensor, any shape
        signs: (padded_dim,) sign vector
        padded_dim: power-of-2 dimension
        scale: 1/sqrt(padded_dim)

    Returns:
        (..., padded_dim) tensor
    """
    orig_shape = x.shape
    d = orig_shape[-1]

    # Pad if needed
    if d < padded_dim:
        import torch.nn.functional as F
        x = F.pad(x, (0, padded_dim - d))

    # Reshape to 2D: [num_rows, padded_dim]
    x_flat = x.reshape(-1, padded_dim).float().contiguous()
    num_rows = x_flat.shape[0]

    out = torch.empty_like(x_flat)
    log2_d = _log2(padded_dim)

    triton_fwht_forward_kernel[(num_rows,)](
        x_flat, out, signs,
        scale,
        padded_dim,  # stride_row
        D=padded_dim,
        LOG2_D=log2_d,
    )

    new_shape = list(orig_shape)
    new_shape[-1] = padded_dim
    return out.view(new_shape)


def triton_fwht_inverse(
    y: torch.Tensor,
    signs: torch.Tensor,
    padded_dim: int,
    original_dim: int,
    scale: float,
) -> torch.Tensor:
    """Apply inverse randomized Hadamard: x = diag(signs) * H * scale * y.

    Args:
        y: (..., padded_dim) tensor
        signs: (padded_dim,) sign vector
        padded_dim: power-of-2 dimension
        original_dim: original dimension to truncate to
        scale: 1/sqrt(padded_dim)

    Returns:
        (..., padded_dim) tensor (caller truncates if needed)
    """
    orig_shape = y.shape

    # Reshape to 2D
    y_flat = y.reshape(-1, padded_dim).float().contiguous()
    num_rows = y_flat.shape[0]

    out = torch.empty_like(y_flat)
    log2_d = _log2(padded_dim)

    triton_fwht_inverse_kernel[(num_rows,)](
        y_flat, out, signs,
        scale,
        padded_dim,  # stride_row
        D=padded_dim,
        LOG2_D=log2_d,
    )

    new_shape = list(orig_shape)
    new_shape[-1] = padded_dim
    result = out.view(new_shape)

    # Truncate to original dimension
    if original_dim < padded_dim:
        result = result[..., :original_dim]
    return result
