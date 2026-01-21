"""
Tensor Core Emulation in PyTorch - Bit-Exact Implementation

This module provides bit-exact tensor core models for NVIDIA GPUs,
reimplemented from the MATLAB code in:
https://github.com/north-numerical-computing/MATLAB-tensor-core

Reference paper:
F. A. Khattak and M. Mikaitis, "Accurate Models of NVIDIA Tensor Cores"
arXiv:2512.07004 [cs.MS]. Dec. 2025.
"""

from dataclasses import dataclass
from typing import Optional

import torch


# Constants for IEEE 754 single precision
FP32_EXP_BITS = 8
FP32_MAN_BITS = 23
FP32_BIAS = 127
FP32_IMPLICIT_BIT = 1 << 23  # 0x800000


@dataclass
class FPFormat:
    """Floating-point format specification."""

    total_bits: int
    exp_bits: int
    man_bits: int
    has_implicit: bool = True

    @property
    def bias(self) -> int:
        """Exponent bias (IEEE rule: 2^(k-1)-1)."""
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def emin(self) -> int:
        """Minimum exponent (unbiased)."""
        return 1 - self.bias

    @property
    def emax(self) -> int:
        """Maximum exponent (unbiased)."""
        return self.bias


def get_fp_format(format_name: str) -> FPFormat:
    """Get floating-point format info by name."""
    format_name = format_name.lower()

    formats = {
        # 64-bit formats
        "binary64": FPFormat(64, 11, 52),
        "float64": FPFormat(64, 11, 52),
        "double": FPFormat(64, 11, 52),
        "fp64": FPFormat(64, 11, 52),
        # 32-bit formats
        "binary32": FPFormat(32, 8, 23),
        "float32": FPFormat(32, 8, 23),
        "single": FPFormat(32, 8, 23),
        "fp32": FPFormat(32, 8, 23),
        # 16-bit formats
        "binary16": FPFormat(16, 5, 10),
        "float16": FPFormat(16, 5, 10),
        "half": FPFormat(16, 5, 10),
        "fp16": FPFormat(16, 5, 10),
        # bfloat16
        "bfloat16": FPFormat(16, 8, 7),
        "bf16": FPFormat(16, 8, 7),
        # TensorFloat32
        "tensorfloat32": FPFormat(19, 8, 10),
        "tf32": FPFormat(19, 8, 10),
        # FP8 formats
        "fp8-e4m3": FPFormat(8, 4, 3),
        "e4m3": FPFormat(8, 4, 3),
        "fp8-e5m2": FPFormat(8, 5, 2),
        "e5m2": FPFormat(8, 5, 2),
    }

    if format_name not in formats:
        raise ValueError(f"Unknown format: {format_name}")

    return formats[format_name]


def float32_to_bits(x: torch.Tensor) -> torch.Tensor:
    """Convert float32 tensor to its IEEE 754 bit representation as int32 tensor."""
    return x.to(torch.float32).view(torch.int32)


def bits_to_float32(bits: torch.Tensor) -> torch.Tensor:
    """Convert IEEE 754 bit representation (int32) to float32 tensor."""
    return (bits & 0xFFFFFFFF).to(torch.int32).view(torch.float32)


def round_to_format(x: torch.Tensor, format_name: str) -> torch.Tensor:
    """
    Round tensor to specified floating-point format.
    Uses PyTorch's built-in types where available.
    """
    format_name = format_name.lower()

    if format_name in ("binary16", "float16", "half", "fp16"):
        return x.to(torch.float16).to(x.dtype)
    elif format_name in ("bfloat16", "bf16"):
        return x.to(torch.bfloat16).to(x.dtype)
    elif format_name in ("binary32", "float32", "single", "fp32"):
        return x.to(torch.float32).to(x.dtype)
    elif format_name in ("binary64", "float64", "double", "fp64"):
        return x.to(torch.float64).to(x.dtype)
    elif format_name in ("tensorfloat32", "tf32"):
        # TF32 has 10 mantissa bits like fp16 but fp32 exponent range
        # Round by truncating the lower 13 bits of fp32 mantissa
        x32 = x.to(torch.float32)
        x_int = x32.view(torch.int32)
        x_int = x_int & 0xFFFFE000
        return x_int.view(torch.float32).to(x.dtype)
    else:
        return x


def _log2_exponent(x: torch.Tensor) -> torch.Tensor:
    """
    Get the exponent of x (like MATLAB's log2 second return value minus 1).
    Vectorized version operating on tensors.
    """
    x = x.to(torch.float32)
    bits = torch.abs(x).view(torch.int32)
    exp_raw = (bits >> 23) & 0xFF

    # Normal case: exp_raw - 127
    result = exp_raw.to(torch.int32) - 127

    # Handle zeros: return -1
    zero_mask = x == 0
    result = torch.where(
        zero_mask, torch.tensor(-1, dtype=torch.int32, device=x.device), result
    )

    # Handle subnormals (exp_raw == 0 but not zero)
    subnormal_mask = (exp_raw == 0) & ~zero_mask
    if torch.any(subnormal_mask):
        frac = bits & 0x7FFFFF
        # Count leading zeros in mantissa (23-bit field)
        # Use integer log2: floor(log2(frac)) gives position of highest bit
        # For a 23-bit mantissa, leading zeros = 22 - floor(log2(frac))
        # Exponent for subnormal = -126 - leading_zeros - 1
        frac_clamped = torch.clamp(frac, min=1)  # Avoid log2(0)
        highest_bit_pos = torch.floor(torch.log2(frac_clamped.to(torch.float32))).to(
            torch.int32
        )
        leading_zeros = 22 - highest_bit_pos
        subnormal_exp = -126 - leading_zeros - 1
        result = torch.where(subnormal_mask, subnormal_exp, result)

        # Special case: frac == 0 means the value is exactly zero (already handled above)
        # but if somehow frac is 0 with exp_raw == 0, return -126
        frac_zero_mask = subnormal_mask & (frac == 0)
        result = torch.where(
            frac_zero_mask,
            torch.tensor(-126, dtype=torch.int32, device=x.device),
            result,
        )

    return result


def _int_bit_length(x: torch.Tensor) -> torch.Tensor:
    """Compute bit length of positive integers (position of highest set bit + 1)."""
    # For x > 0: bit_length = floor(log2(x)) + 1
    # For x == 0: bit_length = 0
    x_clamped = torch.clamp(x, min=1)
    bit_len = torch.floor(torch.log2(x_clamped.to(torch.float64))).to(torch.int64) + 1
    return torch.where(
        x == 0, torch.tensor(0, dtype=torch.int64, device=x.device), bit_len
    )


def _fpbits_ieee2(
    prod_sig: torch.Tensor,
    prod_exp: torch.Tensor,
    c: torch.Tensor,
    neab: torch.Tensor,
    stkbit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched IEEE-754 component extraction and significand alignment.

    Args:
        prod_sig: (batch, nfma) Product significands
        prod_exp: (batch, nfma) Product exponents (int32)
        c: (batch,) Accumulator values
        neab: (batch,) Extra alignment bits per batch element
        stkbit: Sticky bit enabled flag

    Returns:
        max_exp_unbiased: (batch,) Maximum exponent per batch
        aligned_sigs: (batch, nfma+1) Aligned significands including c
        valid_mask: (batch, nfma+1) Mask for valid (non-zero) entries
    """
    device = prod_sig.device
    batch_size, nfma = prod_sig.shape

    # Convert product significands to IEEE representation
    sig_f32 = prod_sig.to(torch.float32)
    bits = sig_f32.view(torch.int32).to(torch.int64)

    exp_raw = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    # Add implicit bit for normal numbers (exp_raw != 0)
    full_sigs = torch.where(exp_raw != 0, frac + FP32_IMPLICIT_BIT, frac)

    # Compute unbiased exponent: raw_exp - 127 + original_prod_exp
    exp_unbiased = exp_raw - 127 + prod_exp.to(torch.int64)

    # Handle accumulator c - extract IEEE components
    c_f32 = c.to(torch.float32)
    c_bits = c_f32.view(torch.int32).to(torch.int64)

    c_exp_raw = (c_bits >> 23) & 0xFF
    c_frac = c_bits & 0x7FFFFF

    c_full_sig = torch.where(c_exp_raw != 0, c_frac + FP32_IMPLICIT_BIT, c_frac)
    c_exp = c_exp_raw - 127
    # Handle subnormal c (exp_raw == 0 means exp = -126)
    c_exp = torch.where(
        c_exp == -127, torch.tensor(-126, dtype=torch.int64, device=device), c_exp
    )

    # Concatenate c to the arrays: (batch, nfma) -> (batch, nfma+1)
    exp_unbiased = torch.cat([exp_unbiased, c_exp.unsqueeze(1)], dim=1)
    full_sigs = torch.cat([full_sigs, c_full_sig.unsqueeze(1)], dim=1)

    # Create validity mask (non-zero significands contribute)
    # Products are valid if prod_sig != 0, c is valid if c != 0
    prod_valid = prod_sig != 0
    c_valid = (c != 0).unsqueeze(1)
    valid_mask = torch.cat([prod_valid, c_valid], dim=1)

    # For invalid entries, set exp to very negative so they don't affect max
    VERY_NEG = torch.tensor(-1000, dtype=torch.int64, device=device)
    exp_for_max = torch.where(valid_mask, exp_unbiased, VERY_NEG)

    # Find maximum exponent per batch (along nfma+1 dimension)
    max_exp_unbiased = torch.max(exp_for_max, dim=1).values

    # Compute shifts per element
    shifts = (max_exp_unbiased.unsqueeze(1) - exp_unbiased).to(torch.int64)

    # Apply neab (extra alignment bits) - neab is per-batch
    neab_expanded = neab.unsqueeze(1).to(torch.int64)
    shifted_sigs = torch.where(
        neab_expanded >= 0, full_sigs << neab_expanded, full_sigs >> (-neab_expanded)
    )

    if stkbit:
        shift_clamped = torch.clamp(shifts, min=0, max=63)
        lost_mask = (
            torch.tensor(1, dtype=torch.int64, device=device) << shift_clamped
        ) - 1
        lost_mask = torch.where(
            shifts > 0, lost_mask, torch.tensor(0, dtype=torch.int64, device=device)
        )
        lost_bits = torch.where(
            (shifted_sigs & lost_mask) != 0,
            torch.tensor(1, dtype=torch.int64, device=device),
            torch.tensor(0, dtype=torch.int64, device=device),
        )
        aligned_sigs = torch.where(
            shifts > 0,
            (shifted_sigs >> shift_clamped) * 2 + lost_bits,
            shifted_sigs * 2,
        )
    else:
        shift_clamped = torch.clamp(shifts, min=0, max=63)
        aligned_sigs = shifted_sigs >> shift_clamped

    return max_exp_unbiased, aligned_sigs, valid_mask


def generic_bfma_tc(
    no_exp_bits_prd: int,
    no_man_bits_prd: int,
    out_round_mode: str,
    neab: int,
    stk_bit_enabled: int,
    no_man_bits_out: int,
    no_exp_bits_out: int,
    a_block: torch.Tensor,
    b_block: torch.Tensor,
    c: torch.Tensor,
    no_exp_bits_in: int,
) -> torch.Tensor:
    """
    Batched Generic Block FMA for Tensor Core emulation.

    Args:
        a_block: (batch, nfma) or (M, N, nfma) input A values
        b_block: (batch, nfma) or (M, N, nfma) input B values
        c: (batch,) or (M, N) accumulator values
        Other args: same as non-batched version

    Returns:
        (batch,) or (M, N) output values
    """
    device = a_block.device
    original_shape = a_block.shape[:-1]  # (batch,) or (M, N)
    nfma = a_block.shape[-1]

    # Flatten batch dimensions
    a_flat = a_block.reshape(-1, nfma).to(torch.float32)
    b_flat = b_block.reshape(-1, nfma).to(torch.float32)
    c_flat = c.reshape(-1).to(torch.float32)
    batch_size = a_flat.shape[0]

    # Compute products
    r2 = a_flat * b_flat  # (batch, nfma)

    # Constants
    emin_output = 1 - (1 << (no_exp_bits_out - 1)) + 1
    emin_input = 1 - (1 << (no_exp_bits_in - 1)) + 1
    emin_product = 1 - (1 << (no_exp_bits_prd - 1)) + 1

    # Check for all-zero products per batch
    any_nonzero = torch.any(r2 != 0, dim=1)

    # Extract exponents using vectorized log2
    a_exp = _log2_exponent(a_flat)  # (batch, nfma)
    b_exp = _log2_exponent(b_flat)

    # Compute special case (spc) for denormalized product handling
    # MATLAB: if abs(c) > abs(r2_nz) then special_case=0 else special_case=1
    c_abs = torch.abs(c_flat)
    r2_abs_max = torch.max(torch.abs(r2), dim=1).values
    special_case = torch.where(
        (c_abs > r2_abs_max) & any_nonzero,
        torch.tensor(0, dtype=torch.int32, device=device),
        torch.tensor(1, dtype=torch.int32, device=device),
    )

    # Handle special case for denormalized products
    spc = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Clamp exponents for spc calculation
    a_exp_u = torch.maximum(
        a_exp, torch.tensor(emin_input, dtype=torch.int32, device=device)
    )
    b_exp_u = torch.maximum(
        b_exp, torch.tensor(emin_input, dtype=torch.int32, device=device)
    )
    prod_exp_check = a_exp_u + b_exp_u

    # Create mask for nonzero products - exclude zeros from max calculation
    nonzero_prod_mask = r2 != 0
    VERY_NEG_EXP = torch.tensor(-10000, dtype=torch.int32, device=device)
    prod_exp_for_max = torch.where(nonzero_prod_mask, prod_exp_check, VERY_NEG_EXP)

    # Max product exponent per batch (only considering nonzero products)
    max_prod_exp = torch.max(prod_exp_for_max, dim=1).values

    # Compute max significand at max exponent positions (only nonzero products)
    max_exp_mask = (prod_exp_check == max_prod_exp.unsqueeze(1)) & nonzero_prod_mask
    r2_at_max = torch.where(
        max_exp_mask,
        torch.abs(r2),
        torch.tensor(0.0, dtype=torch.float32, device=device),
    )
    prod_sig_at_max = r2_at_max / torch.pow(
        torch.tensor(2.0, dtype=torch.float64, device=device),
        max_prod_exp.unsqueeze(1).to(torch.float64),
    )
    max_prod_sig = torch.max(prod_sig_at_max, dim=1).values

    c_exp = _log2_exponent(c_abs.unsqueeze(1)).squeeze(1)

    # spc condition
    spc_condition = (
        (special_case == 1)
        & (
            max_prod_exp
            >= torch.maximum(
                c_exp, torch.tensor(emin_product, dtype=torch.int32, device=device)
            )
        )
        & (max_prod_sig >= 2.0)
    )
    spc = torch.where(
        spc_condition, torch.tensor(1, dtype=torch.int32, device=device), spc
    )

    # Compute product exponents and significands
    prod_exp = a_exp + b_exp  # (batch, nfma)

    # Compute significands: a / 2^a_exp, b / 2^b_exp
    a_sig = a_flat.to(torch.float64) / torch.pow(
        torch.tensor(2.0, dtype=torch.float64, device=device), a_exp.to(torch.float64)
    )
    b_sig = b_flat.to(torch.float64) / torch.pow(
        torch.tensor(2.0, dtype=torch.float64, device=device), b_exp.to(torch.float64)
    )
    prod_sig = a_sig * b_sig  # (batch, nfma)

    # Sign bits
    sign_prods = (prod_sig < 0).to(torch.int64)  # (batch, nfma)
    sign_c = (c_flat < 0).to(torch.int64)  # (batch,)
    sign_bits = torch.cat([sign_prods, sign_c.unsqueeze(1)], dim=1)  # (batch, nfma+1)

    prod_sig_abs = torch.abs(prod_sig).to(torch.float32)

    # Apply special case adjustment
    neab_adjusted = neab + spc  # (batch,)

    # Alignment and accumulation
    max_exp_unbiased, aligned_sigs, valid_mask = _fpbits_ieee2(
        prod_sig_abs, prod_exp, c_flat, neab_adjusted, stk_bit_enabled
    )

    # Accumulate with signs, only for valid entries
    signed_sigs = torch.where(sign_bits == 1, -aligned_sigs, aligned_sigs)
    signed_sigs = torch.where(
        valid_mask, signed_sigs, torch.tensor(0, dtype=torch.int64, device=device)
    )
    sum_unnormalized = torch.sum(signed_sigs, dim=1)  # (batch,)

    # Output sign
    s_out = (sum_unnormalized < 0).to(torch.int64)
    sum_abs = torch.abs(sum_unnormalized)

    # Find bit length for normalization
    bit_len = _int_bit_length(sum_abs)

    # Expected bit length after neab and stkbit shifts
    expected_bits = no_man_bits_prd + neab_adjusted + stk_bit_enabled + 1

    # Compute exponent adjustment
    total_exp = bit_len - expected_bits
    d_exp = max_exp_unbiased + total_exp

    # Normalize: shift to target bit width
    extra_bits = 3  # guard, round, sticky
    target_bits = no_man_bits_out + 1 + extra_bits

    shift_to_normalize = bit_len - target_bits

    # Handle shift right (positive shift)
    shift_amt = torch.clamp(shift_to_normalize, min=0, max=63)
    lost_mask_right = (
        torch.tensor(1, dtype=torch.int64, device=device) << shift_amt
    ) - 1
    sticky_right = torch.where(
        (sum_abs & lost_mask_right) != 0,
        torch.tensor(1, dtype=torch.int64, device=device),
        torch.tensor(0, dtype=torch.int64, device=device),
    )
    mantissa_right = (sum_abs >> shift_amt) | sticky_right

    # Handle shift left (negative shift)
    shift_left = torch.clamp(-shift_to_normalize, min=0, max=63)
    mantissa_left = sum_abs << shift_left

    # Select based on shift direction
    mantissa = torch.where(
        shift_to_normalize > 0,
        mantissa_right,
        torch.where(shift_to_normalize < 0, mantissa_left, sum_abs),
    )

    # Handle subnormal output
    subnormal_shift = torch.clamp(emin_output - d_exp, min=0, max=63).to(torch.int64)
    lost_mask_sub = (
        torch.tensor(1, dtype=torch.int64, device=device) << subnormal_shift
    ) - 1
    sticky_sub = torch.where(
        (mantissa & lost_mask_sub) != 0,
        torch.tensor(1, dtype=torch.int64, device=device),
        torch.tensor(0, dtype=torch.int64, device=device),
    )
    mantissa_subnormal = torch.where(
        subnormal_shift > 0, (mantissa >> subnormal_shift) | sticky_sub, mantissa
    )
    mantissa = torch.where(d_exp < emin_output, mantissa_subnormal, mantissa)
    d_exp = torch.where(
        d_exp < emin_output,
        torch.tensor(emin_output, dtype=torch.int64, device=device),
        d_exp,
    )

    # Apply rounding
    if out_round_mode != "rz":
        guard_bit = (mantissa >> 2) & 1
        round_bit = (mantissa >> 1) & 1
        sticky_bit = mantissa & 1
        grs = guard_bit * 4 + round_bit * 2 + sticky_bit
        lsb = (mantissa >> 3) & 1

        if out_round_mode == "rne":
            round_up = (grs > 4) | ((grs == 4) & (lsb == 1))
        elif out_round_mode == "rd":
            round_up = (s_out == 1) & (grs > 0)
        elif out_round_mode == "ru":
            round_up = (s_out == 0) & (grs > 0)
        else:
            round_up = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Apply rounding
        mantissa_rounded = mantissa + 8  # Add ULP
        new_bit_len = _int_bit_length(mantissa_rounded)
        overflow = new_bit_len > target_bits
        mantissa_overflow = mantissa_rounded >> 1
        d_exp_overflow = d_exp + 1

        mantissa = torch.where(
            round_up,
            torch.where(overflow, mantissa_overflow, mantissa_rounded),
            mantissa,
        )
        d_exp = torch.where(round_up & overflow, d_exp_overflow, d_exp)

    # Truncate to final mantissa (remove GRS bits)
    final_mantissa = mantissa >> 3

    # Extract implicit bit and fraction
    implicit_bit = (final_mantissa >> no_man_bits_out) & 1
    frac_bits = final_mantissa & ((1 << no_man_bits_out) - 1)

    # Build float value
    frac_value = frac_bits.to(torch.float64) / (1 << no_man_bits_out)
    d = (implicit_bit.to(torch.float64) + frac_value) * torch.pow(
        torch.tensor(2.0, dtype=torch.float64, device=device), d_exp.to(torch.float64)
    )

    # Check for overflow
    bias = (1 << (no_exp_bits_out - 1)) - 1
    biased_exp = d_exp + bias
    d = torch.where(
        biased_exp >= (1 << no_exp_bits_out) - 1,
        torch.tensor(float("inf"), dtype=torch.float64, device=device),
        d,
    )

    # Apply sign
    d = torch.where(s_out == 1, -d, d)

    # Handle all-zero case
    d = torch.where(
        any_nonzero | (c_flat != 0),
        d,
        torch.where(
            c_flat != 0,
            c_flat.to(torch.float64),
            torch.tensor(0.0, dtype=torch.float64, device=device),
        ),
    )
    d = torch.where(
        sum_unnormalized == 0, torch.tensor(0.0, dtype=torch.float64, device=device), d
    )

    return d.to(torch.float32).reshape(original_shape)


def gemm(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: Optional[torch.Tensor],
    informat: str,
    outformat: str,
    params: dict,
) -> torch.Tensor:
    """
    Vectorized General Matrix Multiply using tensor core model.

    Computes D = alpha * A @ B + beta * C

    This implementation processes all M x N output elements in parallel
    using batched tensor operations.
    """
    device = A.device
    dtype = A.dtype

    M, K1 = A.shape
    K2, N = B.shape

    if K1 != K2:
        raise ValueError(
            f"Matrix dimensions incompatible: A is {M}x{K1}, B is {K2}x{N}"
        )

    K = K1

    # Initialize C if needed
    if C is None or (isinstance(C, (int, float)) and C == 0):
        C = torch.zeros(M, N, dtype=dtype, device=device)

    # Round inputs to their formats
    A_rounded = round_to_format(alpha * A, informat)
    B_rounded = round_to_format(B, informat)
    C_rounded = round_to_format(beta * C, outformat)

    # Get format info
    in_fmt = get_fp_format(informat)
    out_fmt = get_fp_format(outformat)

    # Extract parameters
    nfma = params["fma"]
    neab = params["neab"]
    out_round_mode = params["frmode"].lower()
    stk_bit_enabled = params.get("stkbitenabled", 0)
    inter_pattern = params.get("inter_pattern", 0)

    # Product format (assumed single precision)
    no_man_bits_prd = 23
    no_exp_bits_prd = 8

    # Handle interleaved pattern
    c_out_round_mode = "rne"

    if inter_pattern:
        nfma = 2 * nfma

    # Handle negative neab (Ada/L40S special case)
    no_man_bits_out = out_fmt.man_bits
    if neab < 0 and no_man_bits_out == no_man_bits_prd:
        no_man_bits_out = no_man_bits_out + neab

    # Pad K to multiple of nfma
    remainder = K % nfma
    pad_size = (nfma - remainder) if remainder != 0 else 0
    padded_K = K + pad_size
    num_blocks = padded_K // nfma

    # Pad A and B if needed
    if pad_size > 0:
        A_padded = torch.cat(
            [A_rounded, torch.zeros(M, pad_size, dtype=dtype, device=device)], dim=1
        )
        B_padded = torch.cat(
            [B_rounded, torch.zeros(pad_size, N, dtype=dtype, device=device)], dim=0
        )
    else:
        A_padded = A_rounded
        B_padded = B_rounded

    # Reshape for block processing
    # A: (M, K) -> (M, num_blocks, nfma)
    # B: (K, N) -> (num_blocks, nfma, N)
    A_blocks = A_padded.reshape(M, num_blocks, nfma)
    B_blocks = B_padded.reshape(num_blocks, nfma, N)

    # Handle special values (NaN/Inf) - compute full products for checking
    # products[m, n, k] = sum of A[m, k_block] * B[k_block, n] for elements in block
    products_check = torch.einsum("mki,kin->mn", A_blocks, B_blocks)
    combined_check = products_check + C_rounded

    # Create masks for special cases
    has_nan = (
        torch.isnan(combined_check)
        | torch.isnan(A_rounded).any(dim=1, keepdim=True)
        | torch.isnan(B_rounded).any(dim=0, keepdim=True)
    )
    has_pos_inf = (
        (combined_check == float("inf"))
        | torch.any(A_rounded == float("inf"), dim=1, keepdim=True)
        | torch.any(B_rounded == float("inf"), dim=0, keepdim=True)
    )
    has_neg_inf = (
        (combined_check == float("-inf"))
        | torch.any(A_rounded == float("-inf"), dim=1, keepdim=True)
        | torch.any(B_rounded == float("-inf"), dim=0, keepdim=True)
    )

    # Initialize accumulator
    D = C_rounded.clone()

    if not inter_pattern:
        # Standard pattern - process all blocks sequentially, but all M x N elements in parallel
        for k in range(num_blocks):
            # Get block k for all output elements
            # a_block: (M, nfma) - row m uses A_blocks[m, k, :]
            # b_block: (N, nfma) - col n uses B_blocks[k, :, n]
            a_block_k = A_blocks[:, k, :]  # (M, nfma)
            b_block_k = B_blocks[k, :, :].T  # (N, nfma)

            # Broadcast to get (M, N, nfma) products
            # For D[m, n], we need A[m, k, :] * B[k, :, n]
            a_expanded = a_block_k.unsqueeze(1).expand(M, N, nfma)  # (M, N, nfma)
            b_expanded = b_block_k.unsqueeze(0).expand(M, N, nfma)  # (M, N, nfma)

            # Call batched FMA
            D = generic_bfma_tc(
                no_exp_bits_prd,
                no_man_bits_prd,
                out_round_mode,
                neab,
                stk_bit_enabled,
                no_man_bits_out,
                out_fmt.exp_bits,
                a_expanded,
                b_expanded,
                D,
                in_fmt.exp_bits,
            )
    else:
        # Interleaved pattern (H100/B200 FP8)
        # Generate interleaved sequences as tensors
        seq = torch.arange(nfma, device=device)
        seq1 = seq[(seq % 4) < 2]
        seq2_candidates = seq1 + 2
        seq2 = seq2_candidates[seq2_candidates < nfma]

        zero_acc = torch.zeros(M, N, dtype=dtype, device=device)

        for k in range(num_blocks):
            a_block_k = A_blocks[:, k, :]  # (M, nfma)
            b_block_k = B_blocks[k, :, :].T  # (N, nfma)

            a_expanded = a_block_k.unsqueeze(1).expand(M, N, nfma)
            b_expanded = b_block_k.unsqueeze(0).expand(M, N, nfma)

            # First interleaved FMA (seq1 elements)
            a_block_1 = a_expanded[..., seq1]  # (M, N, len(seq1))
            b_block_1 = b_expanded[..., seq1]

            d1 = generic_bfma_tc(
                no_exp_bits_prd,
                no_man_bits_prd,
                out_round_mode,
                neab,
                stk_bit_enabled,
                no_man_bits_out,
                out_fmt.exp_bits,
                a_block_1,
                b_block_1,
                zero_acc,
                in_fmt.exp_bits,
            )

            # Second interleaved FMA (seq2 elements)
            a_block_2 = a_expanded[..., seq2]
            b_block_2 = b_expanded[..., seq2]

            d2 = generic_bfma_tc(
                no_exp_bits_prd,
                no_man_bits_prd,
                out_round_mode,
                neab,
                stk_bit_enabled,
                no_man_bits_out,
                out_fmt.exp_bits,
                a_block_2,
                b_block_2,
                d1,
                in_fmt.exp_bits,
            )

            # Final addition with accumulator
            ones = torch.ones(M, N, 1, dtype=dtype, device=device)
            d2_expanded = d2.unsqueeze(-1)

            D = generic_bfma_tc(
                out_fmt.exp_bits,
                no_man_bits_out,
                c_out_round_mode,
                2,
                1,
                no_man_bits_out,
                out_fmt.exp_bits,
                d2_expanded,
                ones,
                D,
                in_fmt.exp_bits,
            )

    # Apply special value handling
    D = torch.where(has_nan, torch.tensor(float("nan"), dtype=dtype, device=device), D)
    D = torch.where(
        ~has_nan & has_pos_inf & has_neg_inf,
        torch.tensor(float("nan"), dtype=dtype, device=device),
        D,
    )
    D = torch.where(
        ~has_nan & has_pos_inf & ~has_neg_inf,
        torch.tensor(float("inf"), dtype=dtype, device=device),
        D,
    )
    D = torch.where(
        ~has_nan & ~has_pos_inf & has_neg_inf,
        torch.tensor(float("-inf"), dtype=dtype, device=device),
        D,
    )

    return D


# GPU-specific model implementations


def V100TC(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: Optional[torch.Tensor],
    outformat: str = "binary32",
) -> torch.Tensor:
    """V100 Tensor Core model."""
    params = {
        "fma": 4,
        "neab": 0,
        "frmode": "rz",
        "stkbitenabled": 0,
        "inter_pattern": 0,
    }

    outformat = outformat.lower()
    if outformat in ("fp16", "binary16", "half"):
        params["frmode"] = "rne"

    return gemm(alpha, A, B, beta, C, "binary16", outformat, params)


def A100TC(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: Optional[torch.Tensor],
    informat: str = "binary16",
    outformat: str = "binary32",
) -> torch.Tensor:
    """A100 Tensor Core model."""
    params = {
        "fma": 8,
        "neab": 1,  # A100 uses 1 extra alignment bit
        "frmode": "rz",
        "stkbitenabled": 0,
        "inter_pattern": 0,
    }

    informat = informat.lower()
    outformat = outformat.lower()

    if informat in ("fp16", "half", "binary16"):
        params["fma"] = 8
        if outformat in ("fp16", "binary16", "half"):
            params["frmode"] = "rne"
    elif informat in ("tf32", "tensorfloat32"):
        params["fma"] = 4
    elif informat in ("bfloat16", "bf16"):
        params["fma"] = 8

    return gemm(alpha, A, B, beta, C, informat, outformat, params)


def H100TC(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: Optional[torch.Tensor],
    informat: str = "binary16",
    outformat: str = "binary32",
) -> torch.Tensor:
    """H100 Tensor Core model."""
    params = {
        "fma": 16,
        "neab": 2,
        "frmode": "rz",
        "stkbitenabled": 0,
        "inter_pattern": 0,
    }

    informat = informat.lower()
    outformat = outformat.lower()

    if informat in ("fp16", "half", "binary16"):
        if outformat in ("fp16", "binary16", "half"):
            params["frmode"] = "rne"
    elif informat in ("tf32", "tensorfloat32"):
        params["fma"] = 8
    elif informat in ("fp8-e5m2", "fp8-e4m3", "e5m2", "e4m3"):
        params["fma"] = 32
        params["inter_pattern"] = 0
        params["neab"] = -10
        if outformat in ("fp16", "binary16", "half"):
            params["frmode"] = "rne"

    return gemm(alpha, A, B, beta, C, informat, outformat, params)


def B200TC(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: Optional[torch.Tensor],
    informat: str = "binary16",
    outformat: str = "binary32",
) -> torch.Tensor:
    """B200 Tensor Core model."""
    params = {
        "fma": 16,
        "neab": 2,
        "frmode": "rz",
        "stkbitenabled": 0,
        "inter_pattern": 0,
    }

    informat = informat.lower()
    outformat = outformat.lower()

    if informat in ("fp16", "half", "binary16"):
        if outformat in ("fp16", "binary16", "half"):
            params["frmode"] = "rne"
    elif informat in ("tf32", "tensorfloat32"):
        params["fma"] = 8
    elif informat in ("fp8-e5m2", "fp8-e4m3", "e5m2", "e4m3"):
        params["fma"] = 16
        params["inter_pattern"] = 1
        if outformat in ("fp16", "binary16", "half"):
            params["frmode"] = "rne"

    return gemm(alpha, A, B, beta, C, informat, outformat, params)


def H200TC(
    alpha: float,
    A: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    C: Optional[torch.Tensor],
    informat: str = "binary16",
    outformat: str = "binary32",
) -> torch.Tensor:
    """H200 Tensor Core model (same as H100)."""
    return H100TC(alpha, A, B, beta, C, informat, outformat)
