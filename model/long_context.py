# File: model/long_context.py
"""
RoPE Context Extension for LMR-Long

Parameter-neutral long-context via RoPE frequency scaling.
Drop-in replacement for precompute_freqs_cis in architecture.py.

Methods:
  - "ntk"    : NTK-aware scaling — RECOMMENDED for RNA
               Preserves high-freq (stacking/WC) while extending
               low-freq (distant stems). Used by Code Llama.
  - "linear" : Position Interpolation (Chen et al. 2023)
               Compresses all frequencies uniformly.
  - "yarn"   : YaRN (Peng et al. 2023)
               Hybrid — selective per-dimension interpolation.
               Best quality but needs ~400 adaptation steps.

All methods return (freqs_cos, freqs_sin) in the same format as
precompute_freqs_cis, using real cos/sin (no ComplexFloat) for
NCCL compatibility.

Usage:
    from model.long_context import precompute_freqs_scaled, upgrade_backbone

    # Option A: Direct replacement
    freqs_cos, freqs_sin = precompute_freqs_scaled(
        dim=64, max_seq_len=4096, theta=10000.0,
        scaling_type="ntk", factor=2.0, original_max_seq_len=2048,
    )

    # Option B: In-place upgrade of existing model
    upgrade_backbone(model, scaling_type="ntk", factor=2.0,
                     original_max_seq_len=2048)
"""

import torch
import math
from typing import Tuple, Optional


# =============================================================================
# Scaled Frequency Computation
# =============================================================================

def precompute_freqs_scaled(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    scaling_type: str = "none",
    factor: float = 2.0,
    original_max_seq_len: int = 2048,
    # YaRN-specific
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE cos/sin tables with optional context extension.

    Drop-in replacement for architecture.precompute_freqs_cis.
    Identical output format: (freqs_cos, freqs_sin) each [max_seq_len, dim//2].

    Args:
        dim:                  Head dimension (d_model // n_heads)
        max_seq_len:          Target context length (e.g. 4096)
        theta:                RoPE base frequency (default 10000)
        scaling_type:         "none" | "ntk" | "linear" | "yarn"
        factor:               Context extension factor (target / original)
        original_max_seq_len: Length the checkpoint was trained at
        beta_fast:            (YaRN) high-freq boundary
        beta_slow:            (YaRN) low-freq boundary

    Returns:
        freqs_cos: [max_seq_len, dim//2]
        freqs_sin: [max_seq_len, dim//2]
    """
    if scaling_type == "none":
        return _freqs_standard(dim, max_seq_len, theta)
    elif scaling_type == "ntk":
        return _freqs_ntk(dim, max_seq_len, theta, factor)
    elif scaling_type == "linear":
        return _freqs_linear(dim, max_seq_len, theta, factor)
    elif scaling_type == "yarn":
        return _freqs_yarn(dim, max_seq_len, theta, factor,
                           original_max_seq_len, beta_fast, beta_slow)
    else:
        raise ValueError(f"Unknown scaling_type: {scaling_type!r}. "
                         f"Use 'none', 'ntk', 'linear', or 'yarn'.")


def _freqs_standard(
    dim: int, max_seq_len: int, theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE — identical to architecture.precompute_freqs_cis."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def _freqs_ntk(
    dim: int, max_seq_len: int, theta: float, factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NTK-aware RoPE scaling.

    Scales base theta:  θ' = θ · factor^(dim / (dim - 2))

    This spreads the frequency spectrum outward, preserving high-frequency
    components (short-range attention, base stacking) while extending
    low-frequency components (long-range stem pairing).

    For RNA this is ideal: WC-adjacent signals (~1-5 nt) stay intact,
    while distant stem contacts (100-4000 nt) become representable.
    """
    theta_scaled = theta * (factor ** (dim / (dim - 2)))
    freqs = 1.0 / (theta_scaled ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def _freqs_linear(
    dim: int, max_seq_len: int, theta: float, factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Position Interpolation (PI).

    Rescales positions: p → p / factor.
    All frequencies stay within the originally-trained range.

    Simple and stable but compresses ALL bands equally — blurs
    short-range signals that RNA structure relies on.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32) / factor
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def _freqs_yarn(
    dim: int, max_seq_len: int, theta: float, factor: float,
    original_max_seq_len: int, beta_fast: float, beta_slow: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    YaRN: Yet another RoPE extensioN.

    Hybrid: preserve high frequencies (NTK-style), interpolate low
    frequencies (PI-style), with smooth ramp between.

    Per-dimension ramp:
        wavelength_d = 2π / freq_d
        ramp(d) = clamp((wavelength_d - low_bound) / (high_bound - low_bound), 0, 1)
        effective_factor(d) = 1 - ramp(d) + ramp(d) / factor

    High-freq dims (ramp≈0): no scaling → preserves short-range.
    Low-freq dims  (ramp≈1): full interpolation → extends range.
    """
    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    wavelengths = 2 * math.pi / freqs_base

    low_bound = original_max_seq_len / beta_fast
    high_bound = original_max_seq_len / beta_slow

    ramp = ((wavelengths - low_bound) / (high_bound - low_bound)).clamp(0.0, 1.0)

    # Blend: high-freq gets factor=1 (preserve), low-freq gets 1/factor (interpolate)
    per_dim_factor = (1.0 - ramp) + ramp * (1.0 / factor)

    # NTK-aware base + per-dimension interpolation
    theta_scaled = theta * (factor ** (dim / (dim - 2)))
    freqs = 1.0 / (theta_scaled ** (torch.arange(0, dim, 2).float() / dim))
    freqs = freqs * per_dim_factor

    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


# =============================================================================
# In-Place Model Upgrade
# =============================================================================

def upgrade_backbone(
    model,
    scaling_type: str = "ntk",
    factor: float = 2.0,
    original_max_seq_len: int = 2048,
    new_max_seq_len: Optional[int] = None,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> None:
    """
    In-place upgrade of an LMR backbone's RoPE tables for long context.

    Replaces freqs_cos / freqs_sin buffers (persistent=False, so NOT in
    state_dict) in every MultiHeadAttention layer. Zero parameter change.

    Args:
        model:                LMR model (or DDP-unwrapped module)
        scaling_type:         "ntk" | "linear" | "yarn"
        factor:               Extension factor (default 2.0 for 2048→4096)
        original_max_seq_len: Length the checkpoint was trained at
        new_max_seq_len:      Target length (inferred from model if None)
        beta_fast/beta_slow:  YaRN boundaries
    """
    # Unwrap DDP if needed
    raw = model.module if hasattr(model, 'module') else model

    # Infer new max_seq_len from model config
    if new_max_seq_len is None:
        new_max_seq_len = raw.config.max_seq_len

    # Auto-compute factor if not explicitly set
    if factor is None:
        factor = new_max_seq_len / original_max_seq_len

    upgraded = 0
    for layer in raw.layers:
        attn = layer.attention
        head_dim = attn.head_dim
        device = attn.freqs_cos.device

        freqs_cos, freqs_sin = precompute_freqs_scaled(
            dim=head_dim,
            max_seq_len=new_max_seq_len,
            theta=raw.config.rope_theta,
            scaling_type=scaling_type,
            factor=factor,
            original_max_seq_len=original_max_seq_len,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )

        attn.freqs_cos = freqs_cos.to(device)
        attn.freqs_sin = freqs_sin.to(device)
        upgraded += 1

    n_params = raw.get_num_params()
    print(f"✓ RoPE upgraded ({scaling_type}, factor={factor:.1f}):")
    print(f"  {original_max_seq_len} → {new_max_seq_len} tokens")
    print(f"  {upgraded} attention layers updated")
    print(f"  Parameters: {n_params:,} (unchanged)")


# =============================================================================
# Diagnostic
# =============================================================================

def validate_scaling(
    dim: int = 64,
    original_len: int = 2048,
    extended_len: int = 4096,
    theta: float = 10000.0,
    scaling_type: str = "ntk",
    factor: float = 2.0,
) -> dict:
    """
    Compare frequency spectra before/after scaling.

    Use to verify high-freq preservation and low-freq extension.
    """
    freqs_orig = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    cos_ext, sin_ext = precompute_freqs_scaled(
        dim, extended_len, theta,
        scaling_type=scaling_type, factor=factor,
        original_max_seq_len=original_len,
    )

    if scaling_type == "ntk":
        theta_s = theta * (factor ** (dim / (dim - 2)))
        freqs_ext = 1.0 / (theta_s ** (torch.arange(0, dim, 2).float() / dim))
    else:
        freqs_ext = freqs_orig

    n = dim // 4  # quarter of frequency bands
    hi = (freqs_ext[:n] / freqs_orig[:n]).mean().item()
    lo = (freqs_ext[-n:] / freqs_orig[-n:]).mean().item()

    return {
        "scaling_type": scaling_type,
        "factor": factor,
        "context": f"{original_len} → {extended_len}",
        "high_freq_ratio": round(hi, 4),
        "low_freq_ratio": round(lo, 4),
        "cos_shape": tuple(cos_ext.shape),
        "param_change": 0,
    }
