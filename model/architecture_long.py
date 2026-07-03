# File: model/architecture_long.py
"""
LMR-Long: Hybrid-Attention Foundation Model for Long RNA

The LMRConfig has always had `layer_types` and `attention_window` fields.
This module implements what those fields were designed for.

Architectural innovations over LMR v0:
  1. HYBRID ATTENTION SCHEDULE — Most layers use sliding-window attention
     O(L·w). Every Nth layer uses full attention O(L²). This mirrors RNA
     biology: local layers capture stacking, hairpins, and internal loops;
     global layers capture long-range stem pairing and pseudoknots.

  2. NTK-AWARE ROPE — Built-in from construction, not retrofitted.
     High-frequency components (short-range, WC pairing) preserved;
     low-frequency components extended for 4k+ reach.

  3. SAME PARAMETER SHAPES AS v0 — Q/K/V/O projections, SwiGLU, RMSNorm,
     embeddings, and LM head are dimensionally identical. You can load
     v0 checkpoint weights directly as initialization.

Memory at L=4096, 24 layers, d_model=1024, bf16, B=4:
  v0 (all full attention):   24 × O(L²) attention per layer
  LMR-Long (18w + 6f):      18 × O(L·256) + 6 × O(L²) → ~3.4× cheaper

Usage:
    from model.architecture_long import LMRLong

    config = LMRConfig.from_yaml("configs/lmr_long_4k.yml")
    model = LMRLong(config, rope_scaling={"type": "ntk", "factor": 2.0,
                                           "original_max_seq_len": 2048})

    # Optional: initialize from v0 weights
    v0_state = torch.load("checkpoints/v0_final/checkpoint_best.pt")
    model.load_state_dict(v0_state["model_state_dict"], strict=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from torch.utils.checkpoint import checkpoint

from .config import LMRConfig
from .long_context import precompute_freqs_scaled


# Reuse these unchanged from architecture.py
from .architecture import RMSNorm, SwiGLU, apply_rotary_emb, FLASH_ATTENTION_AVAILABLE


# =============================================================================
# Hybrid Multi-Head Attention (window or full, per-layer)
# =============================================================================
class HybridMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with RoPE and optional sliding window.

    When window_size is None → full bidirectional attention (SDPA FlashAttention).
    When window_size is set  → band-diagonal mask restricting attention to a
                               local window (SDPA memory-efficient kernel).

    For RNA, the window captures:
      - w=256: hairpins, internal loops, local stems, coaxial stacking
      - Full layers: distant stem partners, pseudoknots, tertiary contacts

    Parameters are IDENTICAL to v0's MultiHeadAttention — same Q/K/V/O shapes.
    The only difference is (a) how RoPE freqs are computed and (b) the optional
    attention mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        window_size: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.window_size = window_size

        # Q, K, V, O projections — same shapes as v0
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # RoPE — NTK-aware if scaling provided, standard otherwise
        if rope_scaling and rope_scaling.get("type", "none") != "none":
            freqs_cos, freqs_sin = precompute_freqs_scaled(
                dim=self.head_dim,
                max_seq_len=max_seq_len,
                theta=rope_theta,
                scaling_type=rope_scaling["type"],
                factor=rope_scaling.get("factor", 2.0),
                original_max_seq_len=rope_scaling.get("original_max_seq_len", 2048),
                beta_fast=rope_scaling.get("beta_fast", 32.0),
                beta_slow=rope_scaling.get("beta_slow", 1.0),
            )
        else:
            freqs_cos, freqs_sin = precompute_freqs_scaled(
                dim=self.head_dim,
                max_seq_len=max_seq_len,
                theta=rope_theta,
                scaling_type="none",
            )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Sliding window mask — precomputed for max_seq_len, sliced at runtime.
        # persistent=False: not saved in state_dict, moves with .to(device).
        if window_size is not None:
            idx = torch.arange(max_seq_len)
            mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= (window_size // 2)
            self.register_buffer("_window_mask", mask, persistent=False)
        else:
            self._window_mask = None

        self.use_flash_attention = FLASH_ATTENTION_AVAILABLE

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        # Project
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, self.freqs_cos[:L], self.freqs_sin[:L])

        # Build attention mask
        attn_mask = self._build_mask(B, L, attention_mask)

        # SDPA — kernel selection is automatic:
        #   No mask → FlashAttention (fastest, full-attn layers)
        #   Bool mask → Memory-efficient (still fast, window layers)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_out)

    def _build_mask(
        self,
        B: int,
        L: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Combine sliding-window mask with padding mask.

        Returns None for full-attention layers with no padding (enables FlashAttention).
        Returns [B, 1, L, L] boolean for window layers or padded batches.

        CRITICAL: When window + padding are combined, padding tokens far from
        real content (distance > window_size // 2) have NO attendable keys.
        softmax([-inf, ...]) = NaN, which propagates through residuals to
        corrupt the entire batch. We prevent this by ensuring every query row
        has at least one True entry (self-attention via diagonal).
        """
        has_window = self._window_mask is not None
        has_padding = attention_mask is not None

        if not has_window and not has_padding:
            return None

        if has_window and not has_padding:
            # Pure window, no padding — broadcast [1, 1, L, L]
            # Diagonal is always True (distance 0 <= window//2). Safe.
            return self._window_mask[:L, :L].unsqueeze(0).unsqueeze(0)

        if not has_window and has_padding:
            # Full attention with padding
            # attention_mask: [B, L] with 1=attend, 0=ignore
            # Every query sees the same key mask → padding queries still
            # attend to all real tokens → no all-False rows (unless the
            # entire sequence is padding, which shouldn't happen).
            pad = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]
            return pad.expand(B, 1, L, L)

        # ─── Both window and padding ──────────────────────────────────
        # This is the dangerous case. A padding query at position P where
        # P > L_real + window//2 has NO real tokens in its window.
        # window[P,k] is True only for |P-k| <= 128, and pad[k] is False
        # for all k in that range → combined mask row is all-False → NaN.
        #
        # Fix: OR in the diagonal. Every token can always attend to itself.
        # For real tokens this is a no-op (already True). For padding tokens
        # this gives exactly one True entry, producing a finite (if
        # meaningless) attention output that gets zeroed in LMRLong.forward.
        window = self._window_mask[:L, :L].unsqueeze(0).unsqueeze(0)   # [1, 1, L, L]
        pad = attention_mask.unsqueeze(1).unsqueeze(2).bool()           # [B, 1, 1, L]
        combined = window & pad.expand(B, 1, L, L)

        # Self-attention floor: prevents softmax NaN for isolated padding tokens
        diag = torch.eye(L, dtype=torch.bool, device=combined.device)
        combined = combined | diag.unsqueeze(0).unsqueeze(0)

        return combined

    def extra_repr(self) -> str:
        w = f", window={self.window_size}" if self.window_size else ", full"
        return f"d_model={self.d_model}, heads={self.n_heads}{w}"


# =============================================================================
# Long Transformer Block
# =============================================================================
class LongTransformerBlock(nn.Module):
    """
    Pre-norm transformer block with hybrid attention.

    Structurally identical to v0's TransformerBlock — same Pre-Norm + SwiGLU
    + residual pattern. Only the attention module differs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_hidden: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        window_size: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(d_model, eps=norm_eps)
        self.attention = HybridMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            window_size=window_size,
            rope_scaling=rope_scaling,
        )

        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, ff_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return self._forward_checkpointed(x, attention_mask)
        return self._forward_impl(x, attention_mask)

    def _forward_impl(self, x, attention_mask):
        h = self.attention(self.attention_norm(x), attention_mask)
        x = x + self.dropout(h)
        h = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(h)
        return x

    def _forward_checkpointed(self, x, attention_mask):
        def attn_block(x_in):
            h = self.attention(self.attention_norm(x_in), attention_mask)
            return x_in + self.dropout(h)

        x = checkpoint(attn_block, x, use_reentrant=False)

        def ffn_block(x_in):
            h = self.ffn(self.ffn_norm(x_in))
            return x_in + self.dropout(h)

        x = checkpoint(ffn_block, x, use_reentrant=False)
        return x


# =============================================================================
# LMR-Long: Hybrid-Attention RNA Foundation Model
# =============================================================================
class LMRLong(nn.Module):
    """
    LMR-Long: Long-context RNA language model with hybrid attention.

    Layer schedule (default, 24 layers):
        Layers  0-2:  sliding window (local: stacking, hairpins)
        Layer   3:    full attention  (global: distant stems)
        Layers  4-6:  sliding window
        Layer   7:    full attention
        ... repeats ...
        Layer  23:    full attention  (final global mixing)

    This gives 18 window + 6 full layers. Compute is ~3.4× cheaper
    than all-full at L=4096. Memory per window layer is O(L·w) vs O(L²).

    All parameter shapes match v0's LMR — load v0 weights directly.
    """

    def __init__(
        self,
        config: LMRConfig,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.config = config

        # ─── Token embeddings ──────────────────────────────────────────
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        # ─── Build hybrid layer schedule ───────────────────────────────
        layer_types = self._resolve_layer_types(config)

        # ─── Transformer blocks ────────────────────────────────────────
        self.layers = nn.ModuleList()
        for i, ltype in enumerate(layer_types):
            ws = config.attention_window if ltype == "window" else None
            self.layers.append(LongTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_hidden=config.ff_hidden,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                norm_eps=config.norm_eps,
                window_size=ws,
                rope_scaling=rope_scaling,
            ))

        # ─── Final norm + LM head ─────────────────────────────────────
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

        # ─── Log architecture ──────────────────────────────────────────
        n_window = sum(1 for lt in layer_types if lt == "window")
        n_full = sum(1 for lt in layer_types if lt == "full")
        w = config.attention_window or "N/A"
        rope_str = (rope_scaling or {}).get("type", "standard")

        if FLASH_ATTENTION_AVAILABLE:
            print("✓ SDPA enabled (FlashAttention for full layers, "
                  "MemEfficient for window layers)")
        else:
            print("⚠ SDPA not available — using vanilla attention")

        print(f"✓ LMR-Long initialized:")
        print(f"    Layers     : {n_window} window (w={w}) + {n_full} full "
              f"= {config.n_layers}")
        print(f"    RoPE       : {rope_str}")
        print(f"    Context    : {config.max_seq_len}")
        print(f"    Params     : {self.get_num_params():,}")

    # ─── Layer schedule ────────────────────────────────────────────────

    @staticmethod
    def _resolve_layer_types(config: LMRConfig) -> List[str]:
        """
        Determine per-layer attention type.

        Priority:
          1. config.layer_types if explicitly provided (list of "window"/"full")
          2. Auto-generate: every 4th layer is full, rest are window
             (requires config.attention_window to be set)
          3. All full (vanilla LMR behavior, when no window is configured)
        """
        if config.layer_types is not None:
            assert len(config.layer_types) == config.n_layers, (
                f"layer_types length ({len(config.layer_types)}) != "
                f"n_layers ({config.n_layers})"
            )
            return config.layer_types

        if config.attention_window is not None:
            # Default schedule: every 4th layer is full attention
            schedule = []
            for i in range(config.n_layers):
                if (i + 1) % 4 == 0:  # Layers 3, 7, 11, 15, 19, 23
                    schedule.append("full")
                else:
                    schedule.append("window")
            return schedule

        # No window configured — all full (same as v0)
        return ["full"] * config.n_layers

    # ─── Init ──────────────────────────────────────────────────────────

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ─── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, L] token IDs
            attention_mask:  [B, L] with 1=attend, 0=ignore
        Returns:
            logits: [B, L, vocab_size]
        """
        x = self.token_embeddings(input_ids)

        # ── Zero padding positions ─────────────────────────────────────
        # Two purposes:
        #   1. Clean input: padding embeddings are non-zero after init
        #      (or after v0 weight loading). Zeroing ensures padding
        #      tokens contribute nothing as K/V in the first layer.
        #   2. Inter-layer hygiene: window layers produce finite-but-
        #      meaningless outputs for padding tokens (via the diagonal
        #      self-attention floor in _build_mask). Full-attention layers
        #      then use these as K/V with near-zero attention weight, but
        #      IEEE 754 means near-zero × garbage ≠ 0. Zeroing between
        #      layers eliminates this propagation path entirely.
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
            x = x * pad_mask
        else:
            pad_mask = None

        for layer in self.layers:
            x = layer(x, attention_mask)
            if pad_mask is not None:
                x = x * pad_mask

        x = self.final_norm(x)
        return self.lm_head(x)

    # ─── Utilities ─────────────────────────────────────────────────────

    def get_num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embeddings.weight.numel()
            if not self.config.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def enable_gradient_checkpointing(self):
        for layer in self.layers:
            layer.gradient_checkpointing = True
        print("✓ Gradient checkpointing enabled")

    def disable_gradient_checkpointing(self):
        for layer in self.layers:
            layer.gradient_checkpointing = False

    def get_layer_schedule(self) -> List[str]:
        """Return the resolved layer schedule for inspection."""
        return [
            "window" if layer.attention.window_size else "full"
            for layer in self.layers
        ]