# File: lmr_g/architecture_foundation.py
"""
LMR-Foundation v3.0 Architecture (Self-Contained)

Extends base LMR architecture with geometric inductive biases:
1. Plücker-biased attention - orientation-sensitive geometric priors
2. Grassmann-window layers - local orthogonal mixing
3. SwiGLU FFN - parameter-matched feed-forward

All components are ablatable via config flags.
Falls back to standard transformer when geometric features disabled.

This module is SELF-CONTAINED - no external imports from base architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from torch.utils.checkpoint import checkpoint


# =============================================================================
# Flash Attention Availability Check
# =============================================================================
def _check_flash_attention_available() -> bool:
    """Check if PyTorch's efficient attention is available."""
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False
    if not torch.cuda.is_available():
        return False
    return True

FLASH_ATTENTION_AVAILABLE = _check_flash_attention_available()


# =============================================================================
# RMSNorm - Root Mean Square Layer Normalization
# =============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


# =============================================================================
# RoPE - Rotary Positional Embeddings (NCCL-Compatible)
# =============================================================================
def precompute_freqs_cis(
        dim: int, 
        max_seq_len: int, 
        theta: float = 10000.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute rotation frequencies for RoPE.
    
    Returns cos and sin separately instead of complex tensor for NCCL compatibility.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim//2]
    
    freqs_cos = torch.cos(freqs)  # [max_seq_len, dim//2]
    freqs_sin = torch.sin(freqs)  # [max_seq_len, dim//2]
    
    return freqs_cos, freqs_sin


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: [batch, n_heads, seq_len, head_dim]
        k: [batch, n_heads, seq_len, head_dim]
        freqs_cos: [seq_len, head_dim//2]
        freqs_sin: [seq_len, head_dim//2]
    """
    # Reshape q and k to separate pairs: [..., head_dim] -> [..., head_dim//2, 2]
    q_reshape = q.float().reshape(*q.shape[:-1], -1, 2)
    k_reshape = k.float().reshape(*k.shape[:-1], -1, 2)
    
    # Split into real/imaginary (even/odd) components
    q_r, q_i = q_reshape[..., 0], q_reshape[..., 1]
    k_r, k_i = k_reshape[..., 0], k_reshape[..., 1]
    
    # Expand freqs for broadcasting: [seq_len, head_dim//2] -> [1, 1, seq_len, head_dim//2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation
    q_out_r = q_r * freqs_cos - q_i * freqs_sin
    q_out_i = q_r * freqs_sin + q_i * freqs_cos
    k_out_r = k_r * freqs_cos - k_i * freqs_sin
    k_out_i = k_r * freqs_sin + k_i * freqs_cos
    
    # Stack back together
    q_rotated = torch.stack([q_out_r, q_out_i], dim=-1).flatten(-2)
    k_rotated = torch.stack([k_out_r, k_out_i], dim=-1).flatten(-2)
    
    return q_rotated.type_as(q), k_rotated.type_as(k)


# =============================================================================
# SwiGLU Feed-Forward Network
# =============================================================================
class SwiGLU(nn.Module):
    """
    SwiGLU: Gated Linear Unit with SiLU (Swish) activation.
    
    SwiGLU(x) = (W1(x) ⊙ SiLU(W2(x))) @ W3
    """
    
    def __init__(self, d_model: int, ff_hidden: int, dropout: float = 0.0):
        super().__init__()
        
        self.w1 = nn.Linear(d_model, ff_hidden, bias=False)  # Gate
        self.w2 = nn.Linear(d_model, ff_hidden, bias=False)  # Value
        self.w3 = nn.Linear(ff_hidden, d_model, bias=False)  # Output
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w2(x)
        hidden = F.silu(gate) * value
        hidden = self.dropout(hidden)
        output = self.w3(hidden)
        return output


# =============================================================================
# Plücker-Biased Attention
# =============================================================================
class PluckerBiasedAttention(nn.Module):
    """
    Multi-head attention augmented with geometric bias from wedge products.
    
    Key innovations:
    1. Normalized low-d projections (prevents magnitude hacks)
    2. Learnable gate γ (starts at 0, warms up)
    3. Tanh clamping (prevents logit explosion)
    
    Args:
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        dropout: Attention dropout probability
        max_seq_len: Maximum sequence length for RoPE
        rope_theta: RoPE base frequency
        geo_dim: Dimension for geometric projections (default: 4)
        use_plucker: Whether to enable Plücker bias (for ablation)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        geo_dim: int = 4,
        use_plucker: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.geo_dim = geo_dim
        self.use_plucker = use_plucker
        self.scale = self.head_dim ** -0.5
        
        # Standard Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # RoPE frequencies (NCCL-compatible: separate cos/sin)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # Plücker pathway (only if enabled)
        if use_plucker:
            # Per-head geometric projections
            self.q_geo = nn.Linear(self.head_dim, geo_dim, bias=False)
            self.k_geo = nn.Linear(self.head_dim, geo_dim, bias=False)
            
            # Wedge feature projection: C(geo_dim, 2) -> 1
            wedge_dim = (geo_dim * (geo_dim - 1)) // 2
            self.wedge_proj = nn.Linear(wedge_dim, 1, bias=False)
            
            # Learnable gate (CRITICAL: initialize to 0)
            self.gamma = nn.Parameter(torch.zeros(1))
            
            # Initialize wedge projection small
            nn.init.normal_(self.wedge_proj.weight, std=0.02)
        
        # Track attention mode
        self.use_flash_attention = FLASH_ATTENTION_AVAILABLE
    
    def compute_wedge_bias(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute oriented area (wedge product) bias.
        
        Args:
            q, k: [batch, heads, length, head_dim]
        Returns:
            bias: [batch, heads, length, length]
        """
        # Project to geometric space
        q_low = self.q_geo(q)  # [B, H, L, geo_dim]
        k_low = self.k_geo(k)
        
        # CRITICAL: Normalize to prevent magnitude exploitation
        q_low = F.normalize(q_low, dim=-1, eps=1e-6)
        k_low = F.normalize(k_low, dim=-1, eps=1e-6)
        
        # Compute all antisymmetric 2-form components
        wedge_features = []
        for i in range(self.geo_dim):
            for j in range(i + 1, self.geo_dim):
                qi = q_low[..., i:i+1]
                qj = q_low[..., j:j+1]
                ki = k_low[..., i:i+1]
                kj = k_low[..., j:j+1]
                
                # Antisymmetric outer product: [B, H, L, L]
                w_ij = (qi @ kj.transpose(-2, -1) - 
                        qj @ ki.transpose(-2, -1))
                wedge_features.append(w_ij)
        
        # Stack: [B, H, L, L, wedge_dim]
        wedge_stack = torch.stack(wedge_features, dim=-1)
        
        # Project to scalar: [B, H, L, L]
        bias = self.wedge_proj(wedge_stack).squeeze(-1)
        
        # CRITICAL: Clamp to prevent explosion
        bias = torch.tanh(bias)
        
        return bias
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_plucker: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional Plücker bias.
        
        Args:
            x: [batch, length, d_model]
            attention_mask: Optional [batch, length]
            use_plucker: Override instance setting (for ablation)
        Returns:
            out: [batch, length, d_model]
        """
        B, L, D = x.shape
        
        # Determine if using Plücker (allow runtime override)
        apply_plucker = self.use_plucker if use_plucker is None else use_plucker
        apply_plucker = apply_plucker and hasattr(self, 'gamma')
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        freqs_cos = self.freqs_cos[:L]
        freqs_sin = self.freqs_sin[:L]
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        
        # Compute attention scores
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add Plücker bias (if enabled)
        if apply_plucker:
            wedge_bias = self.compute_wedge_bias(q, k)
            attn_logits = attn_logits + self.gamma * wedge_bias
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        out = attn_weights @ v
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out
    
    def get_gamma(self) -> float:
        """Get current gamma value (for logging)."""
        if hasattr(self, 'gamma'):
            return self.gamma.item()
        return 0.0


# =============================================================================
# Grassmann-Window Layer
# =============================================================================
class GrassmannWindowLayer(nn.Module):
    """
    Local orthogonal mixing layer.
    
    Components:
    1. Orthogonal projection W (maintained via penalty or Cayley)
    2. Depthwise convolution (local window mixing)
    3. Gated residual (controls contribution)
    
    Args:
        dim: Hidden dimension
        window: Local mixing window size
        ortho_method: "penalty" | "cayley"
    """
    
    def __init__(
        self,
        dim: int = 768,
        window: int = 15,
        ortho_method: str = "penalty"
    ):
        super().__init__()
        self.dim = dim
        self.window = window
        self.ortho_method = ortho_method
        
        if ortho_method == "cayley":
            # Cayley parameterization: W = (I + A)^{-1}(I - A) where A is skew-symmetric
            self.A = nn.Parameter(torch.randn(dim, dim) * 0.01)
        else:
            # Standard linear (maintained orthogonal via penalty)
            self.W_ortho = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.W_ortho.weight)
        
        # Local mixer (depthwise convolution)
        self.local_mix = nn.Conv1d(
            dim, dim,
            kernel_size=window,
            padding=window // 2,
            groups=dim,
            bias=False
        )
        
        # Gate
        self.gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
    
    def get_orthogonal_matrix(self) -> torch.Tensor:
        """Get the orthogonal matrix."""
        if self.ortho_method == "cayley":
            A_skew = self.A - self.A.T
            I = torch.eye(self.dim, device=self.A.device, dtype=self.A.dtype)
            W = torch.linalg.solve(I + A_skew, I - A_skew)
            return W
        else:
            return self.W_ortho.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, length, dim]
        Returns:
            delta: [batch, length, dim]
        """
        # 1. Orthogonal projection
        if self.ortho_method == "cayley":
            W = self.get_orthogonal_matrix()
            h = F.linear(x, W)
        else:
            h = self.W_ortho(x)
        
        # 2. Local window mixing
        h = h.transpose(1, 2)  # [B, dim, L]
        h = self.local_mix(h)
        h = h.transpose(1, 2)  # [B, L, dim]
        
        # 3. Nonlinearity
        h = F.gelu(h)
        
        # 4. Gated output
        g = torch.sigmoid(self.gate(x))
        
        return g * h
    
    def orthogonality_penalty(self) -> torch.Tensor:
        """Compute orthogonality penalty: ||W^T W - I||_F^2"""
        if self.ortho_method == "cayley":
            return torch.tensor(0.0, device=self.A.device)
        
        W = self.W_ortho.weight
        WtW = W.T @ W
        I = torch.eye(self.dim, device=W.device, dtype=W.dtype)
        penalty = torch.norm(WtW - I, p='fro') ** 2
        return penalty


# =============================================================================
# LMR Foundation Block
# =============================================================================
class LMRFoundationBlock(nn.Module):
    """
    One LMR-Foundation block.
    
    Pattern: [Plücker Attention] → [Grassmann × N] → [SwiGLU FFN]
    
    All components ablatable via forward() flags.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        ff_hidden: int = 2048,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        # Foundation-specific
        use_plucker: bool = False,
        geo_dim: int = 4,
        use_grassmann: bool = False,
        grassmann_window: int = 15,
        num_grassmann: int = 3,
        ortho_method: str = "penalty"
    ):
        super().__init__()
        
        self.use_plucker = use_plucker
        self.use_grassmann = use_grassmann
        
        # Pre-attention norm
        self.attention_norm = RMSNorm(d_model, eps=norm_eps)
        
        # Attention (Plücker-biased if enabled)
        self.attention = PluckerBiasedAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            geo_dim=geo_dim,
            use_plucker=use_plucker
        )
        
        # Grassmann layers (if enabled)
        if use_grassmann:
            self.grassmann_layers = nn.ModuleList([
                GrassmannWindowLayer(d_model, grassmann_window, ortho_method)
                for _ in range(num_grassmann)
            ])
            self.grassmann_norms = nn.ModuleList([
                RMSNorm(d_model, eps=norm_eps)
                for _ in range(num_grassmann)
            ])
        
        # Pre-FFN norm
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        
        # SwiGLU FFN
        self.ffn = SwiGLU(d_model, ff_hidden, dropout)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_plucker: Optional[bool] = None,
        use_grassmann: Optional[bool] = None
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, attention_mask, use_plucker, use_grassmann)
        return self._forward_impl(x, attention_mask, use_plucker, use_grassmann)
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_plucker: Optional[bool] = None,
        use_grassmann: Optional[bool] = None
    ) -> torch.Tensor:
        apply_plucker = self.use_plucker if use_plucker is None else use_plucker
        apply_grassmann = self.use_grassmann if use_grassmann is None else use_grassmann
        
        # 1. Attention
        h = self.attention_norm(x)
        h = self.attention(h, attention_mask, use_plucker=apply_plucker)
        h = self.dropout(h)
        x = x + h
        
        # 2. Grassmann layers
        if apply_grassmann and hasattr(self, 'grassmann_layers'):
            for layer, norm in zip(self.grassmann_layers, self.grassmann_norms):
                x = x + layer(norm(x))
        
        # 3. FFN
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.dropout(h)
        x = x + h
        
        return x
    
    def _forward_with_checkpointing(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        use_plucker: Optional[bool],
        use_grassmann: Optional[bool]
    ) -> torch.Tensor:
        apply_plucker = self.use_plucker if use_plucker is None else use_plucker
        apply_grassmann = self.use_grassmann if use_grassmann is None else use_grassmann
        
        def attn_block(x_in):
            h = self.attention_norm(x_in)
            h = self.attention(h, attention_mask, use_plucker=apply_plucker)
            h = self.dropout(h)
            return x_in + h
        
        x = checkpoint(attn_block, x, use_reentrant=False)
        
        if apply_grassmann and hasattr(self, 'grassmann_layers'):
            for layer, norm in zip(self.grassmann_layers, self.grassmann_norms):
                def grass_block(x_in, layer=layer, norm=norm):
                    return x_in + layer(norm(x_in))
                x = checkpoint(grass_block, x, use_reentrant=False)
        
        def ffn_block(x_in):
            h = self.ffn_norm(x_in)
            h = self.ffn(h)
            h = self.dropout(h)
            return x_in + h
        
        x = checkpoint(ffn_block, x, use_reentrant=False)
        
        return x
    
    def get_orthogonality_penalty(self) -> torch.Tensor:
        """Get total orthogonality penalty."""
        if not hasattr(self, 'grassmann_layers'):
            return torch.tensor(0.0)
        
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.grassmann_layers:
            total = total + layer.orthogonality_penalty()
        return total


# =============================================================================
# LMR Foundation Model
# =============================================================================
class LMRFoundation(nn.Module):
    """
    LMR-Foundation: RNA Language Model with Geometric Inductive Bias.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            LMRFoundationBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_hidden=config.ff_hidden,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                norm_eps=config.norm_eps,
                use_plucker=config.use_plucker_bias,
                geo_dim=config.geo_dim,
                use_grassmann=config.use_grassmann,
                grassmann_window=config.grassmann_window,
                num_grassmann=config.num_grassmann_per_block,
                ortho_method=config.ortho_method
            )
            for _ in range(config.n_layers)
        ])
        
        # Final norm
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize
        self.apply(self._init_weights)
        self._log_config()
    
    def _log_config(self):
        print("\n" + "="*70)
        print("LMR-Foundation Model")
        print("="*70)
        print(f"  Layers: {self.config.n_layers}")
        print(f"  d_model: {self.config.d_model}")
        print(f"  n_heads: {self.config.n_heads}")
        print(f"  ff_hidden: {self.config.ff_hidden}")
        print(f"  Plücker bias: {'ENABLED' if self.config.use_plucker_bias else 'disabled'}")
        if self.config.use_plucker_bias:
            print(f"    geo_dim: {self.config.geo_dim}")
        print(f"  Grassmann: {'ENABLED' if self.config.use_grassmann else 'disabled'}")
        if self.config.use_grassmann:
            print(f"    window: {self.config.grassmann_window}")
            print(f"    layers_per_block: {self.config.num_grassmann_per_block}")
        print(f"  Total params: {self.get_num_params() / 1e6:.1f}M")
        print("="*70 + "\n")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_plucker: Optional[bool] = None,
        use_grassmann: Optional[bool] = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask, use_plucker, use_grassmann)
        
        x = self.final_norm(x)
        
        if return_hidden:
            return x
        
        logits = self.lm_head(x)
        return logits
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            if not self.config.tie_word_embeddings:
                n_params -= self.lm_head.weight.numel()
        return n_params
    
    def get_orthogonality_penalty(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total = total + layer.get_orthogonality_penalty()
        return total
    
    def enable_gradient_checkpointing(self):
        for layer in self.layers:
            layer.gradient_checkpointing = True
        print("✓ Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self):
        for layer in self.layers:
            layer.gradient_checkpointing = False
    
    def get_plucker_gammas(self) -> Dict[int, float]:
        gammas = {}
        for i, layer in enumerate(self.layers):
            gammas[i] = layer.attention.get_gamma()
        return gammas
    
    def set_plucker_gammas(self, gamma: float):
        for layer in self.layers:
            if hasattr(layer.attention, 'gamma'):
                layer.attention.gamma.data.fill_(gamma)


# =============================================================================
# Factory Function
# =============================================================================
def create_lmr_foundation(config) -> LMRFoundation:
    """Factory function to create LMRFoundation model."""
    return LMRFoundation(config)