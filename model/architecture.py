# File: model/architecture.py
"""
LMR v0 Architecture with Flash Attention

Changes from previous version:
- Integrated PyTorch 2.0+ scaled_dot_product_attention (Flash Attention)
- Automatic fallback to vanilla attention for older PyTorch versions
- Memory complexity reduced from O(n²) to O(n) for Flash path
- Expected speedup: 3-4x on attention computation

Requires: PyTorch >= 2.0 for Flash Attention (falls back gracefully)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

from .config import LMRConfig


# ============================================================================
# Flash Attention Availability Check
# ============================================================================
def _check_flash_attention_available() -> bool:
    """Check if PyTorch's efficient attention is available."""
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False
    # Check CUDA availability for Flash Attention kernel
    if not torch.cuda.is_available():
        return False
    # PyTorch 2.0+ has SDPA, but Flash Attention kernel requires CUDA
    return True

FLASH_ATTENTION_AVAILABLE = _check_flash_attention_available()


# ============================================================================
# RMSNorm - Root Mean Square Layer Normalization
# ============================================================================
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


# ============================================================================
# RoPE - Rotary Positional Embeddings (NCCL-Compatible - No ComplexFloat)
# ============================================================================
def precompute_freqs_cis(
        dim: int, 
        max_seq_len: int, 
        theta: float = 10000.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute rotation frequencies for RoPE.
    
    Returns cos and sin separately instead of complex tensor for NCCL compatibility.
    NCCL does not support ComplexFloat tensors.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim//2]
    
    # Return cos and sin separately instead of torch.polar()
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
    
    Uses cos/sin directly instead of complex multiplication for NCCL compatibility.
    
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
    q_r, q_i = q_reshape[..., 0], q_reshape[..., 1]  # [..., head_dim//2]
    k_r, k_i = k_reshape[..., 0], k_reshape[..., 1]
    
    # Expand freqs for broadcasting: [seq_len, head_dim//2] -> [1, 1, seq_len, head_dim//2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation using: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    q_out_r = q_r * freqs_cos - q_i * freqs_sin
    q_out_i = q_r * freqs_sin + q_i * freqs_cos
    k_out_r = k_r * freqs_cos - k_i * freqs_sin
    k_out_i = k_r * freqs_sin + k_i * freqs_cos
    
    # Stack back together: [..., head_dim//2] x 2 -> [..., head_dim]
    q_rotated = torch.stack([q_out_r, q_out_i], dim=-1).flatten(-2)
    k_rotated = torch.stack([k_out_r, k_out_i], dim=-1).flatten(-2)
    
    return q_rotated.type_as(q), k_rotated.type_as(k)


# ============================================================================
# Multi-Head Attention with RoPE + Flash Attention
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with RoPE positional encoding.
    
    Uses Flash Attention (via PyTorch 2.0+ scaled_dot_product_attention) when available,
    with automatic fallback to vanilla attention for compatibility.
    
    Flash Attention benefits:
    - Memory: O(n) instead of O(n²)
    - Speed: 3-4x faster attention computation
    - Fused kernel: softmax + dropout + matmul in single operation
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout (only used in vanilla attention path)
        self.dropout = nn.Dropout(dropout)
        
        # Precompute RoPE frequencies (cos and sin separately for NCCL compatibility)
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # Track which attention method is used
        self.use_flash_attention = FLASH_ATTENTION_AVAILABLE
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: [batch, seq, d_model] -> [batch, seq, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose to [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE to Q and K (using cos/sin instead of complex)
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        
        # Choose attention implementation
        if self.use_flash_attention:
            attn_output = self._flash_attention(q, k, v, attention_mask)
        else:
            attn_output = self._vanilla_attention(q, k, v, attention_mask)
        
        # Transpose back and reshape: [batch, n_heads, seq, head_dim] -> [batch, seq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flash Attention via PyTorch 2.0+ scaled_dot_product_attention.
        
        Memory efficient O(n) implementation with fused kernel.
        """
        # Convert attention mask format for SDPA
        # SDPA expects: None, boolean mask, or additive float mask
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] with 1=attend, 0=ignore
            # Convert to [batch, 1, 1, seq_len] boolean mask
            # SDPA with is_causal=False uses mask where True = attend
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
            # Expand to [batch, 1, seq_len, seq_len] for full attention pattern
            attn_mask = attn_mask.expand(-1, -1, q.size(2), -1)
        
        # Use scaled_dot_product_attention with Flash Attention
        # This automatically uses the most efficient kernel available:
        # 1. Flash Attention (fastest, O(n) memory)
        # 2. Memory-efficient attention
        # 3. Math attention (fallback)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,  # MLM uses bidirectional attention
            scale=1.0 / math.sqrt(self.head_dim)
        )
        
        return attn_output
    
    def _vanilla_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention (fallback for older PyTorch).
        
        Memory intensive O(n²) implementation.
        """
        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] with 1=attend, 0=ignore
            # Expand to [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output


# ============================================================================
# SwiGLU Feed-Forward Network
# ============================================================================
class SwiGLU(nn.Module):
    """SwiGLU: Gated Linear Unit with SiLU (Swish) activation"""
    
    def __init__(self, d_model: int, ff_hidden: int, dropout: float = 0.0):
        super().__init__()
        
        # Gate and value projections
        self.w1 = nn.Linear(d_model, ff_hidden, bias=False)  # Gate
        self.w2 = nn.Linear(d_model, ff_hidden, bias=False)  # Value
        
        # Output projection
        self.w3 = nn.Linear(ff_hidden, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w2(x)
        
        # Apply SiLU to gate and multiply with value
        hidden = F.silu(gate) * value
        hidden = self.dropout(hidden)
        
        # Project back to d_model
        output = self.w3(hidden)
        
        return output


# ============================================================================
# Transformer Block (with Pre-Norm)
# ============================================================================
class TransformerBlock(nn.Module):
    """Single transformer block with Pre-Norm architecture"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_hidden: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Pre-attention norm
        self.attention_norm = RMSNorm(d_model, eps=norm_eps)
        
        # Multi-head attention with RoPE + Flash Attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta
        )
        
        # Pre-FFN norm
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        
        # SwiGLU feed-forward
        self.ffn = SwiGLU(d_model, ff_hidden, dropout)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Use gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, attention_mask)
        else:
            return self._forward_impl(x, attention_mask)
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normal forward pass"""
        # Pre-Norm attention with residual
        h = self.attention_norm(x)
        h = self.attention(h, attention_mask)
        h = self.dropout(h)
        x = x + h
        
        # Pre-Norm FFN with residual
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.dropout(h)
        x = x + h
        
        return x
    
    def _forward_with_checkpointing(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with gradient checkpointing (saves memory)"""
        
        # Checkpoint attention block
        def attention_block(x_input):
            h = self.attention_norm(x_input)
            h = self.attention(h, attention_mask)
            h = self.dropout(h)
            return x_input + h
        
        x = checkpoint(attention_block, x, use_reentrant=False)
        
        # Checkpoint FFN block
        def ffn_block(x_input):
            h = self.ffn_norm(x_input)
            h = self.ffn(h)
            h = self.dropout(h)
            return x_input + h
        
        x = checkpoint(ffn_block, x, use_reentrant=False)
        
        return x


# ============================================================================
# LMR - Language Model for RNA
# ============================================================================
class LMR(nn.Module):
    """
    LMR: Language Model for RNA
    
    Transformer encoder for masked language modeling on RNA sequences.
    
    Key features:
    - Flash Attention (PyTorch 2.0+) for efficient O(n) attention
    - RoPE positional embeddings (no learned positions)
    - Pre-Norm architecture with RMSNorm
    - SwiGLU feed-forward network
    - Gradient checkpointing support
    """
    
    def __init__(self, config: LMRConfig):
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
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_hidden=config.ff_hidden,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                norm_eps=config.norm_eps
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # LM head (project to vocabulary)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings (share weights between input and output)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log attention mode
        self._log_attention_mode()
    
    def _log_attention_mode(self):
        """Log which attention implementation is being used."""
        if FLASH_ATTENTION_AVAILABLE:
            print("✓ Flash Attention ENABLED (PyTorch SDPA)")
        else:
            print("⚠ Flash Attention NOT available - using vanilla attention")
            print("  Tip: Upgrade to PyTorch 2.0+ for Flash Attention support")
    
    def _init_weights(self, module):
        """Initialize weights with normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] with 1=attend, 0=ignore
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.token_embeddings(input_ids)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Count number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
            if not self.config.tie_word_embeddings:
                n_params -= self.lm_head.weight.numel()
        return n_params
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        for layer in self.layers:
            layer.gradient_checkpointing = True
        print("✓ Gradient checkpointing enabled")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        for layer in self.layers:
            layer.gradient_checkpointing = False
        print("✓ Gradient checkpointing disabled")