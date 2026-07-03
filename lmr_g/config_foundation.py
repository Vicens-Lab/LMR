# File: model/config_foundation.py
"""
LMR-Foundation v3.0 Configuration

Extends base LMRConfig with geometric inductive bias parameters.
All new fields have safe defaults that fall back to standard transformer behavior.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import yaml


@dataclass
class LMRFoundationConfig:
    """
    Configuration for LMR-Foundation with Geometric Inductive Bias.
    
    All new parameters default to "off" for safe fallback to standard transformer.
    """
    
    # =========================================================================
    # Base Model Architecture (inherited from LMRConfig)
    # =========================================================================
    vocab_size: int = 11
    d_model: int = 768
    n_layers: int = 8
    n_heads: int = 12
    ff_mult: float = 2.67  # SwiGLU parameter-matched (8/3 ≈ 2.67)
    max_seq_len: int = 2048
    
    # Dropout
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # RoPE
    rope_theta: float = 10000.0
    
    # Normalization
    norm_eps: float = 1e-6
    
    # Special tokens
    pad_token_id: int = 0
    start_token_id: int = 1
    end_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4
    msa_sep_token_id: int = 5
    gap_token_id: int = 6
    
    # Weight tying
    tie_word_embeddings: bool = True
    
    # =========================================================================
    # Plücker-Biased Attention (NEW)
    # =========================================================================
    use_plucker_bias: bool = False  # DEFAULT OFF for safe ablation
    geo_dim: int = 4  # Low-dim geometric projection
    plucker_target_gamma: float = 1e-3  # Target gate value after warmup
    plucker_warmup_steps: int = 5000
    
    # =========================================================================
    # Grassmann-Window Layers (NEW)
    # =========================================================================
    use_grassmann: bool = False  # DEFAULT OFF for safe ablation
    grassmann_window: int = 15  # Local mixing window size
    num_grassmann_per_block: int = 3  # Cascade of orthogonal layers
    ortho_method: str = "penalty"  # "penalty" | "cayley" | "muon"
    ortho_penalty_weight: float = 0.01  # Weight for orthogonality penalty loss
    
    # =========================================================================
    # Training Curriculum (NEW)
    # =========================================================================
    use_curriculum: bool = False  # DEFAULT OFF
    curriculum_phase1_steps: int = 30000  # MLM only
    curriculum_phase2_steps: int = 30000  # Add span masking
    curriculum_phase3_steps: int = 40000  # Add stem-span masking
    
    # Task weights (end of curriculum)
    mlm_weight: float = 0.5
    span_weight: float = 0.3
    stem_span_weight: float = 0.2
    
    # =========================================================================
    # Diagnostics (NEW)
    # =========================================================================
    diagnostic_interval: int = 100  # Check orthogonality every N steps
    plucker_diagnostic_interval: int = 500  # Check Plücker influence every N steps
    
    # Warning thresholds
    eigenvalue_spread_warning: float = 100.0
    eigenvalue_spread_critical: float = 1000.0
    plucker_influence_warning: float = 0.001
    
    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def ff_hidden(self) -> int:
        """FFN hidden dimension (parameter-matched for SwiGLU)."""
        hidden = int(self.d_model * self.ff_mult)
        # Round to multiple of 64 for efficiency
        return ((hidden + 63) // 64) * 64
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads
    
    @property
    def wedge_dim(self) -> int:
        """Number of wedge product components for Plücker bias."""
        return (self.geo_dim * (self.geo_dim - 1)) // 2
    
    @property
    def total_curriculum_steps(self) -> int:
        """Total steps for full curriculum."""
        return self.curriculum_phase1_steps + self.curriculum_phase2_steps + self.curriculum_phase3_steps
    
    # =========================================================================
    # YAML Loading
    # =========================================================================
    @classmethod
    def from_yaml(cls, config_path: str) -> 'LMRFoundationConfig':
        """Load config from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract model params from nested structure
        if 'model' in config_dict:
            model_params = config_dict['model']
        else:
            model_params = config_dict
        
        # Filter to valid fields only
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_params = {k: v for k, v in model_params.items() if k in valid_fields}
        
        print(f"[LMRFoundationConfig] Loading from: {config_path}")
        print(f"  Plücker bias: {'ENABLED' if filtered_params.get('use_plucker_bias', False) else 'disabled'}")
        print(f"  Grassmann: {'ENABLED' if filtered_params.get('use_grassmann', False) else 'disabled'}")
        print(f"  Curriculum: {'ENABLED' if filtered_params.get('use_curriculum', False) else 'disabled'}")
        
        return cls(**filtered_params)
    
    def to_yaml(self, save_path: str):
        """Save config to YAML file."""
        import dataclasses
        config_dict = {'model': dataclasses.asdict(self)}
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def estimate_foundation_params(config: LMRFoundationConfig) -> dict:
    """Estimate parameter count for LMR-Foundation model."""
    
    # Embeddings
    emb_params = config.vocab_size * config.d_model
    
    # Per-block params
    # Standard attention: 4 * d² (Q, K, V, O projections)
    attn_params = 4 * config.d_model * config.d_model
    
    # Plücker additions (if enabled)
    plucker_params = 0
    if config.use_plucker_bias:
        # q_geo, k_geo projections per head
        plucker_params = 2 * config.head_dim * config.geo_dim * config.n_heads
        # wedge_proj
        plucker_params += config.wedge_dim * 1
        # gamma parameter
        plucker_params += 1
    
    # Grassmann layers (if enabled)
    grassmann_params = 0
    if config.use_grassmann:
        for _ in range(config.num_grassmann_per_block):
            # W_ortho: d × d
            grassmann_params += config.d_model * config.d_model
            # local_mix (depthwise conv): d × window
            grassmann_params += config.d_model * config.grassmann_window
            # gate: d × d
            grassmann_params += config.d_model * config.d_model
    
    # SwiGLU FFN: 3 * d * ff_hidden
    ffn_params = 3 * config.d_model * config.ff_hidden
    
    # Norms: 2 * d (or more with Grassmann)
    norm_params = 2 * config.d_model
    if config.use_grassmann:
        norm_params += config.num_grassmann_per_block * config.d_model
    
    # Total per block
    block_params = attn_params + plucker_params + grassmann_params + ffn_params + norm_params
    
    # Total model
    total_params = emb_params + config.n_layers * block_params + config.d_model  # final norm
    
    return {
        'embedding_params': emb_params,
        'attention_params_per_block': attn_params,
        'plucker_params_per_block': plucker_params,
        'grassmann_params_per_block': grassmann_params,
        'ffn_params_per_block': ffn_params,
        'norm_params_per_block': norm_params,
        'block_params': block_params,
        'total_params': total_params,
        'total_M': total_params / 1e6,
        'within_budget': total_params < 300_000_000,
    }