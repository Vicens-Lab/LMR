# File: /home/admin/locked/model/config.py
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class LMRConfig:
    # - Configuration for LMR (Language Model for RNA) --
    # ====================================================
    # Model architecture
    vocab_size: int = 11
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    ff_mult: float = 2.5
    max_seq_len: int = 4096
    
    # Architecture type
    architecture_type: str = "transformer" 
    
    # Layer types for hybrid architecture (v2 only)
    layer_types: Optional[List[str]] = None
    
    # Transformer configuration
    attention_window: Optional[int] = None  # None = full attention, int = sliding window
    
    # Dropout
    dropout: float = 0.05
    attention_dropout: float = 0.0
    
    # RoPE configuration
    rope_theta: float = 10000.0
    
    # Normalization
    norm_eps: float = 1e-6
    
    # Initialization
    initializer_range: float = 0.02
    
    # Special token IDs
    pad_token_id: int = 0
    start_token_id: int = 1
    end_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4
    msa_sep_token_id: int = 5
    gap_token_id: int = 6
    
    # Training
    tie_word_embeddings: bool = True
    
    @property
    def ff_hidden(self) -> int:
        """FFN hidden dimension"""
        return int(self.d_model * self.ff_mult)
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'LMRConfig':
        """Load config from YAML file"""
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
        
        # CRITICAL FIX: Filter out only valid config parameters
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_params = {k: v for k, v in model_params.items() if k in valid_fields}
        
        # DEBUG: Print what we're loading
        print(f"Loading config with params: {list(filtered_params.keys())}")
        if 'layer_types' in filtered_params:
            print(f"Layer types: {filtered_params['layer_types'][:5]}... (showing first 5)")
        
        return cls(**filtered_params)
    
    def to_yaml(self, save_path: str):
        # - Save config to YAML file
        config_dict = {
            'model': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'ff_mult': self.ff_mult,
                'max_seq_len': self.max_seq_len,
                'architecture_type': self.architecture_type,
                'layer_types': self.layer_types,
                'attention_window': self.attention_window,
                'd_state': self.d_state,
                'd_conv': self.d_conv,
                'mamba_expand': self.mamba_expand,
                'bidirectional': self.bidirectional,
                'use_moe': self.use_moe,
                'moe_num_experts': self.moe_num_experts,
                'moe_top_k': self.moe_top_k,
                'moe_capacity_factor': self.moe_capacity_factor,
                'moe_aux_loss_weight': self.moe_aux_loss_weight,
                'dropout': self.dropout,
                'attention_dropout': self.attention_dropout,
                'rope_theta': self.rope_theta,
                'norm_eps': self.norm_eps,
                'initializer_range': self.initializer_range,
                'pad_token_id': self.pad_token_id,
                'start_token_id': self.start_token_id,
                'end_token_id': self.end_token_id,
                'mask_token_id': self.mask_token_id,
                'unk_token_id': self.unk_token_id,
                'msa_sep_token_id': self.msa_sep_token_id,
                'gap_token_id': self.gap_token_id,
                'tie_word_embeddings': self.tie_word_embeddings,
            }
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def estimate_params(config: LMRConfig) -> dict:
    # - Estimate model parameters
    # - Embeddings (only count once if tied)
    emb_params = config.vocab_size * config.d_model
    
    # - Pure Transformer
    layer_params = (
        4 * config.d_model * config.d_model +           # Attention
        config.d_model * config.ff_hidden * 3 +         # SwiGLU
        2 * config.d_model                               # 2x RMSNorm
    )
    
    total_params = emb_params + config.n_layers * layer_params
    active_params = total_params
    
    return {
        'embedding_params': emb_params,
        'layer_params': "varies by layer type" if hasattr(config, 'layer_types') and config.layer_types else layer_params,
        'total_layers': config.n_layers,
        'total_params': total_params,
        'total_M': total_params / 1e6,
        'active_params': active_params,
        'active_M': active_params / 1e6,
        'non_embedding_M': (total_params - emb_params) / 1e6
    }