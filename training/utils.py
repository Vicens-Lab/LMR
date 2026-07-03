# File: training/utils_enhanced.py
# ============================================================================
# ENHANCED TRAINING UTILITIES - IMPROVED LR SCHEDULERS
# ============================================================================
# New LR schedulers vs. original utils.py:
#   1. Cosine with warm restarts (escapes plateaus)
#   2. Inverse square root (proven for Transformers)
#   3. Triangular cyclic (explores diverse learning rates)
#   4. Polynomial decay (smooth alternative to cosine)
# 
# All schedulers support proper warmup and are configurable via config file.
# ============================================================================

import torch
import random
import numpy as np
import os
import glob
import math
from typing import Dict, Any


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_checkpoints(checkpoint_dir: str, keep_best: bool = True, keep_last_n: int = 2):
    """
    Clean up checkpoints, keeping only:
    - Best checkpoint (if keep_best=True)
    - Last N step checkpoints
    
    Total kept: 1 best + 2 last = 3 checkpoints max
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all checkpoint files
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    
    if not all_checkpoints:
        return
    
    # Separate best checkpoint from others
    best_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    
    # Get step checkpoints (exclude best and interrupt)
    step_checkpoints = [
        f for f in all_checkpoints
        if "checkpoint_step_" in f and f != best_checkpoint
    ]
    
    # Sort by step number
    def get_step_number(filepath):
        try:
            basename = os.path.basename(filepath)
            step_str = basename.replace("checkpoint_step_", "").replace(".pt", "")
            return int(step_str)
        except:
            return 0
    
    step_checkpoints.sort(key=get_step_number)
    
    # Keep only last N step checkpoints
    if len(step_checkpoints) > keep_last_n:
        checkpoints_to_delete = step_checkpoints[:-keep_last_n]
        
        for ckpt in checkpoints_to_delete:
            try:
                os.remove(ckpt)
                print(f"  Removed old checkpoint: {os.path.basename(ckpt)}")
            except Exception as e:
                print(f"  Warning: Could not remove {ckpt}: {e}")
    
    # Clean up interrupt checkpoints
    interrupt_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_interrupt_*.pt"))
    if len(interrupt_checkpoints) > 1:
        interrupt_checkpoints.sort(key=os.path.getmtime)
        for ckpt in interrupt_checkpoints[:-1]:
            try:
                os.remove(ckpt)
                print(f"  Removed old interrupt checkpoint: {os.path.basename(ckpt)}")
            except Exception as e:
                print(f"  Warning: Could not remove {ckpt}: {e}")
    
    # Print summary
    remaining = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    print(f"\n  Total checkpoints remaining: {len(remaining)}")
    if os.path.exists(best_checkpoint):
        print(f"  - Best checkpoint: checkpoint_best.pt")
    step_remaining = [f for f in remaining if "checkpoint_step_" in f]
    print(f"  - Step checkpoints: {len(step_remaining)}")


def get_lr_scheduler(optimizer, config: Dict, num_training_steps: int):
    """
    Create enhanced learning rate scheduler.
    
    Supported schedulers:
    - 'cosine': Standard cosine decay
    - 'cosine_restarts': Cosine with warm restarts (NEW)
    - 'inverse_sqrt': Inverse square root decay (NEW)
    - 'triangular': Cyclical triangular schedule (NEW)
    - 'polynomial': Polynomial decay (NEW)
    - 'linear': Linear decay (original)
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration dict
        num_training_steps: Total number of training steps
    
    Returns:
        LRScheduler instance
    """
    warmup_steps = config['training']['warmup_steps']
    scheduler_type = config['training'].get('lr_scheduler_type', 'cosine')
    
    print(f"\n{'='*70}")
    print(f"Learning Rate Scheduler: {scheduler_type}")
    print(f"{'='*70}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Total steps: {num_training_steps:,}")
    print(f"  Base LR: {config['training']['learning_rate']:.2e}")
    
    if scheduler_type == 'cosine_restarts':
        # ====================================================================
        # COSINE WITH WARM RESTARTS
        # ====================================================================
        # Periodically resets LR to escape plateaus
        # Proven effective for preventing early convergence
        # ====================================================================
        restart_period = config['training'].get('restart_period', 50000)
        restart_mult = config['training'].get('restart_mult', 1.0)
        
        print(f"  Restart period: {restart_period:,} steps")
        print(f"  Restart multiplier: {restart_mult}")
        print(f"{'='*70}\n")
        
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Post-warmup with restarts
            step_after_warmup = current_step - warmup_steps
            
            # Determine current restart cycle
            current_period = restart_period
            cycle = 0
            accumulated_steps = 0
            
            while accumulated_steps + current_period <= step_after_warmup:
                accumulated_steps += current_period
                cycle += 1
                current_period = int(restart_period * (restart_mult ** cycle))
            
            # Position within current cycle
            step_in_cycle = step_after_warmup - accumulated_steps
            progress = step_in_cycle / current_period
            
            # Cosine decay within cycle
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'inverse_sqrt':
        # ====================================================================
        # INVERSE SQUARE ROOT SCHEDULE
        # ====================================================================
        # Used by original Transformer paper (Vaswani et al.)
        # Formula: lr = base_lr * min(step^(-0.5), step * warmup^(-1.5))
        # Prevents premature decay, works well for large models
        # ====================================================================
        print(f"{'='*70}\n")
        
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            current_step = max(1, current_step)  # Avoid division by zero
            
            # Inverse sqrt formula
            warmup_factor = warmup_steps ** (-1.5)
            decay_factor = current_step ** (-0.5)
            
            return min(decay_factor, current_step * warmup_factor)
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'triangular':
        # ====================================================================
        # TRIANGULAR CYCLICAL SCHEDULE
        # ====================================================================
        # Cycles LR between min and max values
        # Explores diverse learning rates → better generalization
        # ====================================================================
        cycle_length = config['training'].get('cycle_length', 10000)
        min_lr_factor = config['training'].get('min_lr_factor', 0.1)
        
        print(f"  Cycle length: {cycle_length:,} steps")
        print(f"  Min LR factor: {min_lr_factor}")
        print(f"{'='*70}\n")
        
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Triangular cycling
            step_after_warmup = current_step - warmup_steps
            cycle_position = (step_after_warmup % cycle_length) / cycle_length
            
            # Triangle wave: 0 → 1 → 0
            if cycle_position < 0.5:
                # Ascending
                lr_factor = min_lr_factor + (1.0 - min_lr_factor) * (cycle_position * 2)
            else:
                # Descending
                lr_factor = min_lr_factor + (1.0 - min_lr_factor) * (2 - cycle_position * 2)
            
            return lr_factor
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'polynomial':
        # ====================================================================
        # POLYNOMIAL DECAY
        # ====================================================================
        # Smoother alternative to cosine decay
        # Power controls decay rate (higher = steeper)
        # ====================================================================
        power = config['training'].get('polynomial_power', 1.0)
        
        print(f"  Polynomial power: {power}")
        print(f"{'='*70}\n")
        
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Polynomial decay
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, (1.0 - progress) ** power)
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'cosine':
        # ====================================================================
        # STANDARD COSINE DECAY (Original)
        # ====================================================================
        print(f"{'='*70}\n")
        
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'linear':
        # ====================================================================
        # LINEAR DECAY (Original)
        # ====================================================================
        print(f"{'='*70}\n")
        
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                       f"Supported: cosine, cosine_restarts, inverse_sqrt, triangular, polynomial, linear")


# ============================================================================
# BACKWARD COMPATIBILITY - Original get_lr_scheduler still works
# ============================================================================
# The enhanced version is backward compatible with original configs
# If config doesn't specify new parameters, defaults to original behavior
# ============================================================================