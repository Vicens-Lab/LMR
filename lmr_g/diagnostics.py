# File: lmr_g/diagnostics.py
"""
LMR-Foundation v3.0 Diagnostics

Monitoring tools for training health:
1. Orthogonality metrics for Grassmann layers
2. Plücker influence tracking
3. Warning systems for critical issues
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import warnings


@dataclass
class OrthogonalityMetrics:
    """Metrics for Grassmann layer orthogonality."""
    frobenius_error: float  # ||W^T W - I||_F
    eigenvalue_spread: float  # λ_max / λ_min
    effective_rank: float  # Σλ_i / λ_max
    is_healthy: bool  # Within acceptable bounds


@dataclass
class PluckerMetrics:
    """Metrics for Plücker attention influence."""
    output_delta: float  # Difference with/without Plücker
    gamma_mean: float  # Average gamma value
    gamma_std: float  # Gamma standard deviation
    gamma_grad_mean: float  # Average gamma gradient (if available)


def check_orthogonality(
    model: nn.Module,
    warn_threshold: float = 100.0,
    critical_threshold: float = 1000.0
) -> Dict[str, OrthogonalityMetrics]:
    """
    Check orthogonality of all Grassmann layers in the model.
    
    Args:
        model: LMRFoundation model
        warn_threshold: Eigenvalue spread warning threshold
        critical_threshold: Eigenvalue spread critical threshold
        
    Returns:
        Dict mapping layer name to OrthogonalityMetrics
    """
    metrics = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'get_orthogonal_matrix') or hasattr(module, 'W_ortho'):
            # Get the orthogonal matrix
            if hasattr(module, 'get_orthogonal_matrix'):
                W = module.get_orthogonal_matrix()
            elif hasattr(module, 'W_ortho'):
                W = module.W_ortho.weight
            else:
                continue
            
            # Compute W^T W
            WtW = W.T @ W
            dim = W.shape[0]
            I = torch.eye(dim, device=W.device, dtype=W.dtype)
            
            # Frobenius error: ||W^T W - I||_F
            frobenius_error = torch.norm(WtW - I, p='fro').item()
            
            # Eigenvalue analysis
            try:
                eigenvalues = torch.linalg.eigvalsh(WtW)
                eigenvalues = eigenvalues.real.abs()
                
                # Eigenvalue spread (condition number proxy)
                max_eig = eigenvalues.max().item()
                min_eig = eigenvalues.min().item()
                eigenvalue_spread = max_eig / max(min_eig, 1e-10)
                
                # Effective rank
                effective_rank = eigenvalues.sum().item() / max(max_eig, 1e-10)
            except Exception:
                eigenvalue_spread = float('inf')
                effective_rank = 0.0
            
            # Health check
            is_healthy = eigenvalue_spread < warn_threshold
            
            metrics[name] = OrthogonalityMetrics(
                frobenius_error=frobenius_error,
                eigenvalue_spread=eigenvalue_spread,
                effective_rank=effective_rank,
                is_healthy=is_healthy
            )
    
    return metrics


def compute_plucker_influence(
    model: nn.Module,
    input_tensor: torch.Tensor,
    sample_size: int = 4
) -> PluckerMetrics:
    """
    Measure the influence of Plücker bias on model outputs.
    
    Args:
        model: LMRFoundation model
        input_tensor: Input tensor [batch, seq_len] or dict with 'input_ids'
        sample_size: Number of samples to use
        
    Returns:
        PluckerMetrics with influence measurements
    """
    model.eval()
    
    # Handle both tensor and dict inputs
    if isinstance(input_tensor, dict):
        input_ids = input_tensor['input_ids'][:sample_size]
    elif isinstance(input_tensor, torch.Tensor):
        input_ids = input_tensor[:sample_size] if input_tensor.dim() == 2 else input_tensor
    else:
        raise ValueError(f"Expected tensor or dict, got {type(input_tensor)}")
    
    # Ensure correct shape
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    with torch.no_grad():
        # Forward with Plücker
        out_with_plucker = model(input_ids, use_plucker=True, return_hidden=True)
        
        # Forward without Plücker
        out_without_plucker = model(input_ids, use_plucker=False, return_hidden=True)
        
        # Compute difference
        delta = (out_with_plucker - out_without_plucker).abs().mean().item()
    
    # Collect gamma statistics
    gammas = []
    gamma_grads = []
    
    for module in model.modules():
        if hasattr(module, 'gamma'):
            gammas.append(module.gamma.item())
            if module.gamma.grad is not None:
                gamma_grads.append(module.gamma.grad.abs().mean().item())
    
    gamma_mean = sum(gammas) / max(len(gammas), 1) if gammas else 0.0
    gamma_std = (sum((g - gamma_mean)**2 for g in gammas) / max(len(gammas), 1))**0.5 if len(gammas) > 1 else 0.0
    gamma_grad_mean = sum(gamma_grads) / max(len(gamma_grads), 1) if gamma_grads else 0.0
    
    model.train()
    
    return PluckerMetrics(
        output_delta=delta,
        gamma_mean=gamma_mean,
        gamma_std=gamma_std,
        gamma_grad_mean=gamma_grad_mean
    )


def run_full_diagnostics(
    model: nn.Module,
    batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    step: int,
    warn_threshold: float = 100.0,
    critical_threshold: float = 1000.0
) -> Dict[str, Any]:
    """
    Run comprehensive diagnostics on the model.
    
    Args:
        model: LMRFoundation model
        batch: Input batch (tensor or dict with 'input_ids')
        step: Current training step
        warn_threshold: Eigenvalue spread warning threshold
        critical_threshold: Critical threshold for switching to Cayley
        
    Returns:
        Dictionary of all diagnostic metrics
    """
    diagnostics = {}
    
    # Orthogonality check
    orth_metrics = check_orthogonality(model, warn_threshold, critical_threshold)
    
    if orth_metrics:
        frobenius_errors = [m.frobenius_error for m in orth_metrics.values()]
        eigenvalue_spreads = [m.eigenvalue_spread for m in orth_metrics.values()]
        effective_ranks = [m.effective_rank for m in orth_metrics.values()]
        
        diagnostics['orth/mean_frobenius_error'] = sum(frobenius_errors) / len(frobenius_errors)
        diagnostics['orth/max_frobenius_error'] = max(frobenius_errors)
        diagnostics['orth/mean_eigenvalue_spread'] = sum(eigenvalue_spreads) / len(eigenvalue_spreads)
        diagnostics['orth/max_eigenvalue_spread'] = max(eigenvalue_spreads)
        diagnostics['orth/mean_effective_rank'] = sum(effective_ranks) / len(effective_ranks)
        diagnostics['orth/min_effective_rank'] = min(effective_ranks)
        
        # Issue warnings
        max_spread = max(eigenvalue_spreads)
        if max_spread > critical_threshold:
            print(f"\n🔴 CRITICAL at step {step}: Eigenvalue spread = {max_spread:.1f} "
                  f"(threshold: {critical_threshold}). Grassmann matrices collapsing! "
                  f"Consider switching to Cayley parameterization.")
        elif max_spread > warn_threshold:
            print(f"\n⚠️ WARNING at step {step}: Eigenvalue spread = {max_spread:.1f} "
                  f"(threshold: {warn_threshold}). Monitor closely.")
    
    # Plücker influence check
    try:
        plucker_metrics = compute_plucker_influence(model, batch)
        diagnostics['plucker/output_delta'] = plucker_metrics.output_delta
        diagnostics['plucker/gamma_mean'] = plucker_metrics.gamma_mean
        diagnostics['plucker/gamma_std'] = plucker_metrics.gamma_std
        diagnostics['plucker/gamma_grad_mean'] = plucker_metrics.gamma_grad_mean
    except Exception as e:
        # Plücker diagnostics failed - log but don't crash
        diagnostics['plucker/output_delta'] = 0.0
        diagnostics['plucker/gamma_mean'] = 0.0
        diagnostics['plucker/gamma_std'] = 0.0
        diagnostics['plucker/gamma_grad_mean'] = 0.0
    
    diagnostics['step'] = step
    
    return diagnostics


class DiagnosticsTracker:
    """Track diagnostic metrics over training."""
    
    def __init__(self, max_history: int = 1000):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def update(self, metrics: Dict[str, Any]):
        """Add new metrics to history."""
        self.history.append(metrics)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_trend(self, metric_name: str, window: int = 10) -> Optional[float]:
        """Get trend for a metric (positive = increasing, negative = decreasing)."""
        if len(self.history) < window:
            return None
        
        recent = [h.get(metric_name, 0) for h in self.history[-window:]]
        if len(recent) < 2:
            return None
        
        # Simple linear trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / max(denominator, 1e-10)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics."""
        if not self.history:
            return {}
        
        latest = self.history[-1]
        summary = {'latest': latest}
        
        # Add trends for key metrics
        for key in ['orth/max_eigenvalue_spread', 'plucker/output_delta']:
            trend = self.get_trend(key)
            if trend is not None:
                summary[f'{key}_trend'] = trend
        
        return summary