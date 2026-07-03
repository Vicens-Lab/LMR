# File: training/schedulers.py
"""
LMR-Foundation Training Schedulers

1. GammaWarmupScheduler - Gradually increases Plücker gate γ from 0 to target
2. CurriculumScheduler - Manages 3-phase training curriculum for proxy tasks
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class CurriculumPhase:
    """Definition of a curriculum training phase."""
    name: str
    start_step: int
    end_step: int
    task_weights: Dict[str, float]


class GammaWarmupScheduler:
    """
    Gradually increase Plücker gate γ from 0 to target.
    
    Rationale: Let standard attention stabilize first,
    then gradually introduce geometric bias.
    
    Usage:
        scheduler = GammaWarmupScheduler(model, target_gamma=1e-3, warmup_steps=5000)
        
        for step in range(total_steps):
            gamma = scheduler.step(step)
            # ... training step ...
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_gamma: float = 1e-3,
        warmup_steps: int = 5000,
        warmup_start: int = 0
    ):
        self.model = model
        self.target_gamma = target_gamma
        self.warmup_steps = warmup_steps
        self.warmup_start = warmup_start
        
        # Collect all gamma parameters
        self.gamma_params: List[nn.Parameter] = []
        for module in model.modules():
            if hasattr(module, 'gamma') and isinstance(module.gamma, nn.Parameter):
                self.gamma_params.append(module.gamma)
        
        if len(self.gamma_params) == 0:
            print("⚠️ GammaWarmupScheduler: No gamma parameters found in model")
        else:
            print(f"✓ GammaWarmupScheduler initialized:")
            print(f"    Found {len(self.gamma_params)} gamma parameters")
            print(f"    Target γ: {target_gamma}")
            print(f"    Warmup steps: {warmup_steps}")
            print(f"    Warmup start: {warmup_start}")
    
    def step(self, current_step: int) -> float:
        """
        Update gamma based on current training step.
        
        Args:
            current_step: Current global training step
            
        Returns:
            Current gamma value
        """
        if current_step < self.warmup_start:
            # Before warmup starts, keep at 0
            gamma = 0.0
        elif current_step >= self.warmup_start + self.warmup_steps:
            # After warmup, use target value
            gamma = self.target_gamma
        else:
            # Linear warmup
            progress = (current_step - self.warmup_start) / self.warmup_steps
            gamma = self.target_gamma * progress
        
        # Update all gamma parameters
        for param in self.gamma_params:
            param.data.fill_(gamma)
        
        return gamma
    
    def get_gamma(self) -> float:
        """Get current gamma value."""
        if self.gamma_params:
            return self.gamma_params[0].item()
        return 0.0


class CurriculumScheduler:
    """
    Manages 3-phase training curriculum for proxy tasks.
    
    Phase 1 (0-30%): MLM only - learn basic representations
    Phase 2 (30-60%): Add span masking - learn local structure
    Phase 3 (60-100%): Add stem-span masking - learn long-range dependencies
    
    Usage:
        scheduler = CurriculumScheduler(total_steps=100000)
        
        for step in range(total_steps):
            weights = scheduler.get_task_weights(step)
            loss = (weights['mlm'] * mlm_loss + 
                    weights['span'] * span_loss +
                    weights['stem_span'] * stem_span_loss)
    """
    
    def __init__(
        self,
        total_steps: int,
        phase1_ratio: float = 0.3,
        phase2_ratio: float = 0.3,
        phase3_ratio: float = 0.4,
        # Final task weights
        mlm_weight: float = 0.5,
        span_weight: float = 0.3,
        stem_span_weight: float = 0.2,
    ):
        self.total_steps = total_steps
        
        # Calculate phase boundaries
        phase1_end = int(total_steps * phase1_ratio)
        phase2_end = int(total_steps * (phase1_ratio + phase2_ratio))
        
        self.phases = [
            CurriculumPhase(
                name="mlm_only",
                start_step=0,
                end_step=phase1_end,
                task_weights={'mlm': 1.0, 'span': 0.0, 'stem_span': 0.0}
            ),
            CurriculumPhase(
                name="add_span",
                start_step=phase1_end,
                end_step=phase2_end,
                task_weights={'mlm': 0.7, 'span': 0.3, 'stem_span': 0.0}
            ),
            CurriculumPhase(
                name="full_curriculum",
                start_step=phase2_end,
                end_step=total_steps,
                task_weights={'mlm': mlm_weight, 'span': span_weight, 'stem_span': stem_span_weight}
            ),
        ]
        
        self.final_weights = {'mlm': mlm_weight, 'span': span_weight, 'stem_span': stem_span_weight}
        
        print("\n" + "="*70)
        print("Curriculum Scheduler Initialized")
        print("="*70)
        for phase in self.phases:
            print(f"  {phase.name}: steps {phase.start_step:,} - {phase.end_step:,}")
            print(f"    Weights: {phase.task_weights}")
        print("="*70 + "\n")
    
    def get_current_phase(self, step: int) -> CurriculumPhase:
        """Get the phase for current step."""
        for phase in self.phases:
            if phase.start_step <= step < phase.end_step:
                return phase
        # Default to final phase if step exceeds total
        return self.phases[-1]
    
    def get_task_weights(self, step: int) -> Dict[str, float]:
        """
        Get task weights with smooth interpolation between phases.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary of task weights
        """
        current_phase = self.get_current_phase(step)
        
        # Find next phase (if exists)
        current_idx = self.phases.index(current_phase)
        
        if current_idx == len(self.phases) - 1:
            # Last phase - no interpolation
            return current_phase.task_weights.copy()
        
        next_phase = self.phases[current_idx + 1]
        
        # Calculate interpolation progress within current phase
        phase_progress = (step - current_phase.start_step) / (current_phase.end_step - current_phase.start_step)
        
        # Only interpolate in the last 10% of each phase for smooth transition
        if phase_progress < 0.9:
            return current_phase.task_weights.copy()
        
        # Smooth interpolation
        interp_progress = (phase_progress - 0.9) / 0.1
        
        weights = {}
        for task in ['mlm', 'span', 'stem_span']:
            current_w = current_phase.task_weights.get(task, 0.0)
            next_w = next_phase.task_weights.get(task, 0.0)
            weights[task] = current_w + interp_progress * (next_w - current_w)
        
        return weights
    
    def get_phase_name(self, step: int) -> str:
        """Get name of current phase."""
        return self.get_current_phase(step).name
    
    def should_enable_task(self, step: int, task: str) -> bool:
        """Check if a specific task should be enabled at current step."""
        weights = self.get_task_weights(step)
        return weights.get(task, 0.0) > 0.0


class OrthogonalityPenaltyScheduler:
    """
    Schedules orthogonality penalty weight during training.
    
    Can optionally decay the penalty over time as the model stabilizes.
    
    Usage:
        scheduler = OrthogonalityPenaltyScheduler(initial_weight=0.01)
        
        for step in range(total_steps):
            penalty_weight = scheduler.get_weight(step)
            loss = task_loss + penalty_weight * orth_penalty
    """
    
    def __init__(
        self,
        initial_weight: float = 0.01,
        decay_steps: Optional[int] = None,
        final_weight: float = 0.001,
        warmup_steps: int = 1000
    ):
        self.initial_weight = initial_weight
        self.decay_steps = decay_steps
        self.final_weight = final_weight
        self.warmup_steps = warmup_steps
        
        print(f"✓ OrthogonalityPenaltyScheduler initialized:")
        print(f"    Initial weight: {initial_weight}")
        if decay_steps:
            print(f"    Decay over: {decay_steps} steps")
            print(f"    Final weight: {final_weight}")
    
    def get_weight(self, step: int) -> float:
        """Get penalty weight for current step."""
        # Warmup phase
        if step < self.warmup_steps:
            return self.initial_weight * (step / self.warmup_steps)
        
        # No decay configured
        if self.decay_steps is None:
            return self.initial_weight
        
        # Decay phase
        decay_start = self.warmup_steps
        if step >= decay_start + self.decay_steps:
            return self.final_weight
        
        # Linear decay
        progress = (step - decay_start) / self.decay_steps
        return self.initial_weight - progress * (self.initial_weight - self.final_weight)


class CombinedScheduler:
    """
    Combines all LMR-Foundation schedulers for easy management.
    
    Usage:
        scheduler = CombinedScheduler(model, config)
        
        for step in range(total_steps):
            scheduler.step(step)
            
            # Get current values
            weights = scheduler.get_task_weights()
            gamma = scheduler.get_gamma()
            ortho_weight = scheduler.get_ortho_weight()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        total_steps: int
    ):
        # Gamma warmup (if Plücker enabled)
        self.gamma_scheduler = None
        if getattr(config, 'use_plucker_bias', False):
            self.gamma_scheduler = GammaWarmupScheduler(
                model,
                target_gamma=config.plucker_target_gamma,
                warmup_steps=config.plucker_warmup_steps
            )
        
        # Curriculum scheduler (if enabled)
        self.curriculum_scheduler = None
        if getattr(config, 'use_curriculum', False):
            phase1_steps = getattr(config, 'curriculum_phase1_steps', int(total_steps * 0.3))
            phase2_steps = getattr(config, 'curriculum_phase2_steps', int(total_steps * 0.3))
            
            self.curriculum_scheduler = CurriculumScheduler(
                total_steps=total_steps,
                phase1_ratio=phase1_steps / total_steps,
                phase2_ratio=phase2_steps / total_steps,
                mlm_weight=config.mlm_weight,
                span_weight=config.span_weight,
                stem_span_weight=config.stem_span_weight
            )
        
        # Orthogonality penalty scheduler (if Grassmann enabled)
        self.ortho_scheduler = None
        if getattr(config, 'use_grassmann', False):
            self.ortho_scheduler = OrthogonalityPenaltyScheduler(
                initial_weight=config.ortho_penalty_weight
            )
        
        self.current_step = 0
    
    def step(self, step: int):
        """Update all schedulers to current step."""
        self.current_step = step
        
        if self.gamma_scheduler is not None:
            self.gamma_scheduler.step(step)
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        if self.curriculum_scheduler is not None:
            return self.curriculum_scheduler.get_task_weights(self.current_step)
        return {'mlm': 1.0, 'span': 0.0, 'stem_span': 0.0}
    
    def get_gamma(self) -> float:
        """Get current gamma value."""
        if self.gamma_scheduler is not None:
            return self.gamma_scheduler.get_gamma()
        return 0.0
    
    def get_ortho_weight(self) -> float:
        """Get current orthogonality penalty weight."""
        if self.ortho_scheduler is not None:
            return self.ortho_scheduler.get_weight(self.current_step)
        return 0.0
    
    def get_phase_name(self) -> str:
        """Get current curriculum phase name."""
        if self.curriculum_scheduler is not None:
            return self.curriculum_scheduler.get_phase_name(self.current_step)
        return "standard"
    
    def get_all_values(self) -> Dict[str, Any]:
        """Get all scheduler values for logging."""
        return {
            'gamma': self.get_gamma(),
            'ortho_weight': self.get_ortho_weight(),
            'task_weights': self.get_task_weights(),
            'phase': self.get_phase_name(),
            'step': self.current_step
        }