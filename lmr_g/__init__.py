# File: lmr_foundation/__init__.py
"""
LMR-Foundation v3.0 Package

Geometric-biased RNA foundation model with:
- Plücker-biased attention
- Grassmann-window layers
- Curriculum training
"""

from .config_foundation import LMRFoundationConfig, estimate_foundation_params
from .architecture_foundation import (
    LMRFoundation,
    LMRFoundationBlock,
    PluckerBiasedAttention,
    GrassmannWindowLayer,
    create_lmr_foundation,
)
from .diagnostics import (
    check_orthogonality,
    compute_plucker_influence,
    run_full_diagnostics,
    DiagnosticsTracker,
    OrthogonalityMetrics,
    PluckerMetrics,
)
from .schedulers import (
    GammaWarmupScheduler,
    CurriculumScheduler,
    OrthogonalityPenaltyScheduler,
    CombinedScheduler,
)
from .stem_span_masking import (
    StemSpanMasker,
    create_stem_span_aware_collator,
)

__version__ = "3.0.0"
__all__ = [
    # Config
    "LMRFoundationConfig",
    "estimate_foundation_params",
    # Architecture
    "LMRFoundation",
    "LMRFoundationBlock",
    "PluckerBiasedAttention",
    "GrassmannWindowLayer",
    "create_lmr_foundation",
    # Diagnostics
    "check_orthogonality",
    "compute_plucker_influence",
    "run_full_diagnostics",
    "DiagnosticsTracker",
    "OrthogonalityMetrics",
    "PluckerMetrics",
    # Schedulers
    "GammaWarmupScheduler",
    "CurriculumScheduler",
    "OrthogonalityPenaltyScheduler",
    "CombinedScheduler",
    # Masking
    "StemSpanMasker",
    "create_stem_span_aware_collator",
]