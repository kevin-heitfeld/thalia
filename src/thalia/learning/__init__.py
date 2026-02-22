"""Learning strategies and eligibility trace management for synaptic plasticity."""

from __future__ import annotations

from .eligibility_trace_manager import EligibilityTraceManager
from .intrinsic_plasticity import (
    compute_excitability_modulation,
)
from .strategies import (
    BCMConfig,
    BCMStrategy,
    LearningConfig,
    STDPConfig,
    STDPStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
    LearningStrategy,
    CompositeStrategy,
)

__all__ = [
    # Eligibility Traces
    "EligibilityTraceManager",
    # Intrinsic Plasticity
    "compute_excitability_modulation",
    # Learning Strategies
    "LearningConfig",
    "BCMConfig",
    "STDPConfig",
    "ThreeFactorConfig",
    "STDPStrategy",
    "BCMStrategy",
    "ThreeFactorStrategy",
    "LearningStrategy",
    "CompositeStrategy",
]
