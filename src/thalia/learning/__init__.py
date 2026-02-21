"""Learning strategies and eligibility trace management for synaptic plasticity."""

from __future__ import annotations

from .eligibility_trace_manager import EligibilityTraceManager
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
from .unified_homeostasis import (
    UnifiedHomeostasis,
)

__all__ = [
    # Eligibility Traces
    "EligibilityTraceManager",
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
    # Unified Homeostasis
    "UnifiedHomeostasis",
]
