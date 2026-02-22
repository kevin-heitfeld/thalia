"""Learning strategies and eligibility trace management for synaptic plasticity."""

from __future__ import annotations

from .eligibility_trace_manager import (
    EligibilityTraceManager,
)
from .intrinsic_plasticity import (
    compute_excitability_modulation,
)
from .strategies import (
    BCMConfig,
    BCMStrategy,
    LearningConfig,
    STDPConfig,
    D1STDPConfig,
    D2STDPConfig,
    PredictiveCodingConfig,
    STDPStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
    LearningStrategy,
    D1STDPStrategy,
    D2STDPStrategy,
    PredictiveCodingStrategy,
    CompositeStrategy,
)

__all__ = [
    # Eligibility Traces
    "EligibilityTraceManager",
    # Intrinsic Plasticity
    "compute_excitability_modulation",
    # Configurations
    "LearningConfig",
    "BCMConfig",
    "STDPConfig",
    "ThreeFactorConfig",
    "D1STDPConfig",
    "D2STDPConfig",
    "PredictiveCodingConfig",
    # Learning Strategies
    "LearningStrategy",
    "STDPStrategy",
    "BCMStrategy",
    "ThreeFactorStrategy",
    # Striatal MSN strategies
    "D1STDPStrategy",
    "D2STDPStrategy",
    # Cortical predictive coding
    "PredictiveCodingStrategy",
    # Composite strategy for multi-strategy learning
    "CompositeStrategy",
]
