"""Learning strategies and eligibility trace management for synaptic plasticity."""

from __future__ import annotations

from .eligibility_trace_manager import (
    EligibilityTraceManager,
)
from .intrinsic_plasticity import (
    compute_excitability_modulation,
)
from .strategies import (
    LearningConfig,
    LearningStrategy,
    BCMConfig,
    BCMStrategy,
    MaIConfig,
    MaIStrategy,
    PredictiveCodingConfig,
    PredictiveCodingStrategy,
    STDPConfig,
    D1STDPConfig,
    D2STDPConfig,
    STDPStrategy,
    D1STDPStrategy,
    D2STDPStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
    TagAndCaptureConfig,
    TagAndCaptureStrategy,
    CompositeStrategy,
)

__all__ = [
    # Eligibility Traces
    "EligibilityTraceManager",
    # Intrinsic Plasticity
    "compute_excitability_modulation",
    # Learning Strategies
    "LearningConfig",
    "LearningStrategy",
    "BCMConfig",
    "BCMStrategy",
    "MaIConfig",
    "MaIStrategy",
    "PredictiveCodingConfig",
    "PredictiveCodingStrategy",
    "STDPConfig",
    "D1STDPConfig",
    "D2STDPConfig",
    "STDPStrategy",
    "D1STDPStrategy",
    "D2STDPStrategy",
    "ThreeFactorConfig",
    "ThreeFactorStrategy",
    "TagAndCaptureConfig",
    "TagAndCaptureStrategy",
    "CompositeStrategy",
]
