"""Learning strategies and eligibility trace management for synaptic plasticity."""

from __future__ import annotations

from .eligibility_trace_manager import (
    EligibilityTraceConfig,
    EligibilityTraceManager,
)
from .strategies import (
    LearningConfig,
    LearningStrategy,
    BCMConfig,
    BCMStrategy,
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
    MaIConfig,
    MaIStrategy,
    MetaplasticityConfig,
    MetaplasticityStrategy,
    PredictiveCodingConfig,
    PredictiveCodingStrategy,
    STDPConfig,
    D1D2STDPConfig,
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
    "EligibilityTraceConfig",
    "EligibilityTraceManager",
    # Learning Strategies
    "LearningConfig",
    "LearningStrategy",
    "BCMConfig",
    "BCMStrategy",
    "InhibitorySTDPConfig",
    "InhibitorySTDPStrategy",
    "MaIConfig",
    "MaIStrategy",
    "MetaplasticityConfig",
    "MetaplasticityStrategy",
    "PredictiveCodingConfig",
    "PredictiveCodingStrategy",
    "STDPConfig",
    "D1D2STDPConfig",
    "STDPStrategy",
    "D1STDPStrategy",
    "D2STDPStrategy",
    "ThreeFactorConfig",
    "ThreeFactorStrategy",
    "TagAndCaptureConfig",
    "TagAndCaptureStrategy",
    "CompositeStrategy",
]
