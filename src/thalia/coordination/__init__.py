"""Brain Coordination.

Oscillatory dynamics, trial coordination, and growth management.
"""

from thalia.coordination.oscillator import (
    OscillatorManager,
    BrainOscillator,
    OscillatorConfig,
    SinusoidalOscillator,
    OscillatorCoupling,
)
from thalia.coordination.growth import (
    GrowthManager,
    GrowthEvent,
    CapacityMetrics,
    GrowthCoordinator,
)

__all__ = [
    # Oscillator
    "OscillatorManager",
    "BrainOscillator",
    "OscillatorConfig",
    "SinusoidalOscillator",
    "OscillatorCoupling",
    # Growth
    "GrowthManager",
    "GrowthEvent",
    "CapacityMetrics",
    "GrowthCoordinator",
]
