"""Brain Coordination.

Oscillatory dynamics, trial coordination, and growth management.
"""

from __future__ import annotations

from thalia.coordination.growth import (
    CapacityMetrics,
    GrowthEvent,
    GrowthManager,
)
from thalia.coordination.oscillator import (
    BrainOscillator,
    OscillatorConfig,
    OscillatorCoupling,
    OscillatorManager,
    SinusoidalOscillator,
)

__all__ = [
    # Growth
    "CapacityMetrics",
    "GrowthEvent",
    "GrowthManager",
    # Oscillator
    "BrainOscillator",
    "OscillatorConfig",
    "OscillatorCoupling",
    "OscillatorManager",
    "SinusoidalOscillator",
]
