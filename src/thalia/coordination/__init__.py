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
from thalia.coordination.trial_coordinator import TrialCoordinator
from thalia.coordination.growth import (
    GrowthManager,
    GrowthEvent,
    CapacityMetrics,
)

__all__ = [
    # Oscillator
    "OscillatorManager",
    "BrainOscillator",
    "OscillatorConfig",
    "SinusoidalOscillator",
    "OscillatorCoupling",
    # Trial Coordinator
    "TrialCoordinator",
    # Growth
    "GrowthManager",
    "GrowthEvent",
    "CapacityMetrics",
]
