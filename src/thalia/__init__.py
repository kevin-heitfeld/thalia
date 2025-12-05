"""
THALIA - Thinking Architecture via Learning Integrated Attractors

A framework for building genuinely thinking spiking neural networks.

Main entry point:
    from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
    
    brain = EventDrivenBrain(EventDrivenBrainConfig(
        input_size=256,
        cortex_size=128,
        hippocampus_size=64,
        pfc_size=32,
        n_actions=2,
    ))
"""

__version__ = "0.2.0"

# Brain region modules (biologically-specialized learning)
from thalia import regions  # noqa: F401

# Core brain system (recommended)
from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig

__all__ = [
    "regions",
    "EventDrivenBrain",
    "EventDrivenBrainConfig",
]
