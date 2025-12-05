"""
THALIA - Thinking Architecture via Learning Integrated Attractors

A framework for building genuinely thinking spiking neural networks.

Main entry point (recommended):
    from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes
    from thalia.core.brain import EventDrivenBrain
    
    config = ThaliaConfig(
        global_=GlobalConfig(device="cpu"),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=256,
                cortex_size=128,
                hippocampus_size=64,
                pfc_size=32,
                n_actions=2,
            ),
        ),
    )
    brain = EventDrivenBrain.from_thalia_config(config)
    
Legacy API (deprecated):
    from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
    brain = EventDrivenBrain(EventDrivenBrainConfig(...))  # Emits DeprecationWarning
"""

__version__ = "0.2.0"

# Unified configuration system (recommended)
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

# Brain region modules (biologically-specialized learning)
from thalia import regions  # noqa: F401

# Core brain system
from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig

__all__ = [
    # Config (recommended)
    "ThaliaConfig",
    "GlobalConfig", 
    "BrainConfig",
    "RegionSizes",
    # Brain
    "regions",
    "EventDrivenBrain",
    "EventDrivenBrainConfig",  # Legacy, deprecated
]
