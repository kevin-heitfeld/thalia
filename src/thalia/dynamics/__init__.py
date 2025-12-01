"""
Attractor dynamics, simulation infrastructure, and manifold representations.

This module provides:
- Attractor networks for pattern storage and recall
- Core simulation infrastructure (NetworkState, NetworkConfig, forward_timestep)
- Input pattern generation (create_temporal_pattern)
- Activity tracking and trajectory analysis
- Short-term synaptic plasticity (STD, STF, NMDA, dendritic saturation, neuromodulation)
"""

from .attractor import AttractorNetwork, AttractorConfig
from .manifold import ActivityTracker, ThoughtTrajectory
from .simulation import (
    NetworkState,
    NetworkConfig,
    forward_timestep,
    forward_timestep_with_stp,
    forward_pattern,
    select_device,
)

from .patterns import create_temporal_pattern, create_poisson_pattern
from .synaptic import (
    STPConfig,
    ShortTermPlasticity,
    NMDAConfig,
    NMDAGating,
    DendriticConfig,
    DendriticSaturation,
    NeuromodulationConfig,
    Neuromodulation,
    create_synaptic_mechanisms,
)

__all__ = [
    # Attractor dynamics
    "AttractorNetwork",
    "AttractorConfig",
    "ActivityTracker",
    "ThoughtTrajectory",
    # Simulation infrastructure
    "NetworkState",
    "NetworkConfig",
    "forward_timestep",
    "forward_timestep_with_stp",
    "forward_pattern",
    "select_device",
    # Pattern generation
    "create_temporal_pattern",
    "create_poisson_pattern",
    # Short-term synaptic plasticity
    "STPConfig",
    "ShortTermPlasticity",
    "NMDAConfig",
    "NMDAGating",
    "DendriticConfig",
    "DendriticSaturation",
    "NeuromodulationConfig",
    "Neuromodulation",
    "create_synaptic_mechanisms",
]
