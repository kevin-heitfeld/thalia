"""
Synaptic components: weight initialization, short-term plasticity, and traces.

This module provides synaptic mechanisms for spiking neural networks.
"""

from __future__ import annotations

from .neuromodulator_receptor import NeuromodulatorReceptor
from .spillover import (
    SpilloverConfig,
    SpilloverTransmission,
    apply_spillover_to_weights,
)
from .stp import (
    ShortTermPlasticity,
    STPConfig,
    STPType,
    STP_PRESETS,
    STPPreset,
    get_stp_config,
)
from .weight_init import (
    WeightInitializer,
)

__all__ = [
    # Neuromodulator Receptors
    "NeuromodulatorReceptor",
    # Spillover
    "SpilloverConfig",
    "SpilloverTransmission",
    "apply_spillover_to_weights",
    # Weight Initialization
    "WeightInitializer",
    # Short-term plasticity
    "ShortTermPlasticity",
    "STPConfig",
    "STPType",
    # STP presets
    "STP_PRESETS",
    "STPPreset",
    "get_stp_config",
]
