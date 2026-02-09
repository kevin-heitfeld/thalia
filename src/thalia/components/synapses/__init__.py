"""
Synaptic components: weight initialization, short-term plasticity, and traces.

This module provides synaptic mechanisms for spiking neural networks.
"""

from __future__ import annotations

from .spillover import (
    SpilloverConfig,
    SpilloverTransmission,
    apply_spillover_to_weights,
)
from .stp import (
    ShortTermPlasticity,
    STPConfig,
    STPSynapse,
    STPType,
    STP_PRESETS,
    STPPreset,
    create_heterogeneous_stp_configs,
    get_stp_config,
    list_presets,
)
from .traces import (
    update_trace,
)
from .weight_init import (
    InitStrategy,
    WeightInitializer,
)

__all__ = [
    # Spillover
    "SpilloverConfig",
    "SpilloverTransmission",
    "apply_spillover_to_weights",
    # Weight Initialization
    "InitStrategy",
    "WeightInitializer",
    # Short-term plasticity
    "ShortTermPlasticity",
    "STPConfig",
    "STPType",
    "STPSynapse",
    # STP presets
    "STP_PRESETS",
    "STPPreset",
    "get_stp_config",
    "list_presets",
    "create_heterogeneous_stp_configs",
    # Spike Traces
    "update_trace",
]
