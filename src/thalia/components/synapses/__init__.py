"""
Synaptic components: weight initialization, short-term plasticity, and traces.

This module provides synaptic mechanisms for spiking neural networks.
"""

from __future__ import annotations

from .stp import (
    ShortTermPlasticity,
    STPConfig,
    STPSynapse,
    STPType,
)
from .stp_presets import (
    STP_PRESETS,
    STPPreset,
    create_heterogeneous_stp_configs,
    get_stp_config,
    list_presets,
    sample_heterogeneous_stp_params,
)
from .traces import (
    update_trace,
)
from .weight_init import (
    InitStrategy,
    WeightInitializer,
)

__all__ = [
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
    "sample_heterogeneous_stp_params",
    "create_heterogeneous_stp_configs",
    # Spike Traces
    "update_trace",
]
