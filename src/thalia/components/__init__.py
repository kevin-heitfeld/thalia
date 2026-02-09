"""
Neural components: neurons, synapses, and spike coding.

This module provides the fundamental building blocks for constructing
spiking neural networks in Thalia.
"""

from __future__ import annotations

from .gap_junctions import (
    GapJunctionConfig,
    GapJunctionCoupling,
)
from .neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
    NeuronFactory,
)
from .spike_coding import (
    CodingStrategy,
    SpikeCodingConfig,
    SpikeDecoder,
    SpikeEncoder,
)
from .synapses import (
    STP_PRESETS,
    InitStrategy,
    ShortTermPlasticity,
    STPConfig,
    STPPreset,
    STPSynapse,
    STPType,
    WeightInitializer,
    get_stp_config,
    create_heterogeneous_stp_configs,
    list_presets,
    update_trace,
)

__all__ = [
    # Neuron models
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Neuron factory
    "NeuronFactory",
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
    "create_heterogeneous_stp_configs",
    "list_presets",
    # Spike Traces
    "update_trace",
    # Spike Coding
    "CodingStrategy",
    "SpikeCodingConfig",
    "SpikeEncoder",
    "SpikeDecoder",
    # Gap Junctions
    "GapJunctionConfig",
    "GapJunctionCoupling",
]
