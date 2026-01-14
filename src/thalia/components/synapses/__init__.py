"""
Synaptic components: weight initialization, short-term plasticity, and traces.

This module provides synaptic mechanisms for spiking neural networks.
"""

from thalia.components.synapses.weight_init import (
    InitStrategy,
    WeightInitializer,
)
from thalia.components.synapses.stp import (
    ShortTermPlasticity,
    STPConfig,
    STPType,
    STPSynapse,
)
from thalia.components.synapses.stp_presets import (
    STP_PRESETS,
    STPPreset,
    get_stp_config,
    list_presets,
    sample_heterogeneous_stp_params,
    create_heterogeneous_stp_configs,
)
from thalia.components.synapses.traces import (
    SpikeTrace,
    PairedTraces,
    TraceConfig,
    compute_stdp_update,
    create_trace,
    update_trace,
    compute_decay,
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
    "SpikeTrace",
    "PairedTraces",
    "TraceConfig",
    "compute_stdp_update",
    "create_trace",
    "update_trace",
    "compute_decay",
]
