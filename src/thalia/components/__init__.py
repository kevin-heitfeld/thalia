"""
Neural components: neurons, synapses, and spike coding.

This module provides the fundamental building blocks for constructing
spiking neural networks in Thalia.
"""

from __future__ import annotations

# Spike coding
from .coding import (
    CodingStrategy,
    RateDecoder,
    RateEncoder,
    SpikeCodingConfig,
    SpikeDecoder,
    SpikeEncoder,
    compute_firing_rate,
    compute_spike_count,
    compute_spike_density,
    compute_spike_similarity,
    is_saturated,
    is_silent,
)

# Neuron models
from .neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
    DendriticBranch,
    DendriticBranchConfig,
    DendriticNeuron,
    DendriticNeuronConfig,
    compute_branch_selectivity,
    create_clustered_input,
    create_cortical_layer_neurons,
    create_pyramidal_neurons,
    create_relay_neurons,
    create_scattered_input,
    create_trn_neurons,
)

# Synaptic mechanisms
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
    list_presets,
    update_trace,
)

__all__ = [
    # Neuron models
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Neuron factory functions
    "create_pyramidal_neurons",
    "create_relay_neurons",
    "create_trn_neurons",
    "create_cortical_layer_neurons",
    # Dendritic computation
    "DendriticBranch",
    "DendriticBranchConfig",
    "DendriticNeuron",
    "DendriticNeuronConfig",
    "compute_branch_selectivity",
    "create_clustered_input",
    "create_scattered_input",
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
    # Spike Traces
    "update_trace",
    # Spike Coding
    "CodingStrategy",
    "SpikeCodingConfig",
    "SpikeEncoder",
    "SpikeDecoder",
    "RateEncoder",
    "RateDecoder",
    "compute_spike_similarity",
    # Spike Utils
    "compute_firing_rate",
    "compute_spike_count",
    "compute_spike_density",
    "is_silent",
    "is_saturated",
]
