"""
Neural components: neurons, synapses, and spike coding.

This module provides the fundamental building blocks for constructing
spiking neural networks in Thalia.
"""

# Neuron models and constants
from thalia.components.neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
    TAU_MEM_STANDARD,
    TAU_MEM_FAST,
    TAU_MEM_SLOW,
    TAU_SYN_EXCITATORY,
    TAU_SYN_INHIBITORY,
    TAU_SYN_NMDA,
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    V_REST_STANDARD,
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
    G_LEAK_STANDARD,
    G_LEAK_FAST,
    G_LEAK_SLOW,
    ADAPT_INCREMENT_NONE,
    ADAPT_INCREMENT_MODERATE,
    ADAPT_INCREMENT_STRONG,
    ADAPT_INCREMENT_CORTEX_L23,
    STANDARD_PYRAMIDAL,
    FAST_SPIKING_INTERNEURON,
    create_pyramidal_neurons,
    create_relay_neurons,
    create_trn_neurons,
    create_cortical_layer_neurons,
    DendriticBranch,
    DendriticBranchConfig,
    DendriticNeuron,
    DendriticNeuronConfig,
    compute_branch_selectivity,
    create_clustered_input,
    create_scattered_input,
)

# Synaptic mechanisms
from thalia.components.synapses import (
    InitStrategy,
    WeightInitializer,
    ShortTermPlasticity,
    STPConfig,
    STPType,
    STPSynapse,
    STP_PRESETS,
    STPPreset,
    get_stp_config,
    list_presets,
    SpikeTrace,
    PairedTraces,
    TraceConfig,
    compute_stdp_update,
    create_trace,
    update_trace,
    compute_decay,
)

# Spike coding
from thalia.components.coding import (
    CodingStrategy,
    SpikeCodingConfig,
    SpikeEncoder,
    SpikeDecoder,
    RateEncoder,
    RateDecoder,
    compute_spike_similarity,
    compute_firing_rate,
    compute_spike_count,
    compute_spike_density,
    is_silent,
    is_saturated,
)

__all__ = [
    # Neuron models
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Neuron constants
    "TAU_MEM_STANDARD",
    "TAU_MEM_FAST",
    "TAU_MEM_SLOW",
    "TAU_SYN_EXCITATORY",
    "TAU_SYN_INHIBITORY",
    "TAU_SYN_NMDA",
    "V_THRESHOLD_STANDARD",
    "V_RESET_STANDARD",
    "V_REST_STANDARD",
    "E_LEAK",
    "E_EXCITATORY",
    "E_INHIBITORY",
    "G_LEAK_STANDARD",
    "G_LEAK_FAST",
    "G_LEAK_SLOW",
    "ADAPT_INCREMENT_NONE",
    "ADAPT_INCREMENT_MODERATE",
    "ADAPT_INCREMENT_STRONG",
    "ADAPT_INCREMENT_CORTEX_L23",
    "STANDARD_PYRAMIDAL",
    "FAST_SPIKING_INTERNEURON",
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
    "SpikeTrace",
    "PairedTraces",
    "TraceConfig",
    "compute_stdp_update",
    "create_trace",
    "update_trace",
    "compute_decay",
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
