"""
Neuron models and dendritic computation.

This module contains spiking neuron models and dendritic processing components.
"""

from __future__ import annotations

from thalia.components.neurons.dendritic import (
    DendriticBranch,
    DendriticBranchConfig,
    DendriticNeuron,
    DendriticNeuronConfig,
    compute_branch_selectivity,
    create_clustered_input,
    create_scattered_input,
)
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.neurons.neuron_factory import (
    NeuronFactory,
    create_cortical_layer_neurons,
    create_fast_spiking_neurons,
    create_pyramidal_neurons,
    create_relay_neurons,
    create_trn_neurons,
)
from thalia.constants.neuron import (  # Membrane time constants; Synaptic time constants; Voltage parameters; Spike detection; Reversal potentials; Conductances; Adaptation constants; Weight initialization scales; Presets
    ADAPT_INCREMENT_CORTEX_L23,
    ADAPT_INCREMENT_MODERATE,
    ADAPT_INCREMENT_NONE,
    ADAPT_INCREMENT_STRONG,
    E_EXCITATORY,
    E_INHIBITORY,
    E_LEAK,
    FAST_SPIKING_INTERNEURON,
    G_LEAK_FAST,
    G_LEAK_SLOW,
    G_LEAK_STANDARD,
    SPIKE_ACTIVITY_THRESHOLD,
    SPIKE_DETECTION_THRESHOLD,
    STANDARD_PYRAMIDAL,
    TAU_MEM_FAST,
    TAU_MEM_SLOW,
    TAU_MEM_STANDARD,
    TAU_SYN_EXCITATORY,
    TAU_SYN_INHIBITORY,
    TAU_SYN_NMDA,
    V_RESET_STANDARD,
    V_REST_STANDARD,
    V_THRESHOLD_STANDARD,
    WEIGHT_INIT_SCALE_MODERATE,
    WEIGHT_INIT_SCALE_SMALL,
    WEIGHT_INIT_SCALE_SPARSITY_DEFAULT,
)

__all__ = [
    # Neuron models
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Neuron factory
    "NeuronFactory",
    "create_pyramidal_neurons",
    "create_relay_neurons",
    "create_trn_neurons",
    "create_cortical_layer_neurons",
    "create_fast_spiking_neurons",
    # Dendritic computation
    "DendriticBranch",
    "DendriticBranchConfig",
    "DendriticNeuron",
    "DendriticNeuronConfig",
    "compute_branch_selectivity",
    "create_clustered_input",
    "create_scattered_input",
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
    "SPIKE_DETECTION_THRESHOLD",
    "SPIKE_ACTIVITY_THRESHOLD",
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
    "WEIGHT_INIT_SCALE_SMALL",
    "WEIGHT_INIT_SCALE_MODERATE",
    "WEIGHT_INIT_SCALE_SPARSITY_DEFAULT",
    "STANDARD_PYRAMIDAL",
    "FAST_SPIKING_INTERNEURON",
]
