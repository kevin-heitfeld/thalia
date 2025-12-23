"""
Neuron models and dendritic computation.

This module contains spiking neuron models and dendritic processing components.
"""

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.neurons.neuron_constants import (
    # Membrane time constants
    TAU_MEM_STANDARD,
    TAU_MEM_FAST,
    TAU_MEM_SLOW,
    # Synaptic time constants
    TAU_SYN_EXCITATORY,
    TAU_SYN_INHIBITORY,
    TAU_SYN_NMDA,
    # Voltage parameters
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    V_REST_STANDARD,
    # Reversal potentials
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
    # Conductances
    G_LEAK_STANDARD,
    G_LEAK_FAST,
    G_LEAK_SLOW,
    # Adaptation constants
    ADAPT_INCREMENT_NONE,
    ADAPT_INCREMENT_MODERATE,
    ADAPT_INCREMENT_STRONG,
    ADAPT_INCREMENT_CORTEX_L23,
    # Presets
    STANDARD_PYRAMIDAL,
    FAST_SPIKING_INTERNEURON,
)
from thalia.components.neurons.neuron_factory import (
    NeuronFactory,
    create_pyramidal_neurons,
    create_relay_neurons,
    create_trn_neurons,
    create_cortical_layer_neurons,
    create_fast_spiking_neurons,
)
from thalia.components.neurons.dendritic import (
    DendriticBranch,
    DendriticBranchConfig,
    DendriticNeuron,
    DendriticNeuronConfig,
    compute_branch_selectivity,
    create_clustered_input,
    create_scattered_input,
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
]
