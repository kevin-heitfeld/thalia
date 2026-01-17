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
]
