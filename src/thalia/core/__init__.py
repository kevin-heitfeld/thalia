"""
Core components: neurons, synapses, layers, and networks.
"""

from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF, ConductanceLIFConfig
from thalia.core.synapse import Synapse, SynapseConfig, SynapseType
from thalia.core.layer import SNNLayer
from thalia.core.network import SNNNetwork
from thalia.core.dendritic import (
    DendriticBranch,
    DendriticBranchConfig,
    DendriticNeuron,
    DendriticNeuronConfig,
    compute_branch_selectivity,
    create_clustered_input,
    create_scattered_input,
)

__all__ = [
    "LIFNeuron", 
    "LIFConfig",
    "ConductanceLIF",
    "ConductanceLIFConfig",
    "Synapse",
    "SynapseConfig",
    "SynapseType",
    "SNNLayer",
    "SNNNetwork",
    # Dendritic computation
    "DendriticBranch",
    "DendriticBranchConfig",
    "DendriticNeuron",
    "DendriticNeuronConfig",
    "compute_branch_selectivity",
    "create_clustered_input",
    "create_scattered_input",
]
