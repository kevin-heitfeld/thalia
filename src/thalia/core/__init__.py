"""
Core components: neurons, synapses, layers, and networks.
"""

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.synapse import Synapse, SynapseConfig, SynapseType
from thalia.core.layer import SNNLayer
from thalia.core.network import SNNNetwork

__all__ = [
    "LIFNeuron", 
    "LIFConfig",
    "Synapse",
    "SynapseConfig",
    "SynapseType",
    "SNNLayer",
    "SNNNetwork",
]
