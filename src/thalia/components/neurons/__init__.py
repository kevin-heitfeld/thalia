"""
Neuron models and dendritic computation.

This module contains spiking neuron models and dendritic processing components.
"""

from __future__ import annotations

from .acetylcholine_neuron import (
    AcetylcholineNeuronConfig,
    AcetylcholineNeuron,
)
from .conductance_lif_neuron import (
    ConductanceLIFConfig,
    ConductanceLIF,
)
from .norepinephrine_neuron import (
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
)
from .neuron_factory import (
    NeuronFactory,
    NeuronType,
)

__all__ = [
    # Neuron models
    "ConductanceLIFConfig",
    "ConductanceLIF",
    "AcetylcholineNeuronConfig",
    "AcetylcholineNeuron",
    "NorepinephrineNeuronConfig",
    "NorepinephrineNeuron",
    # Neuron factory
    "NeuronFactory",
    "NeuronType",
]
