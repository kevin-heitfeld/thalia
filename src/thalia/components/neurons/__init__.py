"""
Neuron models and dendritic computation.

This module contains spiking neuron models and dendritic processing components.
"""

from __future__ import annotations

from .acetylcholine_neuron import (
    AcetylcholineNeuron,
    AcetylcholineNeuronConfig,
)
from .dopamine_neuron import (
    TonicDopamineNeuron,
)
from .izhikevich_neuron import (
    IzhikevichNeuron,
    IzhikevichNeuronConfig,
)
from .neuron import (
    ConductanceLIF,
    ConductanceLIFConfig,
)
from .norepinephrine_neuron import (
    NorepinephrineNeuron,
    NorepinephrineNeuronConfig,
)
from .neuron_factory import (
    NeuronFactory,
    NeuronType,
)

__all__ = [
    # Neuron models
    "ConductanceLIF",
    "ConductanceLIFConfig",
    "IzhikevichNeuron",
    "IzhikevichNeuronConfig",
    "AcetylcholineNeuron",
    "AcetylcholineNeuronConfig",
    "TonicDopamineNeuron",
    "NorepinephrineNeuron",
    "NorepinephrineNeuronConfig",
    # Neuron factory
    "NeuronFactory",
    "NeuronType",
]
