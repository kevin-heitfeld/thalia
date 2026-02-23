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
    TwoCompartmentLIFConfig,
    TwoCompartmentLIF,
)
from .norepinephrine_neuron import (
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
)
from .serotonin_neuron import (
    SerotoninNeuronConfig,
    SerotoninNeuron,
)
from .neuron_factory import (
    NeuronFactory,
    NeuronType,
)

__all__ = [
    # Neuron models
    "ConductanceLIFConfig",
    "ConductanceLIF",
    "TwoCompartmentLIFConfig",
    "TwoCompartmentLIF",
    "AcetylcholineNeuronConfig",
    "AcetylcholineNeuron",
    "NorepinephrineNeuronConfig",
    "NorepinephrineNeuron",
    "SerotoninNeuronConfig",
    "SerotoninNeuron",
    # Neuron factory
    "NeuronFactory",
    "NeuronType",
]
