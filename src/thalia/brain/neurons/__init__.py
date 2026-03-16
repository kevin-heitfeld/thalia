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
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    split_excitatory_conductance,
)
from .norepinephrine_neuron import (
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
)
from .serotonin_neuron import (
    SerotoninNeuronConfig,
    SerotoninNeuron,
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
    # Utils
    "heterogeneous_adapt_increment",
    "heterogeneous_g_L",
    "heterogeneous_tau_mem",
    "heterogeneous_v_threshold",
    "split_excitatory_conductance",
]
