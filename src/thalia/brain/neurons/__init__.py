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
    heterogeneous_adapt_increment,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
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
from .two_compartment_lif_neuron import (
    TwoCompartmentLIFConfig,
    TwoCompartmentLIF,
)
from .neuron_config_builder import (
    build_conductance_lif_config,
    build_two_compartment_config,
)

__all__ = [
    # Neuron models
    "ConductanceLIFConfig",
    "ConductanceLIF",
    "AcetylcholineNeuronConfig",
    "AcetylcholineNeuron",
    "NorepinephrineNeuronConfig",
    "NorepinephrineNeuron",
    "SerotoninNeuronConfig",
    "SerotoninNeuron",
    "TwoCompartmentLIFConfig",
    "TwoCompartmentLIF",
    # Utils
    "heterogeneous_adapt_increment",
    "heterogeneous_dendrite_coupling",
    "heterogeneous_g_L",
    "heterogeneous_noise_std",
    "heterogeneous_tau_adapt",
    "heterogeneous_tau_mem",
    "heterogeneous_v_reset",
    "heterogeneous_v_threshold",
    "split_excitatory_conductance",
    # Builder functions
    "build_conductance_lif_config",
    "build_two_compartment_config",
]
