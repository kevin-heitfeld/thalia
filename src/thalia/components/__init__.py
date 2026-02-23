"""
Neural components: neurons, synapses, and spike coding.

This module provides the fundamental building blocks for constructing
spiking neural networks in Thalia.
"""

from __future__ import annotations

from .gap_junctions import (
    GapJunctionConfig,
    GapJunctionCoupling,
)
from .neurons import (
    AcetylcholineNeuronConfig,
    AcetylcholineNeuron,
    ConductanceLIFConfig,
    ConductanceLIF,
    NeuronFactory,
    NeuronType,
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
)
from .synapses import (
    ShortTermPlasticity,
    STPConfig,
    STPPreset,
    STPType,
    WeightInitializer,
    NeuromodulatorReceptor,
    STP_PRESETS,
    get_stp_config,
)

__all__ = [
    # Neuron models
    "AcetylcholineNeuronConfig",
    "AcetylcholineNeuron",
    "ConductanceLIFConfig",
    "ConductanceLIF",
    "NorepinephrineNeuronConfig",
    "NorepinephrineNeuron",
    # Neuron factory
    "NeuronFactory",
    "NeuronType",
    # Weight Initialization
    "WeightInitializer",
    # Short-term plasticity
    "ShortTermPlasticity",
    "STPConfig",
    "STPType",
    # STP presets
    "STP_PRESETS",
    "STPPreset",
    "get_stp_config",
    # Gap Junctions
    "GapJunctionConfig",
    "GapJunctionCoupling",
    # Neuromodulator Receptors
    "NeuromodulatorReceptor",
]
