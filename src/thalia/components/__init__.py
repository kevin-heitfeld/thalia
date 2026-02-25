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
    TwoCompartmentLIFConfig,
    TwoCompartmentLIF,
    NeuronFactory,
    NeuronType,
    NorepinephrineNeuronConfig,
    NorepinephrineNeuron,
    SerotoninNeuronConfig,
    SerotoninNeuron,
)
from .synapses import (
    ConductanceScaledSpec,
    ShortTermPlasticity,
    STPConfig,
    STPPreset,
    STPType,
    WeightInitializer,
    NeuromodulatorReceptor,
)

__all__ = [
    # Neuron models
    "AcetylcholineNeuronConfig",
    "AcetylcholineNeuron",
    "ConductanceLIFConfig",
    "ConductanceLIF",
    "TwoCompartmentLIFConfig",
    "TwoCompartmentLIF",
    "NorepinephrineNeuronConfig",
    "NorepinephrineNeuron",
    "SerotoninNeuronConfig",
    "SerotoninNeuron",
    # Neuron factory
    "NeuronFactory",
    "NeuronType",
    # Weight Initialization
    "ConductanceScaledSpec",
    "WeightInitializer",
    # Short-term plasticity
    "ShortTermPlasticity",
    "STPConfig",
    "STPType",
    # STP presets
    "STPPreset",
    # Gap Junctions
    "GapJunctionConfig",
    "GapJunctionCoupling",
    # Neuromodulator Receptors
    "NeuromodulatorReceptor",
]
