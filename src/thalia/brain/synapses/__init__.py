"""
Synaptic components: weight initialization, short-term plasticity, and traces.

This module provides synaptic mechanisms for spiking neural networks.
"""

from __future__ import annotations

from .neuromodulator_receptor import (
    NeuromodulatorReceptor,
    NMReceptorType,
    make_neuromodulator_receptor,
)
from .stp import (
    ShortTermPlasticity,
    STPConfig,
)
from .weight_init import (
    ConductanceScaledSpec,
    WeightInitializer,
)

__all__ = [
    # Neuromodulator Receptors
    "NeuromodulatorReceptor",
    "NMReceptorType",
    "make_neuromodulator_receptor",
    # Weight Initialization
    "ConductanceScaledSpec",
    "WeightInitializer",
    # Short-term plasticity
    "ShortTermPlasticity",
    "STPConfig",
]
