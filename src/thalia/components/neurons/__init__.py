"""
Neuron models and dendritic computation.

This module contains spiking neuron models and dendritic processing components.
"""

from __future__ import annotations

from .neuron import (
    ConductanceLIF,
    ConductanceLIFConfig,
)
from .neuron_factory import (
    NeuronFactory,
    NeuronType,
)

__all__ = [
    # Neuron models
    "ConductanceLIF",
    "ConductanceLIFConfig",
    # Neuron factory
    "NeuronFactory",
    "NeuronType",
]
