"""
Neural components: neurons, synapses, and spike coding.

This module provides the fundamental building blocks for constructing
spiking neural networks in Thalia.
"""

from thalia.components.neurons import *
from thalia.components.neurons import __all__ as neurons_all
from thalia.components.synapses import *
from thalia.components.synapses import __all__ as synapses_all

__all__ = neurons_all + synapses_all
