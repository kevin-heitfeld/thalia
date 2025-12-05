"""
Layered Cortex Package - Multi-layer cortical microcircuit.

This package provides a biologically realistic cortical column with distinct layers:
- L4: Input layer (receives external input)
- L2/3: Processing layer (recurrent, outputs to other cortex)
- L5: Output layer (outputs to subcortical structures)

Usage:
    from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig

Author: Thalia Project
Date: December 2025
"""

from .config import CorticalLayer, LayeredCortexConfig, LayeredCortexState
from .layered_cortex import LayeredCortex

__all__ = [
    "CorticalLayer",
    "LayeredCortex",
    "LayeredCortexConfig",
    "LayeredCortexState",
]
