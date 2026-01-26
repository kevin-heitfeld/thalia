"""
Layered Cortex Package - Multi-layer cortical microcircuit.

This package provides a biologically realistic cortical column with distinct layers:
- L4: Input layer (receives external input)
- L2/3: Processing layer (recurrent, outputs to other cortex)
- L5: Output layer (outputs to subcortical structures)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from .checkpoint_manager import LayeredCortexCheckpointManager
from .layered_cortex import LayeredCortex
from .predictive_cortex import PredictiveCortex
from .state import LayeredCortexState, PredictiveCortexState

__all__ = [
    "LayeredCortex",
    "LayeredCortexCheckpointManager",
    "LayeredCortexState",
    "PredictiveCortex",
    "PredictiveCortexState",
]
