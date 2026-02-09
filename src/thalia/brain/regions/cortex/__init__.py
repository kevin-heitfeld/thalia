"""
Cortex Package - Multi-layer cortical microcircuit.

This package provides a biologically realistic cortical column with distinct layers:
- L4: Input layer (receives external input)
- L2/3: Processing layer (recurrent, outputs to other cortex)
- L5: Output layer (outputs to subcortical structures)
- L6a: Feedback layer (sends feedback to thalamus)
- L6b: Modulatory layer (influences overall excitability)
"""

from __future__ import annotations

from .cortex import Cortex

__all__ = [
    "Cortex",
]
