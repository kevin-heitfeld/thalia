"""Stimulus patterns for sensory input.

This module provides clean abstractions for different temporal input patterns
matching neurobiological reality:

- **Sustained**: Constant stimulus held over time (tonic input)
- **Transient**: Brief pulse followed by silence (phasic input)
- **Sequential**: Time-varying signal with explicit per-timestep values
- **Programmatic**: Generated on-demand via function
"""

from __future__ import annotations

from .base import StimulusPattern
from .programmatic import Programmatic
from .sequential import Sequential
from .sustained import Sustained
from .transient import Transient

__all__ = [
    "StimulusPattern",
    "Sustained",
    "Transient",
    "Sequential",
    "Programmatic",
]
