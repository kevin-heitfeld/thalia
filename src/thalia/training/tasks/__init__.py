"""
Cognitive tasks for Thalia's developmental curriculum.

This module provides task loaders for different curriculum stages:
- Bootstrap: Spontaneous activity + simple patterns (Stage 0)
- Sensorimotor: Motor control, reaching, manipulation (Stage 1)
- Phonology: Sensory foundations + phoneme discrimination (Stage 2)
- More stages to come...
"""

from __future__ import annotations

from .bootstrap import (
    BootstrapConfig,
    BootstrapTaskLoader,
    SimplePatternTask,
    SpontaneousActivityTask,
    TransitionPatternTask,
)

__all__ = [
    # Stage 0: Bootstrap
    "BootstrapConfig",
    "BootstrapTaskLoader",
    "SpontaneousActivityTask",
    "SimplePatternTask",
    "TransitionPatternTask",
]
