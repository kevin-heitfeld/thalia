"""
Centralized Constants for Thalia.

This module provides a single import point for all constants used across
the Thalia codebase, organized by category for easy discovery.

Usage:
======
    # Import from specific category
    from thalia.constants.neuron import TAU_MEM_STANDARD
    from thalia.constants.oscillator import THETA_ENCODING_PHASE_SCALE

    # Or import entire category
    from thalia.constants import learning, neuron, oscillator

Categories:
===========
- architecture: Expansion factors, capacity ratios
- exploration: Epsilon-greedy, UCB, softmax parameters
- learning: Learning rates, eligibility traces, STDP windows
- neuromodulation: Dopamine, acetylcholine, norepinephrine parameters
- neuron: Membrane time constants, thresholds, refractory periods
- oscillator: Theta, gamma, alpha frequencies and coupling
- regions: Thalamus and striatum specialized constants
- sensory: Retinal, cochlear, and somatosensory processing constants
- task: Task-specific parameters
- time: Time unit conversions (ms/s, TAU)
- training: Batch sizes, learning schedules, augmentation rates
- visualization: Plot alphas, thresholds, colors, network graph styling

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
"""

from __future__ import annotations

# Re-export all constants for convenience
from .architecture import *
from .exploration import *
from .learning import *
from .neuromodulation import *
from .neuron import *
from .oscillator import *
from .regions import *
from .sensory import *
from .task import *
from .time import *
from .training import *
from .visualization import *

__all__ = [
    # Submodules
    "architecture",
    "exploration",
    "learning",
    "neuromodulation",
    "neuron",
    "oscillator",
    "regions",
    "sensory",
    "task",
    "time",
    "training",
    "visualization",
]
