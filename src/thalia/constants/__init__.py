"""
Centralized Constants for Thalia.

This module provides a single import point for all constants used across
the Thalia codebase, organized by category for easy discovery.

Usage:
======
    # Import from specific category
    from thalia.constants.learning import LEARNING_RATE_STDP
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
from thalia.constants.architecture import *
from thalia.constants.exploration import *
from thalia.constants.learning import *
from thalia.constants.neuromodulation import *
from thalia.constants.neuron import *
from thalia.constants.oscillator import *
from thalia.constants.regions import *
from thalia.constants.sensory import *
from thalia.constants.task import *
from thalia.constants.time import *
from thalia.constants.training import *
from thalia.constants.visualization import *

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
