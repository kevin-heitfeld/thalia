"""
Centralized Constants for Thalia.

This module provides a single import point for all constants used across
the Thalia codebase, organized by category for easy discovery.

Usage:
======
    # Import from specific category
    from thalia.constants.learning import LEARNING_RATE_STDP
    from thalia.constants.neuron import TAU_MEMBRANE_MS
    from thalia.constants.oscillator import THETA_ENCODING_PHASE_SCALE

    # Or import entire category
    from thalia.constants import learning, neuron, oscillator

Categories:
===========
- architecture: Expansion factors, capacity ratios
- exploration: Epsilon-greedy, UCB, softmax parameters
- homeostasis: Target firing rates, metabolic budgets
- learning: Learning rates, eligibility traces, STDP windows
- neuromodulation: Dopamine, acetylcholine, norepinephrine parameters
- neuron: Membrane time constants, thresholds, refractory periods
- oscillator: Theta, gamma, alpha frequencies and coupling
- regions: Thalamus and striatum specialized constants
- task: Task-specific parameters
- training: Batch sizes, learning schedules, augmentation rates
- visualization: Plot alphas, thresholds, colors

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
"""

# Re-export all constants for convenience
from thalia.constants.architecture import *
from thalia.constants.exploration import *
from thalia.constants.homeostasis import *
from thalia.constants.learning import *
from thalia.constants.neuromodulation import *
from thalia.constants.neuron import *
from thalia.constants.oscillator import *
from thalia.constants.regions import *
from thalia.constants.task import *
from thalia.constants.training import *
from thalia.constants.visualization import *

__all__ = [
    # Submodules
    "architecture",
    "exploration",
    "homeostasis",
    "learning",
    "neuromodulation",
    "neuron",
    "oscillator",
    "regions",
    "task",
    "training",
    "visualization",
]
