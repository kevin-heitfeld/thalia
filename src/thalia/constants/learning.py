"""
Learning Constants

Learning rates, eligibility traces, STDP parameters, and plasticity time constants.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

# =============================================================================
# LEARNING RATES (dimensionless)
# =============================================================================

# Spike-Timing Dependent Plasticity (STDP)
LEARNING_RATE_STDP = 0.001
"""Standard STDP learning rate for cortical synapses."""

# Bienenstock-Cooper-Munro (BCM)
LEARNING_RATE_BCM = 0.01
"""Standard BCM learning rate for unsupervised feature learning."""

# Hebbian Learning
LEARNING_RATE_HEBBIAN = 0.01
"""Basic Hebbian learning rate (pre × post)."""

LEARNING_RATE_HEBBIAN_SLOW = 0.001
"""Slow Hebbian learning for stable multimodal integration."""

LEARNING_RATE_ONE_SHOT = 0.1
"""One-shot learning rate for hippocampal episodic memory."""

# Precision Learning (Predictive Coding)
LEARNING_RATE_PRECISION = 0.001
"""Precision weight learning rate for predictive coding networks."""

# Striatal Learning Modulation
LEARNING_RATE_STRIATUM_PFC_MODULATION = 0.001
"""Learning rate for prefrontal modulation of striatal pathways."""

# =============================================================================
# BCM THRESHOLD TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_BCM_THRESHOLD = 5000.0
"""BCM sliding threshold adaptation time constant (5 seconds)."""

# =============================================================================
# STDP AMPLITUDE CONSTANTS (dimensionless)
# =============================================================================

STDP_A_PLUS_CORTEX = 0.01
"""LTP amplitude for cortical STDP."""

STDP_A_MINUS_CORTEX = 0.012
"""LTD amplitude for cortical STDP (slightly larger for stability)."""

# =============================================================================
# ACTIVITY TRACKING PARAMETERS
# =============================================================================

EMA_DECAY_FAST = 0.99
"""Fast exponential moving average decay (~100 timestep window)."""

EMA_DECAY_SLOW = 0.999
"""Slow exponential moving average decay (~1000 timestep window)."""

# =============================================================================
# NOISE PARAMETERS
# =============================================================================

WM_NOISE_STD_DEFAULT = 0.01
"""Default working memory noise standard deviation."""

# =============================================================================
# ACTIVITY DETECTION THRESHOLDS
# =============================================================================

SILENCE_DETECTION_THRESHOLD = 0.001
"""Firing rate threshold below which a region is considered silent."""


__all__ = [
    "LEARNING_RATE_STDP",
    "LEARNING_RATE_BCM",
    "LEARNING_RATE_HEBBIAN",
    "LEARNING_RATE_HEBBIAN_SLOW",
    "LEARNING_RATE_PRECISION",
    "LEARNING_RATE_STRIATUM_PFC_MODULATION",
    "LEARNING_RATE_ONE_SHOT",
    "TAU_BCM_THRESHOLD",
    "STDP_A_PLUS_CORTEX",
    "STDP_A_MINUS_CORTEX",
    "EMA_DECAY_FAST",
    "EMA_DECAY_SLOW",
    "WM_NOISE_STD_DEFAULT",
    "SILENCE_DETECTION_THRESHOLD",
]
