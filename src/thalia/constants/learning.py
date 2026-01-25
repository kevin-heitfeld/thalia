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

# Precision Learning (Predictive Coding)
LEARNING_RATE_PRECISION = 0.001
"""Precision weight learning rate for predictive coding networks."""

# Striatal Learning Modulation
LEARNING_RATE_STRIATUM_PFC_MODULATION = 0.001
"""Learning rate for prefrontal modulation of striatal pathways."""

# =============================================================================
# ACTIVITY TRACKING PARAMETERS
# =============================================================================

EMA_DECAY_FAST = 0.99
"""Fast exponential moving average decay (~100 timestep window)."""

EMA_DECAY_SLOW = 0.999
"""Slow exponential moving average decay (~1000 timestep window)."""

# =============================================================================
# ACTIVITY DETECTION THRESHOLDS
# =============================================================================

SILENCE_DETECTION_THRESHOLD = 0.001
"""Firing rate threshold below which a region is considered silent."""


__all__ = [
    "LEARNING_RATE_PRECISION",
    "LEARNING_RATE_STRIATUM_PFC_MODULATION",
    "EMA_DECAY_FAST",
    "EMA_DECAY_SLOW",
    "SILENCE_DETECTION_THRESHOLD",
]
