"""
Architecture Constants - Structural ratios and expansion factors.

These define biological structure ratios based on neuroanatomy.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

# =============================================================================
# CORTICAL LAYER ARCHITECTURE
# =============================================================================

CORTEX_L4_DA_FRACTION = 0.2
"""Layer 4 dopamine sensitivity fraction (sensory input layer, low DA)."""

CORTEX_L23_DA_FRACTION = 0.3
"""Layer 2/3 dopamine sensitivity fraction (association layer, moderate DA)."""

CORTEX_L5_DA_FRACTION = 0.4
"""Layer 5 dopamine sensitivity fraction (motor output layer, high DA)."""

CORTEX_L6_DA_FRACTION = 0.1
"""Layer 6 dopamine sensitivity fraction (feedback/attention layer, low DA)."""

# =============================================================================
# NEURAL GROWTH CONSTANTS
# =============================================================================

GROWTH_NEW_WEIGHT_SCALE = 0.2
"""Scaling factor for new weights during neurogenesis (20% of w_max)."""

ACTIVITY_HISTORY_DECAY = 0.99
"""Exponential decay factor for activity history tracking."""

ACTIVITY_HISTORY_INCREMENT = 0.01
"""Increment weight for new activity in exponential moving average."""


__all__ = [
    "CORTEX_L4_DA_FRACTION",
    "CORTEX_L23_DA_FRACTION",
    "CORTEX_L5_DA_FRACTION",
    "CORTEX_L6_DA_FRACTION",
    "GROWTH_NEW_WEIGHT_SCALE",
    "ACTIVITY_HISTORY_DECAY",
    "ACTIVITY_HISTORY_INCREMENT",
]
