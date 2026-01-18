"""
Neuromodulation Constants - Dopamine, Acetylcholine, Norepinephrine.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

# =============================================================================
# Dopamine (DA) - Reward Prediction Error and Reinforcement
# =============================================================================

DA_BASELINE_STANDARD = 0.2
"""Standard dopamine baseline for most regions (cortex, thalamus, hippocampus, cerebellum, prefrontal)."""

DA_BASELINE_STRIATUM = 0.3
"""Striatum dopamine baseline (higher DA innervation for RL)."""

# =============================================================================
# Acetylcholine (ACh) - Attention and Encoding/Retrieval
# =============================================================================

ACH_BASELINE = 0.3
"""Baseline acetylcholine level (resting state)."""

# =============================================================================
# Norepinephrine (NE) - Arousal and Gain Modulation
# =============================================================================

NE_BASELINE = 0.3
"""Baseline norepinephrine (arousal level)."""

NE_GAIN_MIN = 1.0
"""Baseline arousal (no NE modulation)."""

NE_GAIN_MAX = 1.5
"""High arousal → increased gain."""

# =============================================================================
# Helper Functions
# =============================================================================


def compute_ne_gain(ne_level: float) -> float:
    """Compute norepinephrine gain modulation from NE level."""
    return NE_GAIN_MIN + (NE_GAIN_MAX - NE_GAIN_MIN) * ne_level


__all__ = [
    "DA_BASELINE_STANDARD",
    "DA_BASELINE_STRIATUM",
    "ACH_BASELINE",
    "NE_BASELINE",
    "NE_GAIN_MIN",
    "NE_GAIN_MAX",
    "compute_ne_gain",
]
