"""
Neuromodulation Constants - Dopamine, Acetylcholine, Norepinephrine.

Consolidated from neuromodulation/constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
"""

import math

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

ACH_ENCODING_LEVEL = 0.8
"""ACh level during encoding (high = strengthen new memories)."""

ACH_RETRIEVAL_LEVEL = 0.2
"""ACh level during retrieval (low = strengthen recall pathways)."""

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

def decay_constant_to_tau(decay_per_ms: float, dt_ms: float = 1.0) -> float:
    """Convert decay constant to time constant."""
    return -dt_ms / math.log(decay_per_ms)


def tau_to_decay_constant(tau_ms: float, dt_ms: float = 1.0) -> float:
    """Convert time constant to decay factor."""
    return math.exp(-dt_ms / tau_ms)


def compute_ne_gain(ne_level: float) -> float:
    """Compute norepinephrine gain modulation from NE level."""
    return NE_GAIN_MIN + (NE_GAIN_MAX - NE_GAIN_MIN) * ne_level


__all__ = [
    "DA_BASELINE_STANDARD",
    "DA_BASELINE_STRIATUM",
    "ACH_BASELINE",
    "ACH_ENCODING_LEVEL",
    "ACH_RETRIEVAL_LEVEL",
    "NE_BASELINE",
    "NE_GAIN_MIN",
    "NE_GAIN_MAX",
    "decay_constant_to_tau",
    "tau_to_decay_constant",
    "compute_ne_gain",
]
