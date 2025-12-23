"""
Neuromodulation Constants - Biological Parameters for Dopamine, Acetylcholine, Norepinephrine.

This module centralizes timing and decay constants for neuromodulator systems,
providing biologically-motivated values with clear documentation.

References:
-----------
- Schultz et al. (1997): Dopamine reward prediction error
- Dayan & Yu (2006): Expected and unexpected uncertainty (ACh/NE)
- Fiorillo et al. (2003): DA phasic responses (~200ms duration)
- Aston-Jones & Cohen (2005): LC-NE and arousal modulation

Author: Thalia Project
Date: December 2025
"""

import math


# =============================================================================
# Dopamine (DA) - Reward Prediction Error and Reinforcement
# =============================================================================

# Region-specific dopamine baselines
# Different brain regions have different tonic DA levels based on innervation density
DA_BASELINE_STANDARD = 0.2  # Most regions (cortex, thalamus, hippocampus, cerebellum, prefrontal)
DA_BASELINE_STRIATUM = 0.3  # Striatum has higher DA innervation for RL


# =============================================================================
# Acetylcholine (ACh) - Attention and Encoding/Retrieval
# =============================================================================

# Baseline ACh level (resting state)
ACH_BASELINE = 0.3


# =============================================================================
# Norepinephrine (NE) - Arousal and Gain Modulation
# =============================================================================

# Baseline arousal (resting state)
NE_BASELINE = 0.3

# NE gain modulation range
# NE modulates network gain (multiplicative effect on synaptic transmission)
# Biological basis: β-adrenergic receptor effects on neural excitability
# Gain starts at baseline (1.0) and increases with arousal up to 1.5x
NE_GAIN_MIN = 1.0   # Baseline arousal (no NE modulation)
NE_GAIN_MAX = 1.5   # High arousal → increased gain


# =============================================================================
# Conversion Helpers
# =============================================================================

def decay_constant_to_tau(decay_per_ms: float, dt_ms: float = 1.0) -> float:
    """Convert decay constant to time constant.

    Args:
        decay_per_ms: Decay factor per millisecond (e.g., 0.995)
        dt_ms: Simulation timestep in milliseconds

    Returns:
        Time constant τ in milliseconds

    Example:
        >>> decay_constant_to_tau(0.995)
        200.0  # τ = 200ms
    """
    return -dt_ms / math.log(decay_per_ms)


def tau_to_decay_constant(tau_ms: float, dt_ms: float = 1.0) -> float:
    """Convert time constant to decay factor.

    Args:
        tau_ms: Time constant in milliseconds
        dt_ms: Simulation timestep in milliseconds

    Returns:
        Decay factor per timestep

    Example:
        >>> tau_to_decay_constant(200.0, dt_ms=1.0)
        0.995
    """
    return math.exp(-dt_ms / tau_ms)


def compute_ne_gain(ne_level: float) -> float:
    """Compute norepinephrine gain modulation from NE level.

    NE modulates network gain multiplicatively from baseline (1.0) to high arousal (1.5).
    Biological basis: β-adrenergic receptor effects on neural excitability.

    Args:
        ne_level: Norepinephrine level in [0, 1]

    Returns:
        Gain multiplier in [NE_GAIN_MIN, NE_GAIN_MAX]

    Example:
        >>> compute_ne_gain(0.0)
        1.0  # Baseline (no NE)
        >>> compute_ne_gain(1.0)
        1.5  # Maximum arousal
        >>> compute_ne_gain(0.5)
        1.25  # Moderate arousal
    """
    return NE_GAIN_MIN + (NE_GAIN_MAX - NE_GAIN_MIN) * ne_level


__all__ = [
    # Dopamine constants
    "DA_BASELINE_STANDARD",
    "DA_BASELINE_STRIATUM",
    # Acetylcholine constants
    "ACH_BASELINE",
    # Norepinephrine constants
    "NE_BASELINE",
    "NE_GAIN_MIN",
    "NE_GAIN_MAX",
    # Helper functions
    "decay_constant_to_tau",
    "tau_to_decay_constant",
    "compute_ne_gain",
]
