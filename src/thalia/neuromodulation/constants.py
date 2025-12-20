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

# =============================================================================
# Dopamine (DA) - Reward Prediction Error and Reinforcement
# =============================================================================

# Phasic dopamine decay (reuptake by DAT transporters)
# Biological basis: DA phasic bursts last ~200ms
# τ = 200ms → decay = exp(-dt/τ) ≈ 0.995 per ms (assuming dt=1ms)
DA_PHASIC_DECAY_PER_MS = 0.995

# Tonic dopamine smoothing (slow baseline changes)
# Controls how fast baseline DA adapts to sustained reward conditions
# α = 0.05 → τ ≈ 20ms smoothing window
DA_TONIC_ALPHA = 0.05

# Baseline dopamine level (resting state)
# Typical range: 0.2-0.5 (normalized units)
DA_BASELINE = 0.3

# DA burst magnitude (reward delivery)
DA_BURST_MAGNITUDE = 1.0

# DA dip magnitude (reward omission)
DA_DIP_MAGNITUDE = -0.5


# =============================================================================
# Acetylcholine (ACh) - Attention and Encoding/Retrieval
# =============================================================================

# ACh decay (reuptake by AChE)
# Biological basis: ACh has fast hydrolysis, τ ~ 50-100ms
# τ = 50ms → decay = exp(-dt/τ) ≈ 0.980 per ms
ACH_DECAY_PER_MS = 0.980

# Baseline ACh level (resting state)
ACH_BASELINE = 0.3

# ACh encoding boost (novelty/attention)
ACH_ENCODING_LEVEL = 0.8

# ACh retrieval suppression (familiar contexts)
ACH_RETRIEVAL_LEVEL = 0.2


# =============================================================================
# Norepinephrine (NE) - Arousal and Gain Modulation
# =============================================================================

# NE decay (reuptake by NET transporters)
# Biological basis: NE clearance τ ~ 100-200ms
# τ = 100ms → decay = exp(-dt/τ) ≈ 0.990 per ms
NE_DECAY_PER_MS = 0.990

# Baseline arousal (resting state)
NE_BASELINE = 0.3

# Arousal update rate (how fast arousal tracks uncertainty)
# α = 0.1 → moderate smoothing
NE_AROUSAL_ALPHA = 0.1

# NE burst magnitude (surprise/unexpected uncertainty)
NE_BURST_MAGNITUDE = 1.0


# =============================================================================
# Homeostatic Regulation
# =============================================================================

# Adaptation timescale (exponential smoothing)
# Controls how fast receptor sensitivity adapts to sustained neuromodulator exposure
# tau = 0.999 → ~1000 timesteps to fully adapt (~10 seconds if dt=10ms)
HOMEOSTATIC_TAU = 0.999

# Receptor sensitivity bounds (downregulation/upregulation)
MIN_RECEPTOR_SENSITIVITY = 0.5   # Maximum downregulation (50% of baseline)
MAX_RECEPTOR_SENSITIVITY = 1.5   # Maximum upregulation (150% of baseline)

# Target average neuromodulator level (homeostatic setpoint)
TARGET_NEUROMODULATOR_LEVEL = 0.5


# =============================================================================
# Cross-System Interactions
# =============================================================================

# DA-ACh interaction strength
# High DA suppresses ACh release (reward → reduce novelty encoding)
DA_ACH_SUPPRESSION = 0.3

# NE-ACh interaction strength
# High NE enhances ACh release (arousal → boost attention)
NE_ACH_ENHANCEMENT = 0.2

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
    import math
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
    import math
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
    "DA_PHASIC_DECAY_PER_MS",
    "DA_TONIC_ALPHA",
    "DA_BASELINE",
    "DA_BURST_MAGNITUDE",
    "DA_DIP_MAGNITUDE",
    # Acetylcholine constants
    "ACH_DECAY_PER_MS",
    "ACH_BASELINE",
    "ACH_ENCODING_LEVEL",
    "ACH_RETRIEVAL_LEVEL",
    # Norepinephrine constants
    "NE_DECAY_PER_MS",
    "NE_BASELINE",
    "NE_AROUSAL_ALPHA",
    "NE_BURST_MAGNITUDE",
    # Homeostatic regulation
    "HOMEOSTATIC_TAU",
    "MIN_RECEPTOR_SENSITIVITY",
    "MAX_RECEPTOR_SENSITIVITY",
    "TARGET_NEUROMODULATOR_LEVEL",
    # Cross-system interactions
    "DA_ACH_SUPPRESSION",
    "NE_ACH_ENHANCEMENT",
    "NE_GAIN_MIN",
    "NE_GAIN_MAX",
    # Helper functions
    "decay_constant_to_tau",
    "tau_to_decay_constant",
    "compute_ne_gain",
]
