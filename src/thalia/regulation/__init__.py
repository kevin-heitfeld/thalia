"""Regulation Constants.

Constants and utilities for homeostasis, learning, and normalization.
"""

from thalia.regulation.homeostasis_constants import (
    TARGET_FIRING_RATE_STANDARD,
    TARGET_FIRING_RATE_LOW,
    TARGET_FIRING_RATE_MEDIUM,
    TARGET_FIRING_RATE_HIGH,
    HOMEOSTATIC_TAU_STANDARD,
    HOMEOSTATIC_TAU_FAST,
    HOMEOSTATIC_TAU_SLOW,
    SYNAPTIC_SCALING_RATE,
    SYNAPTIC_SCALING_MIN,
    SYNAPTIC_SCALING_MAX,
    INTRINSIC_PLASTICITY_RATE,
    FIRING_RATE_WINDOW_MS,
    MIN_FIRING_RATE_HZ,
    MAX_FIRING_RATE_HZ,
)
from thalia.regulation.learning_constants import (
    LEARNING_RATE_DEFAULT,
    LEARNING_RATE_STDP,
    LEARNING_RATE_BCM,
    LEARNING_RATE_HEBBIAN,
    TAU_ELIGIBILITY_STANDARD,
    TAU_BCM_THRESHOLD,
    TAU_STDP_PLUS,
    TAU_STDP_MINUS,
    WEIGHT_INIT_SCALE_PREDICTIVE,
    WEIGHT_INIT_SCALE_RECURRENT,
    EMA_DECAY_FAST,
    EMA_DECAY_SLOW,
    WM_NOISE_STD_DEFAULT,
)
from thalia.regulation.normalization import (
    DivisiveNormConfig,
    DivisiveNormalization,
    ContrastNormalization,
    SpatialDivisiveNorm,
)

__all__ = [
    # Homeostasis Constants - Firing Rate Targets
    "TARGET_FIRING_RATE_STANDARD",
    "TARGET_FIRING_RATE_LOW",
    "TARGET_FIRING_RATE_MEDIUM",
    "TARGET_FIRING_RATE_HIGH",
    # Homeostasis Constants - Time Constants
    "HOMEOSTATIC_TAU_STANDARD",
    "HOMEOSTATIC_TAU_FAST",
    "HOMEOSTATIC_TAU_SLOW",
    # Homeostasis Constants - Scaling
    "SYNAPTIC_SCALING_RATE",
    "SYNAPTIC_SCALING_MIN",
    "SYNAPTIC_SCALING_MAX",
    "INTRINSIC_PLASTICITY_RATE",
    "FIRING_RATE_WINDOW_MS",
    # Homeostasis Constants - Rate Bounds
    "MIN_FIRING_RATE_HZ",
    "MAX_FIRING_RATE_HZ",
    # Learning Constants - Rates
    "LEARNING_RATE_DEFAULT",
    "LEARNING_RATE_STDP",
    "LEARNING_RATE_BCM",
    "LEARNING_RATE_HEBBIAN",
    # Learning Constants - Time Constants
    "TAU_ELIGIBILITY_STANDARD",
    "TAU_BCM_THRESHOLD",
    "TAU_STDP_PLUS",
    "TAU_STDP_MINUS",
    # Learning Constants - Weight Initialization
    "WEIGHT_INIT_SCALE_PREDICTIVE",
    "WEIGHT_INIT_SCALE_RECURRENT",
    # Learning Constants - Activity Tracking
    "EMA_DECAY_FAST",
    "EMA_DECAY_SLOW",
    # Learning Constants - Noise Parameters
    "WM_NOISE_STD_DEFAULT",
    # Normalization Classes
    "DivisiveNormConfig",
    "DivisiveNormalization",
    "ContrastNormalization",
    "SpatialDivisiveNorm",
]
