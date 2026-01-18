"""
Task Constants - Task-specific parameters for behavioral tasks and datasets.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

# =============================================================================
# MOTOR SPIKE PROBABILITIES
# =============================================================================

SPIKE_PROBABILITY_LOW = 0.05
"""Low motor activity spike probability (5%) - gentle exploration."""

SPIKE_PROBABILITY_MEDIUM = 0.2
"""Medium motor activity spike probability (20%) - moderate activity."""

SPIKE_PROBABILITY_HIGH = 0.4
"""High motor activity spike probability (40%) - active control."""

# =============================================================================
# STIMULUS STRENGTHS
# =============================================================================

STIMULUS_STRENGTH_LOW = 0.2
"""Low stimulus strength for weak/background stimuli."""

STIMULUS_STRENGTH_MEDIUM = 0.5
"""Medium stimulus strength for neutral stimuli."""

STIMULUS_STRENGTH_HIGH = 1.0
"""High stimulus strength for salient/attended stimuli."""

# =============================================================================
# NOISE SCALES
# =============================================================================

PROPRIOCEPTION_NOISE_SCALE = 0.1
"""Noise scale for proprioceptive feedback (10% of signal)."""

STIMULUS_NOISE_SCALE = 0.05
"""General stimulus noise scale for task variation."""

FEATURE_NOISE_MATCH = 0.3
"""Noise scale for creating near-match stimuli."""

# =============================================================================
# SENSORIMOTOR TASK WEIGHTS (Stage 0)
# =============================================================================

SENSORIMOTOR_WEIGHT_MOTOR_CONTROL = 0.25
"""Motor control task weight (25%)."""

SENSORIMOTOR_WEIGHT_REACHING = 0.25
"""Reaching task weight (25%)."""

SENSORIMOTOR_WEIGHT_MANIPULATION = 0.25
"""Manipulation task weight (25%)."""

SENSORIMOTOR_WEIGHT_PREDICTION = 0.25
"""Prediction task weight (25%)."""

# =============================================================================
# REWARD CALCULATION
# =============================================================================

REWARD_SCALE_PREDICTION = 1.0
"""Scaling factor for prediction-based rewards."""

# =============================================================================
# MATCH/MISMATCH PROBABILITIES
# =============================================================================

MATCH_PROBABILITY_DEFAULT = 0.3
"""Default probability for match trials in memory tasks."""

# =============================================================================
# FEATURE VARIATION PARAMETERS
# =============================================================================

FEATURE_INCREMENT_BASE = 0.2
"""Base increment for progressive feature changes."""

FEATURE_INCREMENT_COLUMN = 0.15
"""Column-wise feature increment for spatial patterns."""

FEATURE_INCREMENT_INTERACTION = 0.1
"""Interaction term increment for combinatorial features."""


__all__ = [
    "SPIKE_PROBABILITY_LOW",
    "SPIKE_PROBABILITY_MEDIUM",
    "SPIKE_PROBABILITY_HIGH",
    "STIMULUS_STRENGTH_LOW",
    "STIMULUS_STRENGTH_MEDIUM",
    "STIMULUS_STRENGTH_HIGH",
    "PROPRIOCEPTION_NOISE_SCALE",
    "STIMULUS_NOISE_SCALE",
    "FEATURE_NOISE_MATCH",
    "SENSORIMOTOR_WEIGHT_MOTOR_CONTROL",
    "SENSORIMOTOR_WEIGHT_REACHING",
    "SENSORIMOTOR_WEIGHT_MANIPULATION",
    "SENSORIMOTOR_WEIGHT_PREDICTION",
    "REWARD_SCALE_PREDICTION",
    "MATCH_PROBABILITY_DEFAULT",
    "FEATURE_INCREMENT_BASE",
    "FEATURE_INCREMENT_COLUMN",
    "FEATURE_INCREMENT_INTERACTION",
]
