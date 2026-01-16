"""
Task Constants - Task-specific parameters for behavioral tasks and datasets.

Consolidated from tasks/task_constants.py and training/datasets/constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
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

STIMULUS_STRENGTH_HIGH = 1.0
"""High stimulus strength for salient/attended stimuli."""

STIMULUS_STRENGTH_MEDIUM = 0.5
"""Medium stimulus strength for neutral stimuli."""

STIMULUS_STRENGTH_LOW = 0.2
"""Low stimulus strength for weak/background stimuli."""

# =============================================================================
# NOISE SCALES
# =============================================================================

PROPRIOCEPTION_NOISE_SCALE = 0.1
"""Noise scale for proprioceptive feedback (10% of signal)."""

WEIGHT_INIT_SCALE_SMALL = 0.01
"""Small weight initialization scale for fine-grained variation."""

WEIGHT_INIT_SCALE_MEDIUM = 0.05
"""Medium weight initialization scale for moderate variation."""

STIMULUS_NOISE_SCALE = 0.05
"""General stimulus noise scale for task variation."""

FEATURE_NOISE_MATCH = 0.3
"""Noise scale for creating near-match stimuli."""

# =============================================================================
# DATASET SAMPLING WEIGHTS (Birth Stage)
# =============================================================================

DATASET_WEIGHT_MNIST = 0.40
"""MNIST dataset sampling weight for Birth stage (40%)."""

DATASET_WEIGHT_TEMPORAL = 0.20
"""Temporal sequence dataset weight for Birth stage (20%)."""

DATASET_WEIGHT_PHONOLOGY = 0.30
"""Phonology dataset weight for Birth stage (30%)."""

DATASET_WEIGHT_GAZE = 0.10
"""Gaze following dataset weight for Birth stage (10%)."""

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
# WORKSPACE AND ENVIRONMENT PARAMETERS
# =============================================================================

WORKSPACE_SIZE_DEFAULT = 1.0
"""Default workspace size for sensorimotor tasks."""

# =============================================================================
# REWARD CALCULATION
# =============================================================================

REWARD_SCALE_PREDICTION = 1.0
"""Scaling factor for prediction-based rewards."""

# =============================================================================
# SPATIAL PARAMETERS
# =============================================================================

TARGET_POSITION_MARGIN = 0.1
"""Margin for target position validation (10% of workspace)."""

REACHING_SUCCESS_THRESHOLD = 0.05
"""Distance threshold for successful reaching (5% of workspace)."""

# =============================================================================
# MATCH/MISMATCH PROBABILITIES
# =============================================================================

MATCH_PROBABILITY_DEFAULT = 0.3
"""Default probability for match trials in memory tasks."""

MATCH_PROBABILITY_HIGH = 0.5
"""High match probability for easier tasks."""

MATCH_PROBABILITY_LOW = 0.2
"""Low match probability for harder tasks."""

# =============================================================================
# SPIKE ENCODING PARAMETERS
# =============================================================================

MNIST_SPIKE_RATE = 0.3
"""Spike rate for MNIST image encoding (30% activity)."""

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
    "STIMULUS_STRENGTH_HIGH",
    "STIMULUS_STRENGTH_MEDIUM",
    "STIMULUS_STRENGTH_LOW",
    "PROPRIOCEPTION_NOISE_SCALE",
    "WEIGHT_INIT_SCALE_SMALL",
    "WEIGHT_INIT_SCALE_MEDIUM",
    "STIMULUS_NOISE_SCALE",
    "FEATURE_NOISE_MATCH",
    "DATASET_WEIGHT_MNIST",
    "DATASET_WEIGHT_TEMPORAL",
    "DATASET_WEIGHT_PHONOLOGY",
    "DATASET_WEIGHT_GAZE",
    "SENSORIMOTOR_WEIGHT_MOTOR_CONTROL",
    "SENSORIMOTOR_WEIGHT_REACHING",
    "SENSORIMOTOR_WEIGHT_MANIPULATION",
    "SENSORIMOTOR_WEIGHT_PREDICTION",
    "WORKSPACE_SIZE_DEFAULT",
    "REWARD_SCALE_PREDICTION",
    "TARGET_POSITION_MARGIN",
    "REACHING_SUCCESS_THRESHOLD",
    "MATCH_PROBABILITY_DEFAULT",
    "MATCH_PROBABILITY_HIGH",
    "MATCH_PROBABILITY_LOW",
    "MNIST_SPIKE_RATE",
    "FEATURE_INCREMENT_BASE",
    "FEATURE_INCREMENT_COLUMN",
    "FEATURE_INCREMENT_INTERACTION",
]
