"""
Training and Evaluation Threshold Constants

This module centralizes threshold values used across training, curriculum,
and evaluation code. Replaces scattered magic numbers with named constants
that document their biological/empirical basis.

Import and use these constants instead of hardcoded values:
    from thalia.training.constants import FIRING_RATE_MINIMUM, PERFORMANCE_EXCELLENT

Author: Thalia Project
Date: December 12, 2025
"""

# =============================================================================
# Firing Rate Health Checks
# =============================================================================

FIRING_RATE_MINIMUM = 0.01
"""Minimum firing rate threshold (1%).

Below this rate, neurons are considered silent/dead. Used in health checks
and curriculum progression to ensure all regions remain active.

Biological basis: ~0.1-1 Hz baseline firing in cortex (0.001-0.01 normalized rate)
"""

FIRING_RATE_SILENCE_THRESHOLD = 0.01
"""Silence detection threshold (1%).

Alias for FIRING_RATE_MINIMUM. Used in diagnostics to flag silent regions.
Matches value in visualization/constants.py for consistency.
"""

FIRING_RATE_RUNAWAY_THRESHOLD = 0.9
"""Runaway excitation threshold (90%).

Above this rate, region shows pathological hyperactivity (seizure-like).
Used in health checks to detect unstable dynamics.

Biological basis: Normal cortical firing rarely exceeds 20-30 Hz (0.2-0.3 normalized)
"""

# =============================================================================
# Performance Thresholds (Task Success)
# =============================================================================

PERFORMANCE_EXCELLENT = 0.95
"""Excellent performance threshold (95%).

Tasks achieving this accuracy are considered mastered. Used in curriculum
progression to determine when to increase difficulty.

Empirical basis: Standard criterion in ML/neuroscience literature
"""

PERFORMANCE_GOOD = 0.9
"""Good performance threshold (90%).

Tasks achieving this accuracy are well-learned but not mastered.
Used in curriculum staging and reward shaping.

Empirical basis: Common "passing grade" in educational literature
"""

PERFORMANCE_ACCEPTABLE = 0.85
"""Acceptable performance threshold (85%).

Minimum performance for curriculum progression. Below this, revisit stage.
"""

PERFORMANCE_POOR = 0.7
"""Poor performance threshold (70%).

Below this, current difficulty is too high. May trigger difficulty reduction.
"""

# =============================================================================
# Curriculum Progression
# =============================================================================

CURRICULUM_LOAD_THRESHOLD = 0.9
"""Cognitive overload threshold (90%).

When cognitive load exceeds this, learning becomes inefficient. Used by
CognitiveLoadMonitor to prevent overwhelming the system with too much info.

Basis: Working memory capacity limits (Cowan 2001, ~4 chunks)
"""

CURRICULUM_MARGIN = 0.1
"""Safety margin below threshold (10%).

Target cognitive load = CURRICULUM_LOAD_THRESHOLD - CURRICULUM_MARGIN.
Provides buffer to prevent frequent threshold crossings.
"""

CURRICULUM_DIFFICULTY_MIN = 0.3
"""Minimum curriculum difficulty (30%).

Tasks easier than this provide insufficient learning signal.
Matches DIFFICULTY_RANGE_MIN in visualization constants.
"""

CURRICULUM_DIFFICULTY_MAX = 0.9
"""Maximum curriculum difficulty (90%).

Tasks harder than this are too frustrating and impede learning.
Matches DIFFICULTY_RANGE_MAX in visualization constants.
"""

# =============================================================================
# Calibration Metrics (Metacognition)
# =============================================================================

CALIBRATION_EXCELLENT_ECE = 0.05
"""Expected Calibration Error - excellent threshold (5%).

Below this ECE, confidence estimates are well-calibrated to actual accuracy.
Used in metacognition evaluation.

Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
Note: visualization/constants.py has 0.10, this is stricter criterion
"""

CALIBRATION_GOOD_ECE = 0.10
"""Expected Calibration Error - good threshold (10%).

Reasonable calibration quality. Most models achieve this with temperature scaling.
"""

CALIBRATION_ACCEPTABLE_ECE = 0.15
"""Expected Calibration Error - acceptable threshold (15%).

Calibration is present but not optimal. Room for improvement.
"""

# =============================================================================
# Reward Shaping (Sensorimotor Tasks)
# =============================================================================

REWARD_MOVEMENT_THRESHOLD = -0.5
"""Threshold for movement reward (normalized).

If episode reward > this, agent is making progress. Used in dataset loaders
for basic movement tasks.
"""

REWARD_SMALL_SUCCESS = 0.1
"""Small positive reward for partial success.

Used to shape behavior when full task completion is not achieved.
"""

REWARD_REACHING_THRESHOLD = -0.3
"""Threshold for reaching task success.

Above this, reaching target is considered successful (manipulation tasks).
"""

REWARD_HIGH_SUCCESS = 0.9
"""High reward threshold indicating clear success.

Used to mark episodes where task was definitively completed.
"""

REWARD_MANIPULATION_BASE = 0.5
"""Base reward for manipulation tasks.

Starting reward when object is successfully grasped/manipulated.
"""

# =============================================================================
# Learning Rate Scaling
# =============================================================================

LEARNING_RATE_SCALE_SMALL = 0.05
"""Small learning rate scale (5%).

Used for slow, stable learning (e.g., PFC modulation weights).
Matches WEIGHT_INIT_SCALE_SMALL from neuron_constants.
"""

# =============================================================================
# Error Tolerance
# =============================================================================

PREDICTION_ERROR_SMALL = 0.05
"""Small prediction error threshold (5%).

Below this, predictions are considered accurate. Used in stage evaluation
and predictive coding.
"""

PREDICTION_ERROR_ACCEPTABLE = 0.1
"""Acceptable prediction error threshold (10%).

Reasonable accuracy, learning is working.
"""
