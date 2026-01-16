"""
Training and Evaluation Threshold Constants

This module centralizes threshold values used across training, curriculum,
and evaluation code. Replaces scattered magic numbers with named constants
that document their biological/empirical basis.

Consolidated from training/constants.py to centralized constants/ directory.

Import and use these constants instead of hardcoded values:
    from thalia.constants.training import FIRING_RATE_MINIMUM, PERFORMANCE_EXCELLENT

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2 - Consolidation)
"""

from __future__ import annotations

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

# =============================================================================
# CURRICULUM TRAINING CONSTANTS
# =============================================================================
# Consolidated from training/curriculum/constants.py

from enum import Enum
from typing import Tuple


class AttentionStage(Enum):
    """Developmental stages of attention control.

    Represents the shift from reactive (bottom-up) to proactive (top-down)
    attention control across development, matching curriculum stages.

    Biological basis:
    - Infant: Pure bottom-up (novelty, salience, motion)
    - Toddler: Mostly bottom-up with emerging goal-directed control
    - Preschool: Balanced control (conflict monitoring emerges)
    - School-age: Top-down dominant (strategic attention allocation)

    Implementation:
    - Controls thalamic gating strength (alpha suppression)
    - Modulates PFC→thalamus feedback gain
    - Adjusts NE gain modulation sensitivity

    References:
    - Posner & Petersen (1990): Attention networks
    - Colombo (2001): Infant attention development
    - Diamond (2013): Executive function emergence
    """
    INFANT = 0      # Stage 0: Pure bottom-up (100% reactive)
    TODDLER = 1     # Stage 1: Mostly bottom-up (70% reactive, 30% goal-directed)
    PRESCHOOL = 2   # Stage 2: Balanced (50% reactive, 50% goal-directed)
    SCHOOL_AGE = 3  # Stage 3+: Top-down dominant (30% reactive, 70% goal-directed)


def get_attention_weights(stage: AttentionStage) -> Tuple[float, float]:
    """Get bottom-up and top-down attention weights for a curriculum stage.

    Args:
        stage: Current attention developmental stage

    Returns:
        Tuple of (bottom_up_weight, top_down_weight) normalized to sum to 1.0
    """
    weights = {
        AttentionStage.INFANT: (1.0, 0.0),
        AttentionStage.TODDLER: (0.7, 0.3),
        AttentionStage.PRESCHOOL: (0.5, 0.5),
        AttentionStage.SCHOOL_AGE: (0.3, 0.7),
    }
    return weights[stage]


# Performance Monitoring Thresholds
WM_CRITICAL_FIRING_THRESHOLD = 0.65  # Minimum acceptable WM activity (Stage 1)
WM_CRITICAL_FIRING_THRESHOLD_GENERAL = 0.60  # General threshold (all stages)
THETA_VARIANCE_MAX_STRICT = 0.18  # Maximum theta phase variance (Stage 1)
THETA_VARIANCE_MAX_GENERAL = 0.20  # General variance threshold
PERFORMANCE_DROP_WARNING_STRICT = 0.08  # 8% drop triggers warning (Stage 1)
PERFORMANCE_DROP_WARNING_GENERAL = 0.10  # 10% drop triggers warning (general)

# Safety System Thresholds (Graceful Degradation)
SAFETY_CRITICAL_THRESHOLD = 0.30  # 30% performance drop → emergency shutdown
SAFETY_LIMITED_THRESHOLD = 0.50   # 50% performance drop → partial shutdown
SAFETY_DEGRADABLE_THRESHOLD = 0.70  # 70% performance drop → intervention needed
CRITICAL_SYSTEMS = {'working_memory', 'oscillators', 'replay'}  # Cannot degrade
DEGRADABLE_SYSTEMS = {'language', 'grammar', 'reading'}  # Can degrade gracefully
LIMITED_DEGRADATION = {'vision', 'phonology'}  # Limited degradation allowed

# Stage Evaluation Thresholds
RUNAWAY_EXCITATION_THRESHOLD = 0.8  # Maximum acceptable firing rate
BCM_DRIFT_THRESHOLD = 0.01  # Maximum threshold drift for convergence

# Stage 0 (Sensorimotor Foundation)
STAGE0_MNIST_ACCURACY = 0.95  # 95% classification accuracy
STAGE0_PHONOLOGY_ACCURACY = 0.90  # 90% phoneme recognition
STAGE0_TEMPORAL_ACCURACY = 0.85  # 85% temporal sequence prediction
STAGE0_STABILITY_THRESHOLD = 0.05  # 5% maximum variance

# Stage 1 (Episodic Memory & Working Memory)
STAGE1_EPISODIC_ACCURACY = 0.95  # 95% episode retrieval
STAGE1_WM_MAINTENANCE = 0.90  # 90% working memory maintenance
STAGE1_THETA_COORDINATION = 0.90  # 90% theta phase coordination

# Stage 2 (Grammar & Language Structure)
STAGE2_GRAMMAR_ACCURACY = 0.80  # 80% grammar correctness

# Stage 3 (Reading Comprehension)
STAGE3_OBJECT_TRACKING = 0.70  # 70% multi-object tracking
STAGE3_SCENE_UNDERSTANDING = 0.80  # 80% scene understanding
STAGE3_READING_COMPREHENSION = 0.85  # 85% reading comprehension
STAGE3_METACOGNITION_THRESHOLD = 0.70  # 70% metacognitive accuracy

# Noise Adaptation Thresholds
NOISE_PERFORMANCE_LOW = 0.6  # Below this, reduce noise (struggling)
NOISE_PERFORMANCE_HIGH = 0.85  # Above this, increase noise (ready for challenge)
NOISE_CRITICALITY_BOOST = 1.5  # Multiply noise when subcritical (increase exploration)
NOISE_CRITICALITY_REDUCTION = 0.5  # Multiply noise when supercritical (reduce chaos)

# Cognitive Load Thresholds
COGNITIVE_LOAD_THRESHOLD_DEFAULT = 0.9  # 90% load triggers intervention
COGNITIVE_LOAD_CHECK_INTERVAL = 1000  # Check every 1000 steps

# Stage Progression Gate Checks
MIN_STEPS_STAGE0 = 50000  # Sensorimotor foundation
MIN_STEPS_STAGE1 = 100000  # Episodic/WM
MIN_STEPS_STAGE2 = 150000  # Grammar
MIN_STEPS_STAGE3 = 200000  # Reading
STABILITY_WINDOW_STEPS = 20000  # 20k steps of stable performance
BCM_CONVERGENCE_WINDOW = 50000  # 50k steps for BCM stabilization


__all__ = [
    # Firing rate health checks
    "FIRING_RATE_MINIMUM",
    "FIRING_RATE_SILENCE_THRESHOLD",
    "FIRING_RATE_RUNAWAY_THRESHOLD",
    # Performance thresholds
    "PERFORMANCE_EXCELLENT",
    "PERFORMANCE_GOOD",
    "PERFORMANCE_ACCEPTABLE",
    "PERFORMANCE_POOR",
    # Curriculum progression
    "CURRICULUM_LOAD_THRESHOLD",
    "CURRICULUM_MARGIN",
    "CURRICULUM_DIFFICULTY_MIN",
    "CURRICULUM_DIFFICULTY_MAX",
    # Calibration metrics
    "CALIBRATION_EXCELLENT_ECE",
    "CALIBRATION_GOOD_ECE",
    "CALIBRATION_ACCEPTABLE_ECE",
    # Reward shaping
    "REWARD_MOVEMENT_THRESHOLD",
    "REWARD_SMALL_SUCCESS",
    "REWARD_REACHING_THRESHOLD",
    "REWARD_HIGH_SUCCESS",
    "REWARD_MANIPULATION_BASE",
    # Learning rate scaling
    "LEARNING_RATE_SCALE_SMALL",
    # Error tolerance
    "PREDICTION_ERROR_SMALL",
    "PREDICTION_ERROR_ACCEPTABLE",
    # Curriculum training (from training/curriculum/constants.py)
    "AttentionStage",
    "get_attention_weights",
    "WM_CRITICAL_FIRING_THRESHOLD",
    "WM_CRITICAL_FIRING_THRESHOLD_GENERAL",
    "THETA_VARIANCE_MAX_STRICT",
    "THETA_VARIANCE_MAX_GENERAL",
    "PERFORMANCE_DROP_WARNING_STRICT",
    "PERFORMANCE_DROP_WARNING_GENERAL",
    "SAFETY_CRITICAL_THRESHOLD",
    "SAFETY_LIMITED_THRESHOLD",
    "SAFETY_DEGRADABLE_THRESHOLD",
    "CRITICAL_SYSTEMS",
    "DEGRADABLE_SYSTEMS",
    "LIMITED_DEGRADATION",
    "RUNAWAY_EXCITATION_THRESHOLD",
    "BCM_DRIFT_THRESHOLD",
    "STAGE0_MNIST_ACCURACY",
    "STAGE0_PHONOLOGY_ACCURACY",
    "STAGE0_TEMPORAL_ACCURACY",
    "STAGE0_STABILITY_THRESHOLD",
    "STAGE1_EPISODIC_ACCURACY",
    "STAGE1_WM_MAINTENANCE",
    "STAGE1_THETA_COORDINATION",
    "STAGE2_GRAMMAR_ACCURACY",
    "STAGE3_OBJECT_TRACKING",
    "STAGE3_SCENE_UNDERSTANDING",
    "STAGE3_READING_COMPREHENSION",
    "STAGE3_METACOGNITION_THRESHOLD",
    "NOISE_PERFORMANCE_LOW",
    "NOISE_PERFORMANCE_HIGH",
    "NOISE_CRITICALITY_BOOST",
    "NOISE_CRITICALITY_REDUCTION",
    "COGNITIVE_LOAD_THRESHOLD_DEFAULT",
    "COGNITIVE_LOAD_CHECK_INTERVAL",
    "MIN_STEPS_STAGE0",
    "MIN_STEPS_STAGE1",
    "MIN_STEPS_STAGE2",
    "MIN_STEPS_STAGE3",
    "STABILITY_WINDOW_STEPS",
    "BCM_CONVERGENCE_WINDOW",
]
