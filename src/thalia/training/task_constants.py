"""
Task-specific constants for training curriculum.

This module consolidates magic numbers used across task loaders and task modules,
making them discoverable and tunable from a single location.

Biological Motivation:
=====================
These constants define task difficulty levels, spike probabilities for motor
exploration, and dataset sampling weights during curriculum stages.

Usage:
======
    from thalia.training.task_constants import (
        SPIKE_PROBABILITY_LOW,
        DATASET_WEIGHT_MNIST
    )
    
    motor_spikes = torch.rand(n_motor, device=device) < SPIKE_PROBABILITY_LOW

Author: Thalia Project
Date: December 12, 2025
"""

# =============================================================================
# MOTOR SPIKE PROBABILITIES
# =============================================================================

SPIKE_PROBABILITY_LOW = 0.15
"""Low motor activity spike probability (15%).

Used for gentle motor exploration during early stages.
Corresponds to sparse motor patterns during observation/planning.
"""

SPIKE_PROBABILITY_MEDIUM = 0.30
"""Medium motor activity spike probability (30%).

Used for moderate motor activity during reaching tasks.
Balance between exploration and energy efficiency.
"""

SPIKE_PROBABILITY_HIGH = 0.50
"""High motor activity spike probability (50%).

Used for active manipulation and high-complexity motor tasks.
Dense motor patterns during active control.
"""

# =============================================================================
# DATASET SAMPLING WEIGHTS (Birth Stage - Stage 0)
# =============================================================================

DATASET_WEIGHT_MNIST = 0.40
"""MNIST dataset sampling weight for Birth stage (40%).

Visual foundation through handwritten digit recognition.
"""

DATASET_WEIGHT_TEMPORAL = 0.20
"""Temporal sequence dataset weight for Birth stage (20%).

Sequence learning and temporal prediction.
"""

DATASET_WEIGHT_PHONOLOGY = 0.30
"""Phonology dataset weight for Birth stage (30%).

Phoneme discrimination and early language processing.
"""

DATASET_WEIGHT_GAZE = 0.10
"""Gaze following dataset weight for Birth stage (10%).

Social attention and joint attention mechanisms.
"""

# =============================================================================
# SENSORIMOTOR TASK WEIGHTS (Stage 0)
# =============================================================================

SENSORIMOTOR_WEIGHT_MOTOR_CONTROL = 0.25
"""Motor control task weight (25%).

Basic motor pattern generation and proprioceptive integration.
"""

SENSORIMOTOR_WEIGHT_REACHING = 0.25
"""Reaching task weight (25%).

Visuomotor coordination for goal-directed reaching.
"""

SENSORIMOTOR_WEIGHT_MANIPULATION = 0.25
"""Manipulation task weight (25%).

Object manipulation and fine motor control.
"""

SENSORIMOTOR_WEIGHT_PREDICTION = 0.25
"""Prediction task weight (25%).

Forward model learning and sensory prediction.
"""

# =============================================================================
# NOISE AND SCALE PARAMETERS
# =============================================================================

PROPRIOCEPTION_NOISE_SCALE = 0.1
"""Proprioceptive noise standard deviation.

Realistic sensor noise in proprioceptive feedback (joint angles, velocities).
Typical biological proprioception has ~10% noise.
"""

WEIGHT_INIT_SCALE_SMALL = 0.05
"""Small weight initialization scale.

Used for stimulus perturbations and small random variations.
"""

STIMULUS_STRENGTH_HIGH = 1.0
"""High stimulus strength for salient stimuli.

Strong sensory input for attention-demanding tasks.
"""

# =============================================================================
# WORKSPACE AND ENVIRONMENT PARAMETERS
# =============================================================================

WORKSPACE_SIZE_DEFAULT = 1.0
"""Default workspace size for sensorimotor tasks.

Normalized workspace dimensions (0 to 1) for reaching/manipulation.
"""

# =============================================================================
# PROBABILITY THRESHOLDS
# =============================================================================

MATCH_PROBABILITY_DEFAULT = 0.3
"""Default match probability for n-back working memory tasks (30%).

Probability that current stimulus matches n positions back.
Balanced difficulty for working memory challenge.
"""

# =============================================================================
# REWARD CALCULATION
# =============================================================================

REWARD_SCALE_PREDICTION = 1.0
"""Scaling factor for prediction-based rewards.

reward = max(0.0, REWARD_SCALE_PREDICTION - prediction_error)
"""
