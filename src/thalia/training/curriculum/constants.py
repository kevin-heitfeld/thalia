"""
Curriculum Training Constants

This module centralizes threshold values and hyperparameters used across
curriculum training, evaluation, and safety systems.

Rationale:
==========
- Named constants clarify biological/engineering rationale
- Centralized values enable easier tuning and experimentation
- Comments explain "why this value" rather than "what is this value"
- Facilitates coordinated changes across related thresholds

Organization:
=============
1. Performance Monitoring Thresholds
2. Safety System Thresholds (Graceful Degradation)
3. Stage Evaluation Thresholds
4. Noise Adaptation Thresholds
5. Cognitive Load Thresholds

Author: Thalia Project
Date: December 22, 2025
"""

# =============================================================================
# PERFORMANCE MONITORING THRESHOLDS
# =============================================================================
# Used by CognitiveLoadMonitor and stage-specific monitors

# Working Memory Critical Thresholds
WM_CRITICAL_FIRING_THRESHOLD = 0.65  # Minimum acceptable WM activity (Stage 1)
WM_CRITICAL_FIRING_THRESHOLD_GENERAL = 0.60  # General threshold (all stages)

# Theta Oscillation Stability
THETA_VARIANCE_MAX_STRICT = 0.18  # Maximum theta phase variance (Stage 1)
THETA_VARIANCE_MAX_GENERAL = 0.20  # General variance threshold

# Performance Degradation Warnings
PERFORMANCE_DROP_WARNING_STRICT = 0.08  # 8% drop triggers warning (Stage 1)
PERFORMANCE_DROP_WARNING_GENERAL = 0.10  # 10% drop triggers warning (general)


# =============================================================================
# SAFETY SYSTEM THRESHOLDS (GRACEFUL DEGRADATION)
# =============================================================================
# Used by GracefulDegradationManager to determine intervention levels

# Performance Drop Thresholds (percentage of baseline)
SAFETY_CRITICAL_THRESHOLD = 0.30  # 30% performance drop → emergency shutdown
SAFETY_LIMITED_THRESHOLD = 0.50   # 50% performance drop → partial shutdown
SAFETY_DEGRADABLE_THRESHOLD = 0.70  # 70% performance drop → intervention needed

# Module Classification
CRITICAL_SYSTEMS = {'working_memory', 'oscillators', 'replay'}  # Cannot degrade
DEGRADABLE_SYSTEMS = {'language', 'grammar', 'reading'}  # Can degrade gracefully
LIMITED_DEGRADATION = {'vision', 'phonology'}  # Limited degradation allowed


# =============================================================================
# STAGE EVALUATION THRESHOLDS
# =============================================================================
# Minimum performance requirements for stage progression

# Excitation Control
RUNAWAY_EXCITATION_THRESHOLD = 0.8  # Maximum acceptable firing rate

# BCM Convergence
BCM_DRIFT_THRESHOLD = 0.01  # Maximum threshold drift for convergence

# Stage 0 (Sensorimotor Foundation) - High requirements
STAGE0_MNIST_ACCURACY = 0.95  # 95% classification accuracy
STAGE0_PHONOLOGY_ACCURACY = 0.90  # 90% phoneme recognition
STAGE0_TEMPORAL_ACCURACY = 0.85  # 85% temporal sequence prediction
STAGE0_STABILITY_THRESHOLD = 0.05  # 5% maximum variance

# Stage 1 (Episodic Memory & Working Memory) - High requirements
STAGE1_EPISODIC_ACCURACY = 0.95  # 95% episode retrieval
STAGE1_WM_MAINTENANCE = 0.90  # 90% working memory maintenance
STAGE1_THETA_COORDINATION = 0.90  # 90% theta phase coordination

# Stage 2 (Grammar & Language Structure) - Moderate requirements
STAGE2_GRAMMAR_ACCURACY = 0.80  # 80% grammar correctness

# Stage 3 (Reading Comprehension) - Moderate requirements
STAGE3_OBJECT_TRACKING = 0.70  # 70% multi-object tracking
STAGE3_SCENE_UNDERSTANDING = 0.80  # 80% scene understanding
STAGE3_READING_COMPREHENSION = 0.85  # 85% reading comprehension
STAGE3_METACOGNITION_THRESHOLD = 0.70  # 70% metacognitive accuracy


# =============================================================================
# NOISE ADAPTATION THRESHOLDS
# =============================================================================
# Used by NoiseScheduler for performance-based noise adaptation

# Performance-Based Noise Scaling
NOISE_PERFORMANCE_LOW = 0.6  # Below this, reduce noise (struggling)
NOISE_PERFORMANCE_HIGH = 0.85  # Above this, increase noise (ready for challenge)

# Criticality-Based Noise Scaling
NOISE_CRITICALITY_BOOST = 1.5  # Multiply noise when subcritical (increase exploration)
NOISE_CRITICALITY_REDUCTION = 0.5  # Multiply noise when supercritical (reduce chaos)


# =============================================================================
# COGNITIVE LOAD THRESHOLDS
# =============================================================================
# Used by CognitiveLoadMonitor to detect overload

COGNITIVE_LOAD_THRESHOLD_DEFAULT = 0.9  # 90% load triggers intervention
COGNITIVE_LOAD_CHECK_INTERVAL = 1000  # Check every 1000 steps


# =============================================================================
# STAGE PROGRESSION GATE CHECKS
# =============================================================================
# Consolidated thresholds for readiness gates

# Minimum steps before progression consideration
MIN_STEPS_STAGE0 = 50000  # Sensorimotor foundation
MIN_STEPS_STAGE1 = 100000  # Episodic/WM
MIN_STEPS_STAGE2 = 150000  # Grammar
MIN_STEPS_STAGE3 = 200000  # Reading

# Stability window requirements (consecutive successful checks)
STABILITY_WINDOW_STEPS = 20000  # 20k steps of stable performance
BCM_CONVERGENCE_WINDOW = 50000  # 50k steps for BCM stabilization


__all__ = [
    # Performance Monitoring
    'WM_CRITICAL_FIRING_THRESHOLD',
    'WM_CRITICAL_FIRING_THRESHOLD_GENERAL',
    'THETA_VARIANCE_MAX_STRICT',
    'THETA_VARIANCE_MAX_GENERAL',
    'PERFORMANCE_DROP_WARNING_STRICT',
    'PERFORMANCE_DROP_WARNING_GENERAL',

    # Safety System
    'SAFETY_CRITICAL_THRESHOLD',
    'SAFETY_LIMITED_THRESHOLD',
    'SAFETY_DEGRADABLE_THRESHOLD',
    'CRITICAL_SYSTEMS',
    'DEGRADABLE_SYSTEMS',
    'LIMITED_DEGRADATION',

    # Stage Evaluation
    'RUNAWAY_EXCITATION_THRESHOLD',
    'BCM_DRIFT_THRESHOLD',
    'STAGE0_MNIST_ACCURACY',
    'STAGE0_PHONOLOGY_ACCURACY',
    'STAGE0_TEMPORAL_ACCURACY',
    'STAGE0_STABILITY_THRESHOLD',
    'STAGE1_EPISODIC_ACCURACY',
    'STAGE1_WM_MAINTENANCE',
    'STAGE1_THETA_COORDINATION',
    'STAGE2_GRAMMAR_ACCURACY',
    'STAGE3_OBJECT_TRACKING',
    'STAGE3_SCENE_UNDERSTANDING',
    'STAGE3_READING_COMPREHENSION',
    'STAGE3_METACOGNITION_THRESHOLD',

    # Noise Adaptation
    'NOISE_PERFORMANCE_LOW',
    'NOISE_PERFORMANCE_HIGH',
    'NOISE_CRITICALITY_BOOST',
    'NOISE_CRITICALITY_REDUCTION',

    # Cognitive Load
    'COGNITIVE_LOAD_THRESHOLD_DEFAULT',
    'COGNITIVE_LOAD_CHECK_INTERVAL',

    # Stage Progression
    'MIN_STEPS_STAGE0',
    'MIN_STEPS_STAGE1',
    'MIN_STEPS_STAGE2',
    'MIN_STEPS_STAGE3',
    'STABILITY_WINDOW_STEPS',
    'BCM_CONVERGENCE_WINDOW',
]
