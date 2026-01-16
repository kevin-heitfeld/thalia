"""
Learning Constants - Consolidated from regulation/learning_constants.py

Learning rates, eligibility traces, STDP parameters, and plasticity time constants.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
"""

# =============================================================================
# LEARNING RATES (dimensionless)
# =============================================================================

# Spike-Timing Dependent Plasticity (STDP)
LEARNING_RATE_STDP = 0.001
"""Standard STDP learning rate for cortical synapses."""

LEARNING_RATE_STDP_MODERATE = 0.0015
"""Moderate STDP for pathways requiring slightly faster adaptation."""

LEARNING_RATE_STDP_FAST = 0.005
"""Fast STDP for rapid adaptation (critical periods)."""

LEARNING_RATE_STDP_SLOW = 0.0001
"""Slow STDP for stable, incremental learning."""

# Bienenstock-Cooper-Munro (BCM)
LEARNING_RATE_BCM = 0.01
"""Standard BCM learning rate for unsupervised feature learning."""

# Three-Factor Learning (Striatum)
LEARNING_RATE_THREE_FACTOR = 0.001
"""Dopamine-modulated three-factor learning rate (eligibility × dopamine)."""

# Hebbian Learning
LEARNING_RATE_HEBBIAN = 0.01
"""Basic Hebbian learning rate (pre × post)."""

LEARNING_RATE_HEBBIAN_SLOW = 0.001
"""Slow Hebbian learning for stable multimodal integration."""

LEARNING_RATE_ONE_SHOT = 0.1
"""One-shot learning rate for hippocampal episodic memory."""

# Precision Learning (Predictive Coding)
LEARNING_RATE_PRECISION = 0.001
"""Precision weight learning rate for predictive coding networks."""

# Error-Corrective Learning (Cerebellum)
LEARNING_RATE_ERROR_CORRECTIVE = 0.005
"""Supervised error-corrective learning rate (delta rule)."""

# Default/Generic
LEARNING_RATE_DEFAULT = 0.01
"""Default learning rate when specific rule is not specified."""

# =============================================================================
# LEARNING RATE PRESETS (for easy configuration)
# =============================================================================

LR_VERY_SLOW = 0.0001
"""Very slow learning for stable, incremental learning (late training)."""

LR_SLOW = 0.001
"""Standard slow learning (default for most regions)."""

LR_MODERATE = 0.01
"""Moderate learning (early training, rapid adaptation)."""

LR_FAST = 0.1
"""Fast learning (one-shot learning, critical periods)."""

# =============================================================================
# REGION-SPECIFIC LEARNING RATE DEFAULTS
# =============================================================================

LR_CORTEX_DEFAULT = LR_SLOW
"""Cortex: slow, stable learning (0.001)."""

LR_HIPPOCAMPUS_DEFAULT = LR_MODERATE
"""Hippocampus: faster for episodic memory (0.01)."""

LR_STRIATUM_DEFAULT = LR_SLOW
"""Striatum: slow RL policy improvement (0.001)."""

LR_CEREBELLUM_DEFAULT = LR_MODERATE
"""Cerebellum: error-corrective learning (0.01)."""

LR_PFC_DEFAULT = LR_SLOW
"""PFC: stable working memory maintenance (0.001)."""

# =============================================================================
# ELIGIBILITY TRACE TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_ELIGIBILITY_STANDARD = 1000.0
"""Standard eligibility trace time constant (1 second)."""

TAU_ELIGIBILITY_SHORT = 500.0
"""Short eligibility trace (500ms) for fast credit assignment."""

TAU_ELIGIBILITY_LONG = 2000.0
"""Long eligibility trace (2 seconds) for delayed rewards."""

# =============================================================================
# BCM THRESHOLD TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_BCM_THRESHOLD = 5000.0
"""BCM sliding threshold adaptation time constant (5 seconds)."""

TAU_BCM_THRESHOLD_FAST = 2000.0
"""Fast BCM threshold adaptation (2 seconds)."""

TAU_BCM_THRESHOLD_SLOW = 10000.0
"""Slow BCM threshold adaptation (10 seconds)."""

# =============================================================================
# STDP AMPLITUDE CONSTANTS (dimensionless)
# =============================================================================

STDP_A_PLUS_CORTEX = 0.01
"""LTP amplitude for cortical STDP."""

STDP_A_MINUS_CORTEX = 0.012
"""LTD amplitude for cortical STDP (slightly larger for stability)."""

STDP_A_PLUS_HIPPOCAMPUS = 0.02
"""LTP amplitude for hippocampal STDP (stronger than cortex)."""

STDP_A_MINUS_HIPPOCAMPUS = 0.022
"""LTD amplitude for hippocampal STDP."""

# =============================================================================
# STDP TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_STDP_PLUS = 20.0
"""STDP potentiation time constant (20ms)."""

TAU_STDP_MINUS = 20.0
"""STDP depression time constant (20ms)."""

# =============================================================================
# TRACE DECAY TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_TRACE_SHORT = 10.0
"""Short-term trace for fast synaptic dynamics (10ms)."""

TAU_TRACE_MEDIUM = 50.0
"""Medium-term trace for working memory operations (50ms)."""

TAU_TRACE_LONG = 200.0
"""Long-term trace for sustained activity patterns (200ms)."""

# =============================================================================
# WEIGHT INITIALIZATION SCALES
# =============================================================================

WEIGHT_INIT_SCALE_PREDICTIVE = 0.1
"""Weight initialization scale for predictive coding pathways."""

WEIGHT_INIT_SCALE_RECURRENT = 0.01
"""Weight initialization scale for recurrent/associative connections."""

WEIGHT_INIT_SCALE_SMALL = 0.01
"""Small weight initialization scale for fine-grained variation."""

WEIGHT_INIT_SCALE_MEDIUM = 0.05
"""Medium weight initialization scale for moderate variation."""

# =============================================================================
# ACTIVITY TRACKING PARAMETERS
# =============================================================================

EMA_DECAY_FAST = 0.99
"""Fast exponential moving average decay (~100 timestep window)."""

EMA_DECAY_SLOW = 0.999
"""Slow exponential moving average decay (~1000 timestep window)."""

# =============================================================================
# NOISE PARAMETERS
# =============================================================================

WM_NOISE_STD_DEFAULT = 0.01
"""Default working memory noise standard deviation."""

WORKING_MEMORY_NOISE_STD = 0.01
"""Working memory noise for prefrontal updates."""

# =============================================================================
# ACTIVITY DETECTION THRESHOLDS
# =============================================================================

SILENCE_DETECTION_THRESHOLD = 0.001
"""Firing rate threshold below which a region is considered silent."""

# =============================================================================
# PHASE INITIALIZATION
# =============================================================================

PHASE_RANGE_2PI = 6.283185307179586  # 2π
"""Full phase range [0, 2π) for oscillator phase preferences."""


__all__ = [
    "LEARNING_RATE_STDP",
    "LEARNING_RATE_STDP_MODERATE",
    "LEARNING_RATE_STDP_FAST",
    "LEARNING_RATE_STDP_SLOW",
    "LEARNING_RATE_BCM",
    "LEARNING_RATE_THREE_FACTOR",
    "LEARNING_RATE_HEBBIAN",
    "LEARNING_RATE_HEBBIAN_SLOW",
    "LEARNING_RATE_PRECISION",
    "LEARNING_RATE_ERROR_CORRECTIVE",
    "LEARNING_RATE_ONE_SHOT",
    "LEARNING_RATE_DEFAULT",
    "LR_VERY_SLOW",
    "LR_SLOW",
    "LR_MODERATE",
    "LR_FAST",
    "LR_CORTEX_DEFAULT",
    "LR_HIPPOCAMPUS_DEFAULT",
    "LR_STRIATUM_DEFAULT",
    "LR_CEREBELLUM_DEFAULT",
    "LR_PFC_DEFAULT",
    "TAU_ELIGIBILITY_STANDARD",
    "TAU_ELIGIBILITY_SHORT",
    "TAU_ELIGIBILITY_LONG",
    "TAU_BCM_THRESHOLD",
    "TAU_BCM_THRESHOLD_FAST",
    "TAU_BCM_THRESHOLD_SLOW",
    "STDP_A_PLUS_CORTEX",
    "STDP_A_MINUS_CORTEX",
    "STDP_A_PLUS_HIPPOCAMPUS",
    "STDP_A_MINUS_HIPPOCAMPUS",
    "TAU_STDP_PLUS",
    "TAU_STDP_MINUS",
    "TAU_TRACE_SHORT",
    "TAU_TRACE_MEDIUM",
    "TAU_TRACE_LONG",
    "WEIGHT_INIT_SCALE_PREDICTIVE",
    "WEIGHT_INIT_SCALE_RECURRENT",
    "WEIGHT_INIT_SCALE_SMALL",
    "WEIGHT_INIT_SCALE_MEDIUM",
    "EMA_DECAY_FAST",
    "EMA_DECAY_SLOW",
    "WM_NOISE_STD_DEFAULT",
    "WORKING_MEMORY_NOISE_STD",
    "SILENCE_DETECTION_THRESHOLD",
    "PHASE_RANGE_2PI",
]
