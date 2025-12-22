"""
Learning rate and plasticity constants used across Thalia.

This module defines standard learning rate values and plasticity time constants
for different learning rules, ensuring biological consistency and eliminating
magic numbers scattered throughout the codebase.

Biological Basis:
=================

Learning Rates:
--------------
Learning rates vary by learning rule and brain region:
- **STDP** (0.001): Spike-timing dependent plasticity, relatively slow
- **BCM** (0.01): Bienenstock-Cooper-Munro, moderate rate
- **Three-Factor** (0.001): Striatum dopamine-modulated plasticity
- **Hebbian** (0.01): Simple correlation-based learning
- **Error-Corrective** (0.005): Cerebellum supervised learning

Eligibility Traces:
------------------
- **Standard** (1000ms = 1s): Typical for reinforcement learning
- **Short** (500ms): Fast temporal credit assignment
- **Long** (2000ms): Delayed reward tasks (Yagishita et al., 2014)

BCM Threshold:
-------------
- **Standard** (5000ms = 5s): Sliding threshold adaptation rate
- Longer than typical synaptic plasticity to provide stability

References:
-----------
- Yagishita et al. (2014): A critical time window for dopamine actions
- Bi & Poo (1998): Synaptic modifications in cultured hippocampal neurons
- Bienenstock, Cooper & Munro (1982): BCM theory of visual cortex development

Usage:
======
    from thalia.regulation.learning_constants import (
        LEARNING_RATE_STDP,
        LEARNING_RATE_BCM,
        TAU_ELIGIBILITY_STANDARD
    )

    config = LearningConfig(
        learning_rate=LEARNING_RATE_STDP,
        eligibility_tau_ms=TAU_ELIGIBILITY_STANDARD
    )

Author: Thalia Project
Date: December 11, 2025
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
"""Precision weight learning rate for predictive coding networks.

Controls adaptation of precision (inverse variance) estimates in
hierarchical predictive coding models. Slower than prediction weights
to maintain stability of uncertainty estimates.
"""

# Error-Corrective Learning (Cerebellum)
LEARNING_RATE_ERROR_CORRECTIVE = 0.005
"""Supervised error-corrective learning rate (delta rule)."""

# Default/Generic
LEARNING_RATE_DEFAULT = 0.01
"""Default learning rate when specific rule is not specified."""

# =============================================================================
# ELIGIBILITY TRACE TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_ELIGIBILITY_STANDARD = 1000.0
"""Standard eligibility trace time constant (1 second).

Used for typical reinforcement learning tasks with moderate delays
between action and reward.
"""

TAU_ELIGIBILITY_SHORT = 500.0
"""Short eligibility trace (500ms).

For fast temporal credit assignment in rapidly changing environments.
"""

TAU_ELIGIBILITY_LONG = 2000.0
"""Long eligibility trace (2 seconds).

For delayed reward tasks where credit assignment spans longer intervals.
Based on Yagishita et al. (2014) findings on dopamine action windows.
"""

# =============================================================================
# BCM THRESHOLD TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_BCM_THRESHOLD = 5000.0
"""BCM sliding threshold adaptation time constant (5 seconds).

Longer than synaptic plasticity time constants to provide stability.
The threshold adapts slowly based on average postsynaptic activity squared.
"""

# =============================================================================
# STDP AMPLITUDE CONSTANTS (dimensionless)
# =============================================================================

STDP_A_PLUS_CORTEX = 0.01
"""LTP amplitude for cortical STDP (potentiation when post follows pre).

Biologically realistic value for cortical synapses. This controls the
strength of long-term potentiation when postsynaptic spikes follow
presynaptic spikes within the STDP time window.

References:
- Bi & Poo (1998): Values range 0.005-0.02 for cultured neurons
- Clopath et al. (2010): 0.01 for cortical models
"""

STDP_A_MINUS_CORTEX = 0.012
"""LTD amplitude for cortical STDP (depression when pre follows post).

Slightly larger than A+ for stability (LTD > LTP prevents runaway
potentiation). This controls the strength of long-term depression
when presynaptic spikes follow postsynaptic spikes.

Biologically, LTD is often slightly stronger than LTP to provide
homeostatic balance and prevent saturation.

References:
- Bi & Poo (1998): Asymmetric STDP window
- Sjöström et al. (2001): Spike-timing dependent plasticity in neocortex
"""

STDP_A_PLUS_HIPPOCAMPUS = 0.02
"""LTP amplitude for hippocampal STDP (stronger than cortex).

Hippocampus shows stronger STDP for rapid episodic memory formation.
"""

STDP_A_MINUS_HIPPOCAMPUS = 0.022
"""LTD amplitude for hippocampal STDP (stronger than cortex).

Hippocampus shows stronger plasticity for rapid episodic memory formation.
"""

TAU_BCM_THRESHOLD_FAST = 2000.0
"""Fast BCM threshold adaptation (2 seconds) for rapid environment changes."""

TAU_BCM_THRESHOLD_SLOW = 10000.0
"""Slow BCM threshold adaptation (10 seconds) for stable learning."""

# =============================================================================
# STDP TIME CONSTANTS (milliseconds)
# =============================================================================

TAU_STDP_PLUS = 20.0
"""STDP potentiation time constant (20ms).

Time window for pre → post spike causality leading to LTP.
"""

TAU_STDP_MINUS = 20.0
"""STDP depression time constant (20ms).

Time window for post → pre spike anti-causality leading to LTD.
Some models use asymmetric values (e.g., 20ms vs 30ms).
"""

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
"""Weight initialization scale for predictive coding pathways.

Used for prediction and encoding weights in predictive coding networks.
Small scale (0.1) ensures stable predictions initially.
"""

WEIGHT_INIT_SCALE_RECURRENT = 0.01
"""Weight initialization scale for recurrent/associative connections.

Used for sequence memory and pattern completion networks.
Very small scale (0.01) prevents runaway recurrence.
"""

# =============================================================================
# ACTIVITY TRACKING PARAMETERS
# =============================================================================

EMA_DECAY_FAST = 0.99
"""Fast exponential moving average decay for activity history.

Corresponds to ~100 timestep averaging window.
Used for short-term activity tracking and homeostasis.
"""

EMA_DECAY_SLOW = 0.999
"""Slow exponential moving average decay for long-term tracking.

Corresponds to ~1000 timestep averaging window.
Used for stable long-term statistics.
"""

# =============================================================================
# NOISE PARAMETERS
# =============================================================================

WM_NOISE_STD_DEFAULT = 0.01
"""Default working memory noise standard deviation.

Adds stochasticity to working memory updates and predictions.
Based on neural variability in prefrontal cortex.

Note: Changed from 0.02 to 0.01 to match inline usage in prefrontal.py (line 692).
"""

# =============================================================================
# ACTIVITY DETECTION THRESHOLDS
# =============================================================================

SILENCE_DETECTION_THRESHOLD = 0.001
"""Firing rate threshold below which a region is considered silent.

Used in health monitoring to detect pathological silence states.
A region with mean firing rate < 0.001 (0.1%) indicates:
- No effective computation occurring
- Potential dead neurons or vanished gradients
- Insufficient input drive or excessive inhibition

Biological context:
- Cortical firing rates: 1-10 Hz in active states
- In 1ms timesteps: 0.001-0.01 spike probability per neuron
- Threshold of 0.001 allows detection of near-zero activity
"""

# =============================================================================
# PHASE INITIALIZATION
# =============================================================================

PHASE_RANGE_2PI = 6.283185307179586  # 2π
"""
Full phase range [0, 2π) for oscillator phase preferences.

Used for initializing random phase preferences in oscillator-coupled neurons.
"""


__all__ = [
    # Core learning rates
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
    # Plasticity time constants
    "TAU_ELIGIBILITY_STANDARD",
    "TAU_ELIGIBILITY_SHORT",
    "TAU_ELIGIBILITY_LONG",
    "TAU_BCM_THRESHOLD",
    "TAU_BCM_THRESHOLD_FAST",
    "TAU_BCM_THRESHOLD_SLOW",
    "TAU_STDP_PLUS",
    "TAU_STDP_MINUS",
    # STDP parameters
    "STDP_A_PLUS_CORTEX",
    "STDP_A_MINUS_CORTEX",
    "STDP_A_PLUS_HIPPOCAMPUS",
    "STDP_A_MINUS_HIPPOCAMPUS",
    # Activity tracking
    "EMA_DECAY_FAST",
    "EMA_DECAY_SLOW",
    # Noise parameters
    "WM_NOISE_STD_DEFAULT",
    # Activity detection
    "SILENCE_DETECTION_THRESHOLD",
    # Phase initialization
    "PHASE_RANGE_2PI",
]
