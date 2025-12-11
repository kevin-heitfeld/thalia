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
    from thalia.core.learning_constants import (
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

LEARNING_RATE_ONE_SHOT = 0.1
"""One-shot learning rate for hippocampal episodic memory."""

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
