"""
Homeostasis Constants - Target firing rates, metabolic budgets.

Consolidated from regulation/homeostasis_constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
"""

from __future__ import annotations

# =============================================================================
# TARGET FIRING RATES (Hz)
# =============================================================================

TARGET_FIRING_RATE_STANDARD = 5.0
"""Standard target firing rate for pyramidal neurons (5 Hz)."""

TARGET_FIRING_RATE_LOW = 1.0
"""Low target firing rate (1 Hz) for very sparse representations."""

TARGET_FIRING_RATE_MEDIUM = 10.0
"""Medium target firing rate (10 Hz) for moderate activity levels."""

TARGET_FIRING_RATE_HIGH = 30.0
"""High target firing rate (30 Hz) for interneurons."""

TARGET_FIRING_RATE_INTERNEURON = 20.0
"""Typical firing rate for fast-spiking interneurons (20 Hz)."""

TARGET_FIRING_RATE_SPARSE = 2.0
"""Sparse coding target (2 Hz) for energy-efficient representations."""

# =============================================================================
# HOMEOSTATIC TIME CONSTANTS (milliseconds)
# =============================================================================

HOMEOSTATIC_TAU_FAST = 100.0
"""Fast homeostatic adaptation (100ms)."""

HOMEOSTATIC_TAU_STANDARD = 1000.0
"""Standard homeostatic time constant (1 second)."""

HOMEOSTATIC_TAU_SLOW = 10000.0
"""Slow homeostatic adaptation (10 seconds)."""

# =============================================================================
# METABOLIC PARAMETERS
# =============================================================================

METABOLIC_COST_SPIKE = 1.0
"""Energy cost per spike (arbitrary units)."""

METABOLIC_COST_WEIGHT = 0.1
"""Energy cost per unit synaptic weight (arbitrary units)."""

METABOLIC_BUDGET_DEFAULT = 1000.0
"""Default metabolic budget per timestep (arbitrary units)."""

# =============================================================================
# SCALING PARAMETERS
# =============================================================================

SYNAPTIC_SCALING_RATE = 0.01
"""Rate of synaptic scaling for homeostasis."""

SYNAPTIC_SCALING_MIN = 0.1
"""Minimum scaling factor (prevents complete shutdown)."""

SYNAPTIC_SCALING_MAX = 10.0
"""Maximum scaling factor (prevents runaway growth)."""

INTRINSIC_PLASTICITY_RATE = 0.001
"""Rate of intrinsic excitability adjustment."""

# =============================================================================
# FIRING RATE PARAMETERS
# =============================================================================

FIRING_RATE_WINDOW_MS = 100.0
"""Time window for estimating instantaneous firing rate (100ms)."""

MIN_FIRING_RATE_HZ = 0.1
"""Minimum acceptable firing rate before considering neuron dead (0.1 Hz)."""

MAX_FIRING_RATE_HZ = 100.0
"""Maximum acceptable firing rate before considering runaway activity (100 Hz)."""

# =============================================================================
# SPARSITY CONSTRAINTS
# =============================================================================

SPARSITY_TARGET_LOW = 0.05
"""Low sparsity target (5% active)."""

SPARSITY_TARGET_MEDIUM = 0.10
"""Medium sparsity target (10% active)."""

SPARSITY_TARGET_HIGH = 0.20
"""High sparsity target (20% active)."""


__all__ = [
    "TARGET_FIRING_RATE_STANDARD",
    "TARGET_FIRING_RATE_LOW",
    "TARGET_FIRING_RATE_MEDIUM",
    "TARGET_FIRING_RATE_HIGH",
    "TARGET_FIRING_RATE_INTERNEURON",
    "TARGET_FIRING_RATE_SPARSE",
    "HOMEOSTATIC_TAU_FAST",
    "HOMEOSTATIC_TAU_STANDARD",
    "HOMEOSTATIC_TAU_SLOW",
    "METABOLIC_COST_SPIKE",
    "METABOLIC_COST_WEIGHT",
    "METABOLIC_BUDGET_DEFAULT",
    "SYNAPTIC_SCALING_RATE",
    "SYNAPTIC_SCALING_MIN",
    "SYNAPTIC_SCALING_MAX",
    "INTRINSIC_PLASTICITY_RATE",
    "FIRING_RATE_WINDOW_MS",
    "MIN_FIRING_RATE_HZ",
    "MAX_FIRING_RATE_HZ",
    "SPARSITY_TARGET_LOW",
    "SPARSITY_TARGET_MEDIUM",
    "SPARSITY_TARGET_HIGH",
]
