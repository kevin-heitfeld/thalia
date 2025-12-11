"""
Homeostatic regulation constants used across Thalia.

This module defines standard values for homeostatic mechanisms that maintain
stable neural activity levels, preventing runaway excitation or silence.

Biological Basis:
=================

Target Firing Rates:
-------------------
Different neuron types have characteristic baseline firing rates:
- **Pyramidal neurons** (cortex): 1-5 Hz sparse coding
- **Interneurons** (fast-spiking): 10-30 Hz continuous activity
- **Striatal MSNs**: 1-10 Hz sparse, state-dependent
- **Hippocampal place cells**: 1-5 Hz (in-field), <0.1 Hz (out-of-field)

Homeostatic Time Constants:
---------------------------
Homeostatic plasticity operates on slower timescales than Hebbian plasticity:
- **Fast** (100ms): Rapid adaptation to input changes
- **Standard** (1s): Typical synaptic scaling
- **Slow** (10s): Stable long-term regulation
- **Very slow** (minutes-hours): Biological synaptic scaling

Mechanisms:
----------
1. **Synaptic scaling**: Global adjustment of synaptic weights
2. **Intrinsic plasticity**: Adjustment of neuronal excitability
3. **BCM threshold**: Sliding modification threshold
4. **Inhibitory plasticity**: Adjustment of inhibitory strength

References:
-----------
- Turrigiano & Nelson (2004): Homeostatic plasticity in the developing nervous system
- Turrigiano (2008): The self-tuning neuron: synaptic scaling of excitatory synapses
- Zenke et al. (2013): Synaptic plasticity in neural networks needs homeostasis

Usage:
======
    from thalia.core.homeostasis_constants import (
        TARGET_FIRING_RATE_STANDARD,
        HOMEOSTATIC_TAU_STANDARD
    )

    config = HomeostasisConfig(
        target_firing_rate_hz=TARGET_FIRING_RATE_STANDARD,
        homeostatic_tau_ms=HOMEOSTATIC_TAU_STANDARD
    )

Author: Thalia Project
Date: December 11, 2025
"""

# =============================================================================
# TARGET FIRING RATES (Hz)
# =============================================================================

TARGET_FIRING_RATE_STANDARD = 5.0
"""Standard target firing rate for pyramidal neurons (5 Hz).

Typical sparse coding regime for cortical excitatory neurons.
"""

TARGET_FIRING_RATE_LOW = 1.0
"""Low target firing rate (1 Hz) for very sparse representations."""

TARGET_FIRING_RATE_MEDIUM = 10.0
"""Medium target firing rate (10 Hz) for moderate activity levels."""

TARGET_FIRING_RATE_HIGH = 30.0
"""High target firing rate (30 Hz) for interneurons and dense representations."""

TARGET_FIRING_RATE_INTERNEURON = 20.0
"""Typical firing rate for fast-spiking interneurons (20 Hz)."""

TARGET_FIRING_RATE_SPARSE = 2.0
"""Sparse coding target (2 Hz) for energy-efficient representations."""

# =============================================================================
# HOMEOSTATIC TIME CONSTANTS (milliseconds)
# =============================================================================

HOMEOSTATIC_TAU_FAST = 100.0
"""Fast homeostatic adaptation (100ms).

For rapid responses to sudden changes in input statistics.
"""

HOMEOSTATIC_TAU_STANDARD = 1000.0
"""Standard homeostatic time constant (1 second).

Typical synaptic scaling timescale in computational models.
Biological synaptic scaling is slower (hours-days), but we use
faster timescales for practical learning.
"""

HOMEOSTATIC_TAU_SLOW = 10000.0
"""Slow homeostatic adaptation (10 seconds).

For stable, gradual regulation that doesn't interfere with
fast learning dynamics.
"""

HOMEOSTATIC_TAU_VERY_SLOW = 60000.0
"""Very slow homeostatic adaptation (1 minute).

Approaches biological timescales for long-term stability.
"""

# =============================================================================
# SYNAPTIC SCALING PARAMETERS
# =============================================================================

SYNAPTIC_SCALING_RATE = 0.001
"""Rate of synaptic scaling (global weight adjustment).

Controls how quickly all synapses scale to maintain target firing rate.
"""

SYNAPTIC_SCALING_MIN = 0.1
"""Minimum scaling factor (prevents complete shutdown)."""

SYNAPTIC_SCALING_MAX = 10.0
"""Maximum scaling factor (prevents runaway growth)."""

# =============================================================================
# INTRINSIC PLASTICITY PARAMETERS
# =============================================================================

INTRINSIC_PLASTICITY_RATE = 0.0001
"""Rate of intrinsic plasticity (threshold/excitability adjustment)."""

THRESHOLD_ADAPTATION_TAU = 1000.0
"""Time constant for adaptive threshold mechanisms (1 second)."""

# =============================================================================
# FIRING RATE ESTIMATION PARAMETERS
# =============================================================================

FIRING_RATE_WINDOW_MS = 100.0
"""Time window for estimating instantaneous firing rate (100ms)."""

FIRING_RATE_EMA_TAU = 1000.0
"""Time constant for exponential moving average of firing rate (1 second)."""

# =============================================================================
# BOUNDS AND LIMITS
# =============================================================================

MIN_FIRING_RATE_HZ = 0.1
"""Minimum acceptable firing rate before considering neuron dead (0.1 Hz)."""

MAX_FIRING_RATE_HZ = 100.0
"""Maximum acceptable firing rate before considering runaway activity (100 Hz)."""

SILENCE_THRESHOLD_HZ = 0.5
"""Firing rate below which region is considered silent (0.5 Hz)."""

SATURATION_THRESHOLD_HZ = 50.0
"""Firing rate above which region is considered saturated (50 Hz)."""
