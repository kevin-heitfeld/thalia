# pyright: strict
"""
Time conversion constants for spike-based simulations.

This module defines constants for converting between milliseconds and seconds,
which is common in neuroscience simulations (timesteps in ms, frequencies in Hz).

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

import math

# ============================================================================
# TIME UNIT CONVERSIONS
# ============================================================================

MS_PER_SECOND = 1000.0
"""Milliseconds per second (1000.0 ms/s)."""

SECONDS_PER_MS = 1.0 / 1000.0
"""Seconds per millisecond (0.001 s/ms)."""

# ============================================================================
# PHASE AND OSCILLATION CONSTANTS
# ============================================================================

TAU = 2.0 * math.pi
"""Full circle in radians (τ ≈ 6.283185307179586).

Tau (τ) represents one complete turn, making circle mathematics more intuitive:
- 1/4 turn = τ/4 (vs π/2)
- 1/2 turn = τ/2 (vs π)
- Full turn = τ (vs 2π)

Common in oscillator phase calculations:
- phase_increment = TAU * freq_hz * dt_seconds
- normalized_phase = phase % TAU

Reference: https://tauday.com/tau-manifesto
"""

TWO_PI = TAU  # Alias for compatibility
"""Alias for TAU. Use TAU for new code (team Tau!)."""

# Common uses:
# - firing_rate_hz = spike_rate * (MS_PER_SECOND / dt_ms)
# - phase_increment = TAU * freq_hz * (dt_ms * SECONDS_PER_MS)
# - normalized_phase = phase % TAU

__all__ = [
    "MS_PER_SECOND",
    "SECONDS_PER_MS",
    "TAU",
    "TWO_PI",
]
