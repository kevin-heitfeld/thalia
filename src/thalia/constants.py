# pyright: strict
"""
Time conversion constants for spike-based simulations.

This module defines constants for converting between milliseconds and seconds,
which is common in neuroscience simulations (timesteps in ms, frequencies in Hz).
"""

from __future__ import annotations

# ============================================================================
# SIMULATION DEFAULTS
# ============================================================================

DEFAULT_DT_MS = 1.0
"""Default timestep in milliseconds (1.0 ms)."""
