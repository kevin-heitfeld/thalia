"""
Region-Specific Constants - Thalamus and striatum parameters.

These are specialized constants for specific brain region implementations.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

# =============================================================================
# THALAMUS - MODE SWITCHING
# =============================================================================

THALAMUS_BURST_THRESHOLD = -0.2
"""Membrane potential threshold for burst mode (normalized, ~-65mV biological)."""

THALAMUS_TONIC_THRESHOLD = 0.3
"""Membrane potential threshold for tonic mode (normalized, ~-55mV biological)."""

THALAMUS_BURST_SPIKE_COUNT = 3
"""Number of spikes in a thalamic burst (typical value 2-5, most common is 3)."""

THALAMUS_BURST_GAIN = 2.0
"""Amplification factor for burst mode (2x more effective than single spikes)."""

THALAMUS_MODE_THRESHOLD = 0.5
"""Threshold for burst/tonic mode detection (0=burst, 1=tonic)."""

# =============================================================================
# THALAMUS - ATTENTION GATING (ALPHA OSCILLATIONS)
# =============================================================================

THALAMUS_ALPHA_SUPPRESSION = 0.5
"""Alpha oscillation suppression strength (50% suppression at trough)."""

THALAMUS_ALPHA_GATE_THRESHOLD = 0.0
"""Alpha phase threshold for suppression (0=trough, π=peak)."""

# =============================================================================
# THALAMUS - TRN (THALAMIC RETICULAR NUCLEUS)
# =============================================================================

THALAMUS_TRN_INHIBITION = 0.3
"""Strength of TRN → relay inhibition (feedback gating)."""

THALAMUS_TRN_RECURRENT = 0.4
"""TRN recurrent inhibition strength (generates spindle oscillations)."""

# =============================================================================
# THALAMUS - SPATIAL FILTERING
# =============================================================================

THALAMUS_SPATIAL_FILTER_WIDTH = 0.15
"""Gaussian filter width for center-surround receptive fields (15% of input dimension)."""

THALAMUS_CENTER_EXCITATION = 1.5
"""Center excitation strength in receptive field (1.5x amplification)."""

THALAMUS_SURROUND_INHIBITION = 0.5
"""Surround inhibition strength in receptive field (50% suppression)."""

THALAMUS_SURROUND_WIDTH_RATIO = 3.0
"""Surround width as multiple of center width (surround is 3x wider)."""

# =============================================================================
# THALAMUS - RELAY PARAMETERS
# =============================================================================

THALAMUS_RELAY_STRENGTH = 1.2
"""Base relay gain for thalamic amplification (20% amplification)."""

THALAMUS_NE_GAIN_SCALE = 0.5
"""Norepinephrine gain modulation scale (gain = 1.0 + 0.5 × NE)."""

# =============================================================================
# THALAMUS - WEIGHT INITIALIZATION SPARSITY
# =============================================================================

THALAMUS_RELAY_SPARSITY = 0.3
"""Sparsity for thalamus → cortex relay connections (30% active)."""

THALAMUS_RELAY_SCALE = 0.3
"""Weight scale for relay connections."""

THALAMUS_TRN_FEEDBACK_SPARSITY = 0.2
"""Sparsity for cortex → TRN feedback connections (20% for strong sparse control)."""

THALAMUS_TRN_FEEDBACK_SCALE = 0.4
"""Weight scale for TRN feedback connections (strong feedback)."""

THALAMUS_TRN_FEEDFORWARD_SPARSITY = 0.3
"""Sparsity for TRN → relay feedforward connections (30% for broad inhibition)."""

THALAMUS_SPATIAL_CENTER_SPARSITY = 0.2
"""Sparsity for center-surround spatial filters (20% for local receptive fields)."""


__all__ = [
    "THALAMUS_BURST_THRESHOLD",
    "THALAMUS_TONIC_THRESHOLD",
    "THALAMUS_BURST_SPIKE_COUNT",
    "THALAMUS_BURST_GAIN",
    "THALAMUS_MODE_THRESHOLD",
    "THALAMUS_ALPHA_SUPPRESSION",
    "THALAMUS_ALPHA_GATE_THRESHOLD",
    "THALAMUS_TRN_INHIBITION",
    "THALAMUS_TRN_RECURRENT",
    "THALAMUS_SPATIAL_FILTER_WIDTH",
    "THALAMUS_CENTER_EXCITATION",
    "THALAMUS_SURROUND_INHIBITION",
    "THALAMUS_SURROUND_WIDTH_RATIO",
    "THALAMUS_RELAY_STRENGTH",
    "THALAMUS_NE_GAIN_SCALE",
    "THALAMUS_RELAY_SPARSITY",
    "THALAMUS_RELAY_SCALE",
    "THALAMUS_TRN_FEEDBACK_SPARSITY",
    "THALAMUS_TRN_FEEDBACK_SCALE",
    "THALAMUS_TRN_FEEDFORWARD_SPARSITY",
    "THALAMUS_SPATIAL_CENTER_SPARSITY",
]
