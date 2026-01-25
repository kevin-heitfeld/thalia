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

THALAMUS_MODE_THRESHOLD = 0.5
"""Threshold for burst/tonic mode detection (0=burst, 1=tonic)."""

# =============================================================================
# THALAMUS - SPATIAL FILTERING
# =============================================================================

THALAMUS_SURROUND_WIDTH_RATIO = 3.0
"""Surround width as multiple of center width (surround is 3x wider)."""

# =============================================================================
# THALAMUS - RELAY PARAMETERS
# =============================================================================

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
    "THALAMUS_MODE_THRESHOLD",
    "THALAMUS_SURROUND_WIDTH_RATIO",
    "THALAMUS_NE_GAIN_SCALE",
    "THALAMUS_RELAY_SPARSITY",
    "THALAMUS_RELAY_SCALE",
    "THALAMUS_TRN_FEEDBACK_SPARSITY",
    "THALAMUS_TRN_FEEDBACK_SCALE",
    "THALAMUS_TRN_FEEDFORWARD_SPARSITY",
    "THALAMUS_SPATIAL_CENTER_SPARSITY",
]
