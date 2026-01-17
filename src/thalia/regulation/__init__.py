"""
Regulation Constants.

Constants and utilities for homeostasis, learning, exploration, and normalization.

NOTE: Constants have been fully consolidated to thalia.constants module.
Import constants directly from thalia.constants.<submodule> instead.
"""

from __future__ import annotations

from thalia.regulation.normalization import (
    ContrastNormalization,
    DivisiveNormalization,
    DivisiveNormConfig,
    SpatialDivisiveNorm,
)

__all__ = [
    # Normalization Classes
    "DivisiveNormConfig",
    "DivisiveNormalization",
    "ContrastNormalization",
    "SpatialDivisiveNorm",
]
