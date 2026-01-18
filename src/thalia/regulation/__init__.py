"""Regulation components for Thalia."""

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
