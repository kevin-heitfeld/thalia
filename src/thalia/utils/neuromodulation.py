"""Neuromodulation utilities for Thalia."""

from __future__ import annotations


def compute_ne_gain(
    ne_level: float,
    ne_gain_min: float = 1.0,
    ne_gain_max: float = 1.5,
) -> float:
    """Compute norepinephrine gain modulation from NE level."""
    return ne_gain_min + (ne_gain_max - ne_gain_min) * ne_level
