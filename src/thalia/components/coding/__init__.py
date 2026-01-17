"""
Spike coding and spike utilities.

This module provides spike encoding/decoding strategies and spike manipulation utilities.
"""

from __future__ import annotations

from thalia.components.coding.spike_coding import (
    CodingStrategy,
    RateDecoder,
    RateEncoder,
    SpikeCodingConfig,
    SpikeDecoder,
    SpikeEncoder,
    compute_spike_similarity,
)
from thalia.components.coding.spike_utils import (
    compute_firing_rate,
    compute_spike_count,
    compute_spike_density,
    is_saturated,
    is_silent,
)

__all__ = [
    # Spike Coding
    "CodingStrategy",
    "SpikeCodingConfig",
    "SpikeEncoder",
    "SpikeDecoder",
    "RateEncoder",
    "RateDecoder",
    "compute_spike_similarity",
    # Spike Utils
    "compute_firing_rate",
    "compute_spike_count",
    "compute_spike_density",
    "is_silent",
    "is_saturated",
]
