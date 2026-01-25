"""
Synaptic Components.

This package contains synaptic layers for neural regions, separating
spike transmission (axons) from synaptic integration (weights + learning).

Architecture v2.0: Synapses belong to POST-synaptic regions (at dendrites),
not to pathways. This matches biological reality.
"""

from __future__ import annotations

from .spillover import SpilloverConfig, SpilloverTransmission, apply_spillover_to_weights

__all__ = [
    "SpilloverConfig",
    "SpilloverTransmission",
    "apply_spillover_to_weights",
]
