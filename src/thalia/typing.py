# pyright: strict
"""
Type Aliases for Thalia

This module defines type aliases used throughout the Thalia codebase for
clearer type hints and better IDE support.

All type aliases are organized by category and should be imported from this
module rather than defining them inline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


RegionName = str
"""Name of a brain region, e.g. 'cortex', 'hippocampus'."""

LayerName = str
"""Name of a layer within a region, e.g. 'relay', 'trn', 'l23', 'dg'."""

PortName = str
"""Name of an output port, e.g. 'relay', 'l5', 'executive'."""

RegionLayerSizes = Dict[LayerName, int]
"""Mapping of layer names to their sizes (number of neurons)."""

SpikesSourceKey = str
"""Compound key for spike sources in the format 'region:port', e.g. 'thalamus:trn'."""

RegionSpikesDict = Dict[SpikesSourceKey, torch.Tensor]
"""Mapping of spike source keys to their corresponding spike tensors."""

BrainSpikesDict = Dict[RegionName, RegionSpikesDict]
"""Mapping of region names to their output spikes, where each region's output is a RegionSpikesDict."""

def compound_key(region_name: RegionName, port_name: PortName) -> SpikesSourceKey:
    """Utility function to create a compound key for spike sources."""
    return f"{region_name}:{port_name}"

def parse_compound_key(compound_key: SpikesSourceKey) -> Tuple[RegionName, PortName]:
    """Utility function to parse a compound key into region and port names."""
    if ":" not in compound_key:
        raise ValueError(f"Invalid compound key '{compound_key}'. Expected format 'region:port'.")

    region_name, port_name = compound_key.split(":")
    return region_name, port_name
