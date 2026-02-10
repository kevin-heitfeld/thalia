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

PopulationName = str
"""Name of a population within a region, e.g. 'excitatory', 'inhibitory'."""

PopulationSizes = Dict[PopulationName, int]
"""Mapping of population names to their population sizes, e.g. {'excitatory': 100, 'inhibitory': 50}."""

SpikesSourceKey = str
"""Compound key for spike sources in the format 'region:population', e.g. 'thalamus:inhibitory'."""

RegionSpikesDict = Dict[SpikesSourceKey, torch.Tensor]
"""Mapping of spike source keys to their corresponding spike tensors."""

BrainSpikesDict = Dict[RegionName, RegionSpikesDict]
"""Mapping of region names to their output spikes, where each region's output is a RegionSpikesDict."""

def compound_key(region_name: RegionName, population_name: PopulationName) -> SpikesSourceKey:
    """Utility function to create a compound key from region and population names."""
    return f"{region_name}:{population_name}"

def parse_compound_key(compound_key: SpikesSourceKey) -> Tuple[RegionName, PopulationName]:
    """Utility function to parse a compound key into region and population names."""
    if ":" not in compound_key:
        raise ValueError(f"Invalid compound key '{compound_key}'. Expected format 'region:population'.")

    region_name, population_name = compound_key.split(":")
    return region_name, population_name
