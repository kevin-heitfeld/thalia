# pyright: strict
"""
Type Aliases for Thalia

This module defines type aliases used throughout the Thalia codebase for
clearer type hints and better IDE support.

All type aliases are organized by category and should be imported from this
module rather than defining them inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch


RegionName = str
"""Name of a brain region, e.g. 'cortex', 'hippocampus'."""


PopulationName = str
"""Name of a population within a region, e.g. 'excitatory', 'inhibitory'."""


PopulationSizes = Dict[PopulationName, int]
"""Mapping of population names to their population sizes, e.g. {'excitatory': 100, 'inhibitory': 50}."""


@dataclass(frozen=True)
class SynapseId:
    """Unique identifier for a synaptic connection, defined by its source and target."""
    source_region: RegionName
    source_population: PopulationName
    target_region: RegionName
    target_population: PopulationName
    is_inhibitory: bool = False

    def __str__(self) -> str:
        return f"{self.source_region}:{self.source_population} → {self.target_region}:{self.target_population} ({'inhibitory' if self.is_inhibitory else 'excitatory'})"

    def is_external_sensory_input(self) -> bool:
        """Check if this synapse represents an external sensory input, which is defined as coming from the 'external' region and 'sensory' population."""
        return self.source_region == "external" and self.source_population == "sensory"

    @staticmethod
    def external_sensory_to_thalamus_relay() -> SynapseId:
        """Factory method to create a SynapseId for external sensory input to thalamus relay population."""
        from thalia.brain.regions.population_names import ThalamusPopulation  # type: ignore[import]  # Avoid circular import
        return SynapseId(
            source_region="external",
            source_population="sensory",
            target_region="thalamus",
            target_population=ThalamusPopulation.RELAY.value,
            is_inhibitory=False
        )


NeuromodulatorType = Literal["da", "ne", "ach", "5ht"]
"""Type of neuromodulator for volume transmission signaling.

Neuromodulators use diffuse broadcast rather than point-to-point synaptic connections:
- 'da': Dopamine (from VTA, substantia nigra)
- 'ne': Norepinephrine (from locus coeruleus)
- 'ach': Acetylcholine (from nucleus basalis, medial septum)
- '5ht': Serotonin (from raphe nuclei) - future support
"""


NeuromodulatorInput = Dict[NeuromodulatorType, Optional[torch.Tensor]]
"""Mapping of neuromodulator types to their spike tensors for broadcast signaling.

Unlike synaptic connections (SynapticInput), neuromodulators are broadcast to all regions
and processed by receptors. Regions ignore neuromodulators they don't use.

Example:
    {
        'da': torch.tensor([True, False, True, ...]),  # VTA dopamine neuron spikes
        'ne': torch.tensor([False, True, False, ...]), # LC norepinephrine spikes
        'ach': None,  # No acetylcholine signal this timestep
    }
"""


SynapticInput = Dict[SynapseId, torch.Tensor]
"""Mapping of SynapseId to its corresponding input spike tensor for point-to-point synaptic connections.

This structure allows for flexible routing of spikes to the correct dendritic compartments
based on their source and target, enabling complex connectivity patterns within neural regions.
"""


RegionOutput = Dict[PopulationName, torch.Tensor]
"""Mapping of population names to their output spike tensors, where each population's output is a binary spike tensor."""


BrainOutput = Dict[RegionName, RegionOutput]
"""Mapping of region names to their output spikes, where each region's output is a RegionOutput containing the spike tensors for each population."""
