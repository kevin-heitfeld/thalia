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
from typing import ClassVar, Dict, Literal, Optional, Protocol

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

    _SEP: ClassVar[str] = "|"

    def __str__(self) -> str:
        return f"{self.source_region}:{self.source_population} → {self.target_region}:{self.target_population} ({'inhibitory' if self.is_inhibitory else 'excitatory'})"

    def to_key(self) -> str:
        """Encode this SynapseId to a stable pipe-delimited ASCII string.

        Used as keys for nn.ParameterDict / nn.ModuleDict (which require str keys).
        The encoding is deterministic and fully reversible via :meth:`from_key`.

        Returns:
            Stable pipe-delimited string, e.g.
            ``"thalamus|relay|cortex|l4_pyr|0"``
        """
        inh = "1" if self.is_inhibitory else "0"
        return f"{self.source_region}{self._SEP}{self.source_population}{self._SEP}{self.target_region}{self._SEP}{self.target_population}{self._SEP}{inh}"

    @classmethod
    def from_key(cls, key: str) -> "SynapseId":
        """Decode a pipe-delimited key back to a :class:`SynapseId`.

        Args:
            key: String previously returned by :meth:`to_key`.

        Returns:
            Reconstructed :class:`SynapseId` instance.

        Raises:
            ValueError: If *key* does not have the expected format.
        """
        parts = key.split(cls._SEP)
        if len(parts) != 5:
            raise ValueError(
                f"Invalid SynapseId key '{key}': expected 5 pipe-separated parts, got {len(parts)}."
            )
        src_r, src_p, tgt_r, tgt_p, inh = parts
        return cls(
            source_region=src_r,
            source_population=src_p,
            target_region=tgt_r,
            target_population=tgt_p,
            is_inhibitory=(inh == "1"),
        )

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


class NeuromodulatorSource(Protocol):
    """Protocol marking a NeuralRegion as a neuromodulator volume-transmission source.

    Any region that produces neuromodulator signals (DA, NE, ACh, 5-HT) should
    declare this class variable.  The dict maps neuromodulator type strings
    (``"da"``, ``"ne"``, ``"ach"``) to the name of the source population within
    that region whose spike output represents the modulator signal.

    Example::

        class VTARegion(NeuralRegion[VTAConfig]):
            neuromodulator_outputs: ClassVar[Dict[str, str]] = {'da': 'da'}

    Runtime detection uses ``hasattr(region, 'neuromodulator_outputs')`` rather
    than ``isinstance`` because ``ClassVar`` members are invisible to Python's
    ``runtime_checkable`` Protocol machinery.
    """
    neuromodulator_outputs: ClassVar[Dict[str, str]]


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
