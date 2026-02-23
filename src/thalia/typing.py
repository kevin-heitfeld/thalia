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
from enum import StrEnum
from typing import ClassVar, Dict, Literal, NewType, Optional, Protocol

import torch


# =============================================================================
# RECEPTOR TYPES
# =============================================================================


class ReceptorType(StrEnum):
    """Synaptic receptor type, encoding both neurotransmitter identity and polarity.

    Using an explicit receptor type rather than a bare ``is_inhibitory`` boolean:
    - Accurately represents biophysical receptor diversity (AMPA vs NMDA kinetics,
      GABA_A fast shunting vs GABA_B slow metabotropic)
    - Enforces Dale's Law at the type level: excitatory populations can only
      create ``AMPA`` or ``NMDA`` synapses; inhibitory populations only ``GABA_A``
      or ``GABA_B``
    - Allows downstream code to apply receptor-appropriate conductance kinetics

    In the current implementation AMPA and NMDA are summed (split via
    ``split_excitatory_conductance``) by each region's ``forward()``; the receptor
    type here records which biological pathway the conductance belongs to.
    """

    AMPA = "ampa"       # Fast excitatory (glutamatergic)
    NMDA = "nmda"       # Slow excitatory, voltage-gated (glutamatergic)
    GABA_A = "gaba_a"   # Fast inhibitory (GABAergic, ionotropic Cl⁻)
    GABA_B = "gaba_b"   # Slow inhibitory (GABAergic, metabotropic K⁺)

    @property
    def is_inhibitory(self) -> bool:
        """True for GABAergic (inhibitory) receptor types."""
        return self in (ReceptorType.GABA_A, ReceptorType.GABA_B)

    @property
    def is_excitatory(self) -> bool:
        """True for glutamatergic (excitatory) receptor types."""
        return self in (ReceptorType.AMPA, ReceptorType.NMDA)


# =============================================================================
# POPULATION POLARITY
# =============================================================================


class PopulationPolarity(StrEnum):
    """Intrinsic neurotransmitter polarity of a neuron population.

    Used to enforce Dale's Law at connection registration time: an EXCITATORY
    population must only form AMPA/NMDA synapses; an INHIBITORY population
    must only form GABA_A/GABA_B synapses.

    ``ANY`` disables enforcement (used for external inputs or populations whose
    polarity is not yet specified).
    """

    EXCITATORY = "excitatory"   # Glutamatergic (pyramidal, stellate, granule, etc.)
    INHIBITORY = "inhibitory"   # GABAergic (FSI, SST, VIP, OLM, MSN, etc.)
    ANY = "any"                 # No enforcement (external inputs, mixed)


# =============================================================================
# TENSOR TYPES
# =============================================================================


VoltageTensor = NewType("VoltageTensor", torch.Tensor)
"""Tensor of voltages [n_neurons] or [batch, n_neurons]."""


ConductanceTensor = NewType("ConductanceTensor", torch.Tensor)
"""Tensor of conductances [n_neurons] or [batch, n_neurons]."""


# =============================================================================
# GAP JUNCTION TYPES
# =============================================================================


GapJunctionConductance = NewType("GapJunctionConductance", torch.Tensor)
"""Gap junction conductance tensor [n_neurons].

Gap junctions are electrical synapses with bidirectional current flow.
Unlike chemical synapses with fixed reversals, gap junctions couple to
neighbor voltages dynamically.
"""


GapJunctionReversal = NewType("GapJunctionReversal", torch.Tensor)
"""Dynamic reversal potential for gap junctions [n_neurons].

For gap junctions, the "reversal" is the weighted average of neighbor
voltages, making it time-varying and neuron-specific.

Physics: I_gap[i] = g_gap × (E_eff[i] - V[i])
    where E_eff[i] = Σ_j [g_gap[i,j] × V[j]] / Σ_j g_gap[i,j]
"""


# =============================================================================
# BRAIN INPUT/OUTPUT TYPES
# =============================================================================


RegionName = str
"""Name of a brain region, e.g. 'cortex', 'hippocampus'."""


PopulationName = str
"""Name of a population within a region."""


PopulationSizes = Dict[PopulationName, int]
"""Mapping of population names to their population sizes."""


@dataclass(frozen=True)
class SynapseId:
    """Unique identifier for a synaptic connection.

    Encodes the full routing key (source region/population → target
    region/population) plus the receptor type at the post-synaptic terminal.

    **Receptor type replaces the old ``is_inhibitory: bool`` flag (P1-04).**
    Using :class:`ReceptorType` rather than a bare boolean:

    - Accurately captures biophysical receptor diversity
    - Enforces Dale's Law: check ``receptor_type.is_inhibitory`` against the
      source population's :class:`PopulationPolarity` at connection registration

    Factory methods
    ---------------
    :meth:`external_reward_to_vta_da` and :meth:`external_sensory_to_thalamus_relay`
    provide convenient construction for the two standard external-input patterns.
    """

    source_region: RegionName
    source_population: PopulationName
    target_region: RegionName
    target_population: PopulationName
    receptor_type: ReceptorType

    _EXTERNAL_REGION_NAME: ClassVar[RegionName] = "external"

    _SEP: ClassVar[str] = "|"

    def __post_init__(self) -> None:
        """Validate that none of the fields contain the separator character."""
        for field_name, value in (
            ("source_region", self.source_region),
            ("source_population", self.source_population),
            ("target_region", self.target_region),
            ("target_population", self.target_population),
        ):
            if self._SEP in value or '.' in value:
                raise ValueError(
                    f"{field_name} cannot contain '{self._SEP}' or '.' character: {value}"
                )

    # ------------------------------------------------------------------
    # Polarity convenience (backward-compatible property)
    # ------------------------------------------------------------------

    @property
    def is_inhibitory(self) -> bool:
        """True when ``receptor_type`` is ``GABA_A`` or ``GABA_B``."""
        return self.receptor_type.is_inhibitory

    @property
    def is_excitatory(self) -> bool:
        """True when ``receptor_type`` is ``AMPA`` or ``NMDA``."""
        return self.receptor_type.is_excitatory

    def __str__(self) -> str:
        return (
            f"{self.source_region}:{self.source_population} → "
            f"{self.target_region}:{self.target_population} "
            f"({self.receptor_type})"
        )

    def to_key(self) -> str:
        """Encode this SynapseId to a stable pipe-delimited ASCII string.

        Used as keys for nn.ParameterDict / nn.ModuleDict (which require str keys).
        The encoding is deterministic and fully reversible via :meth:`from_key`.

        Returns:
            Stable pipe-delimited string, e.g.
            ``"thalamus|relay|cortex|l4_pyr|ampa"``
        """
        return (
            f"{self.source_region}{self._SEP}"
            f"{self.source_population}{self._SEP}"
            f"{self.target_region}{self._SEP}"
            f"{self.target_population}{self._SEP}"
            f"{self.receptor_type}"
        )

    @classmethod
    def from_key(cls, key: str) -> "SynapseId":
        """Decode a pipe-delimited key back to a :class:`SynapseId`.

        Args:
            key: String previously returned by :meth:`to_key`.

        Returns:
            Reconstructed :class:`SynapseId` instance.

        Raises:
            ValueError: If *key* does not have the expected format or contains
                an unknown receptor type string.
        """
        parts = key.split(cls._SEP)
        if len(parts) != 5:
            raise ValueError(
                f"Invalid SynapseId key '{key}': expected 5 pipe-separated parts, "
                f"got {len(parts)}."
            )
        src_r, src_p, tgt_r, tgt_p, rtype_raw = parts

        try:
            rtype = ReceptorType(rtype_raw)
        except ValueError as exc:
            raise ValueError(
                f"Unknown receptor_type '{rtype_raw}' in SynapseId key '{key}'. "
                f"Valid values: {list(ReceptorType)}"
            ) from exc

        return cls(
            source_region=src_r,
            source_population=src_p,
            target_region=tgt_r,
            target_population=tgt_p,
            receptor_type=rtype,
        )

    def is_external_reward_input(self) -> bool:
        """True when this synapse carries external reward input (from 'external' region)."""
        from thalia.brain.regions.population_names import ExternalPopulation  # type: ignore[import]  # Avoid circular import
        return (
            self.source_region == self._EXTERNAL_REGION_NAME
            and self.source_population == ExternalPopulation.REWARD
        )

    def is_external_sensory_input(self) -> bool:
        """True when this synapse carries external sensory input (from 'external' region)."""
        from thalia.brain.regions.population_names import ExternalPopulation  # type: ignore[import]  # Avoid circular import
        return (
            self.source_region == self._EXTERNAL_REGION_NAME
            and self.source_population == ExternalPopulation.SENSORY
        )

    @classmethod
    def external_reward_to_vta_da(cls, vta_region: RegionName) -> "SynapseId":
        """Factory: external reward input → VTA DA mesolimbic population.

        Reward signals target the mesolimbic sub-population (primary RPE pathway).
        Mesocortical DA neurons receive contextual arousal signals, not raw reward.
        """
        from thalia.brain.regions.population_names import ExternalPopulation, VTAPopulation  # type: ignore[import]  # Avoid circular import
        return SynapseId(
            source_region=cls._EXTERNAL_REGION_NAME,
            source_population=ExternalPopulation.REWARD,
            target_region=vta_region,
            target_population=VTAPopulation.DA_MESOLIMBIC,
            receptor_type=ReceptorType.AMPA,
        )

    @classmethod
    def external_sensory_to_thalamus_relay(cls, thalamus_region: RegionName) -> "SynapseId":
        """Factory: external sensory input → thalamus relay population."""
        from thalia.brain.regions.population_names import ExternalPopulation, ThalamusPopulation  # type: ignore[import]  # Avoid circular import
        return SynapseId(
            source_region=cls._EXTERNAL_REGION_NAME,
            source_population=ExternalPopulation.SENSORY,
            target_region=thalamus_region,
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        )


NeuromodulatorType = Literal["da", "da_mesolimbic", "da_mesocortical", "da_nigrostriatal", "ne", "ach", "ach_septal", "5ht"]
"""Type of neuromodulator for volume transmission signaling.

Neuromodulators use diffuse broadcast rather than point-to-point synaptic connections:
- 'da': Legacy combined DA key (avoid; use pathway-specific keys)
- 'da_mesolimbic': Mesolimbic DA (VTA → ventral striatum, hippocampus, amygdala)
- 'da_mesocortical': Mesocortical DA (VTA → PFC, prefrontal areas)
- 'da_nigrostriatal': Nigrostriatal DA (SNc → dorsal striatum, motor learning)
- 'ne': Norepinephrine (from locus coeruleus)
- 'ach': Acetylcholine (from nucleus basalis — cortical attention)
- 'ach_septal': Acetylcholine (from medial septum — hippocampal theta)
- '5ht': Serotonin (from raphe nuclei) - future support
"""


class NeuromodulatorSource(Protocol):
    """Protocol marking a NeuralRegion as a neuromodulator volume-transmission source.

    Any region that produces neuromodulator signals (DA, NE, ACh, 5-HT) should
    declare this class variable.  The dict maps neuromodulator type strings
    (``"da"``, ``"ne"``, ``"ach"``) to the name of the source population within
    that region whose spike output represents the modulator signal.

    Example::

        class VTARegion(NeuralRegion[VTAConfig]):
            neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]] = {'da': 'da'}

    Runtime detection uses ``hasattr(region, 'neuromodulator_outputs')`` rather
    than ``isinstance`` because ``ClassVar`` members are invisible to Python's
    ``runtime_checkable`` Protocol machinery.
    """
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]]


NeuromodulatorInput = Dict[NeuromodulatorType, Optional[torch.Tensor]]
"""Mapping of neuromodulator types to their spike tensors for broadcast signaling.

Unlike synaptic connections (SynapticInput), neuromodulators are broadcast to all regions
and processed by receptors. Regions ignore neuromodulators they don't use.
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
