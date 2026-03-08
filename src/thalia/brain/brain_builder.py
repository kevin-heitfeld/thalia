"""
Brain Builder - Fluent API for Brain Construction

This module provides a fluent, progressive API for building brain architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from thalia import GlobalConfig
from thalia.brain.regions.population_names import StriatumPopulation
from thalia.brain.synapses import STPConfig, ConductanceScaledSpec
from thalia.typing import (
    PopulationName,
    PopulationSizes,
    ReceptorType,
    RegionName,
    SynapseId,
)

from .axonal_tract import AxonalTract, AxonalTractSourceSpec
from .brain import Brain
from .configs import BrainConfig, NeuralRegionConfig
from .regions import NeuralRegionRegistry, NeuralRegion

if TYPE_CHECKING:
    from thalia.learning.strategies import LearningStrategy


@dataclass
class RegionSpec:
    """Specification for a brain region.

    Attributes:
        name: Instance name
        registry_name: Region type in registry
        population_sizes: Population size specifications
        config: Region configuration parameters
        instance: Instantiated region (set after build())
    """

    name: RegionName
    registry_name: RegionName
    population_sizes: PopulationSizes
    config: Optional[NeuralRegionConfig] = None
    instance: Optional[NeuralRegion] = None


@dataclass
class ConnectionSpec:
    """Specification for a connection between two regions.

    Attributes:
        synapse_id: Unique identifier for the synapse (source/target/population combination)
        axonal_delay_ms: Axonal conduction delay in milliseconds (mean)
        axonal_delay_std_ms: Standard deviation for heterogeneous delays (0 = uniform)
        connectivity: Connection probability (fraction of connections present, 0-1)
        weight_scale: Initial weight scale (normalized conductance)
        instance: Instantiated axonal tract (set after build())
    """

    synapse_id: SynapseId
    axonal_delay_ms: float
    axonal_delay_std_ms: float
    connectivity: float
    weight_scale: Union[float, ConductanceScaledSpec]
    stp_config: Optional[STPConfig]
    learning_strategy: Optional[LearningStrategy] = None
    instance: Optional[AxonalTract] = None


@dataclass
class ExternalInputSpec:
    """Specification for an external input source.

    Attributes:
        synapse_id: Unique identifier for the synapse (source/target/population combination)
        n_input: Number of input neurons from external source
        connectivity: Connection probability (0-1)
        weight_scale: Initial weight scale
    """

    synapse_id: SynapseId
    n_input: int
    connectivity: float
    weight_scale: Union[float, ConductanceScaledSpec]
    stp_config: Optional[STPConfig]
    learning_strategy: Optional[LearningStrategy] = None


@dataclass
class SourceContribution:
    """Per-source contribution to a target population's conductance budget."""

    source_key: str
    """Human-readable 'region:population' label."""

    source_rate_hz: float
    """Expected presynaptic firing rate (from ConductanceScaledSpec)."""

    fraction_of_drive: float
    """Fraction of total reference g_ss this source was specified to deliver."""

    u_eff: float
    """Steady-state STP utilization factor at source_rate_hz (1.0 if no STP)."""

    stp_correction_applied: bool
    """True when the weight was inflated to compensate for STP depletion."""

    g_intended: float
    """Conductance this source was designed to deliver (g_ss_total * fraction)."""

    g_effective: float
    """Conductance actually delivered after STP depletion (= g_intended * u_eff / stp_factor)."""

    is_inhibitory: bool
    """True for GABA_A/GABA_B receptor types (negative reversal potential)."""

    E_reversal: float
    """Synaptic reversal potential (spec.target_E_E: ≈3.0 for AMPA/NMDA, ≈-0.5 for GABA_A/B)."""


@dataclass
class ConductanceBudgetEntry:
    """Analytical conductance budget for one excitatory target population.

    Produced by :meth:`BrainBuilder.validate_conductance_budget`.
    """

    target_region: str
    target_population: str
    sources: List[SourceContribution]

    g_intended_total: float
    """Sum of g_ss values as designed (ignoring STP depletion at runtime)."""

    g_effective_total: float
    """Sum of g_ss values after STP depletion (what neurons actually receive)."""

    v_inf_intended: float
    """V_inf if all sources deliver exactly their designed conductance."""

    v_inf_effective: float
    """V_inf after STP depletion — the quantity that determines firing rate."""

    stp_penalty: float
    """v_inf_intended − v_inf_effective (positive = STP is suppressing activity)."""

    target_g_L: float
    target_E_E: float
    target_E_L: float

    g_exc_intended: float
    """Sum of designed conductance for AMPA/NMDA (excitatory) sources only."""

    g_exc_effective: float
    """Excitatory conductance after STP depletion."""

    g_inh_intended: float
    """Sum of designed conductance for GABA_A/GABA_B (inhibitory) sources only."""

    g_inh_effective: float
    """Inhibitory conductance after STP depletion."""

    ei_ratio: float
    """g_exc_effective / g_inh_effective.  ``inf`` when no inhibitory input is tracked."""

    issues: List[str]
    """Non-empty list of warnings; empty means no issues detected."""


def _apply_stp_correction(
    weight_scale: Union[float, ConductanceScaledSpec],
    stp_config: Optional[STPConfig],
) -> Union[float, ConductanceScaledSpec]:
    """Return a copy of *weight_scale* with ``stp_utilization_factor`` set to the
    analytically computed steady-state utilization, unless the caller already
    provided a non-default value (``!= 1.0``).

    For plain-float weight_scale, or when stp_config is None, returns unchanged.
    """
    if not isinstance(weight_scale, ConductanceScaledSpec):
        return weight_scale
    if stp_config is None:
        return weight_scale
    if weight_scale.stp_utilization_factor != 1.0:
        # Caller set it explicitly — trust them
        return weight_scale
    u_eff = stp_config.steady_state_utilization(weight_scale.source_rate_hz)
    return replace(weight_scale, stp_utilization_factor=u_eff)


class BrainBuilder:
    """Fluent API for progressive brain construction.

    Supports:
        - Incremental region addition via method chaining
        - Connection definition with automatic axonal tract creation
        - Preset architectures for common use cases
        - Validation before building
        - Save/load graph specifications to JSON
    """

    # Registry of preset architectures
    _presets: Dict[str, PresetArchitecture] = {}

    def __init__(self, brain_config: Optional[BrainConfig] = None):
        """Initialize builder with brain configuration.

        Args:
            brain_config: Brain configuration (device, dt_ms, etc.)
        """
        self.brain_config = brain_config or BrainConfig()
        self._region_specs: Dict[RegionName, RegionSpec] = {}
        self._connection_specs: List[ConnectionSpec] = []
        self._external_input_specs: List[ExternalInputSpec] = []

    def add_region(
        self,
        name: RegionName,
        registry_name: RegionName,
        population_sizes: PopulationSizes,
        config: Optional[NeuralRegionConfig] = None,
    ) -> BrainBuilder:
        """Add a region to the brain.

        Args:
            name: Instance name
            registry_name: Region type in registry
            population_sizes: Population size specifications
            config: Optional region configuration parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If region name already exists
            KeyError: If registry_name not found in NeuralRegionRegistry
        """
        # Validate name uniqueness
        if name in self._region_specs:
            raise ValueError(f"Region '{name}' already exists")

        # Validate registry name exists
        if not NeuralRegionRegistry.is_registered(registry_name):
            available = NeuralRegionRegistry.list_regions()
            raise KeyError(f"Registry name '{registry_name}' not found. Available: {available}")

        spec = RegionSpec(
            name=name,
            registry_name=registry_name,
            population_sizes=population_sizes,
            config=config,
        )

        self._region_specs[name] = spec
        return self

    def connect(
        self,
        synapse_id: SynapseId,
        axonal_delay_ms: float,
        axonal_delay_std_ms: float,
        connectivity: float,
        weight_scale: Union[float, ConductanceScaledSpec],
        stp_config: Optional[STPConfig] = None,
    ) -> BrainBuilder:
        """Connect two regions with an axonal tract.

        Args:
            synapse_id: Unique identifier for the synapse (source/target/population combination)
            axonal_delay_ms: Axonal conduction delay in milliseconds (mean)
            axonal_delay_std_ms: Standard deviation for heterogeneous delays (0 = uniform delay)
            connectivity: Connection probability (fraction of connections present, 0-1)
            weight_scale: Initial weight scale (normalized conductance)
            stp_config: Optional short-term plasticity configuration for this connection (if None, no STP applied)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If source or target region doesn't exist
        """
        if synapse_id.source_region not in self._region_specs:
            raise ValueError(f"Source region '{synapse_id.source_region}' not found")
        if synapse_id.target_region not in self._region_specs:
            raise ValueError(f"Target region '{synapse_id.target_region}' not found")
        if isinstance(weight_scale, float) and weight_scale < 0:
            raise ValueError("Weight scale must be non-negative")

        spec = ConnectionSpec(
            synapse_id=synapse_id,
            axonal_delay_ms=axonal_delay_ms,
            axonal_delay_std_ms=axonal_delay_std_ms,
            connectivity=connectivity,
            weight_scale=weight_scale,
            stp_config=stp_config,
        )

        self._connection_specs.append(spec)
        return self

    def add_external_input_source(
        self,
        synapse_id: SynapseId,
        n_input: int,
        connectivity: float,
        weight_scale: Union[float, ConductanceScaledSpec],
        stp_config: Optional[STPConfig] = None,
    ) -> BrainBuilder:
        """Add an external input source to a region.

        Args:
            synapse_id: Unique identifier for the synapse (source/target/population combination)
            n_input: Number of input neurons from external source
            connectivity: Connection probability (0-1)
            weight_scale: Initial weight scale
            stp_config: Optional short-term plasticity configuration for this connection (if None, no STP applied)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If target region doesn't exist
        """
        if synapse_id.target_region not in self._region_specs:
            raise ValueError(f"Target region '{synapse_id.target_region}' not found")
        if isinstance(weight_scale, float) and weight_scale < 0:
            raise ValueError("Weight scale must be non-negative")

        spec = ExternalInputSpec(
            synapse_id=synapse_id,
            n_input=n_input,
            connectivity=connectivity,
            weight_scale=weight_scale,
            stp_config=stp_config,
        )

        self._external_input_specs.append(spec)
        return self

    def connect_to_striatum(
        self,
        source_region: RegionName,
        source_population: PopulationName,
        axonal_delay_ms: float,
        axonal_delay_std_ms: float,
        connectivity: float,
        weight_scale: Union[float, ConductanceScaledSpec],
        *,
        target_region: RegionName = "striatum",
        d2_weight_scale: Optional[Union[float, ConductanceScaledSpec]] = None,
        fsi_connectivity: float = 0.5,
        fsi_weight_scale: Optional[Union[float, ConductanceScaledSpec]] = None,
        tan_connectivity: float = 0.3,
        stp_config: Optional[STPConfig] = None,
    ) -> "BrainBuilder":
        """Connect a source region to the striatum with explicit D1, D2, FSI, and TAN pathways.

        Creates four ConnectionSpecs (one per sub-population) at the same axonal
        delay but with biologically-motivated connectivity and weight differences:

        +----------+--------------------+------------------------------+--------------------+
        | Pop      | connectivity       | weight_scale                 | STP                |
        +==========+====================+==============================+====================+
        | D1       | *connectivity*     | *weight_scale*               | *stp_config*       |
        | D2       | *connectivity*     | *d2_weight_scale*            | *stp_config*       |
        | FSI      | *fsi_connectivity* | *fsi_weight_scale*           | None               |
        | TAN      | *tan_connectivity* | weight_scale×0.5             | None               |
        +----------+--------------------+------------------------------+--------------------+

        FSI weight default is ``weight_scale * 20``.  FSI cells are fast-spiking
        interneurons with a higher spike threshold and shorter membrane time constant
        than MSNs, so they require substantially stronger drive to reach firing
        threshold from cortical/thalamic afferents.  Without this boost, FSI input
        is sub-threshold and D1/D2 MSNs receive no lateral inhibition, causing the
        E/I ratio to blow up (observed: 54.9 during diagnostics).

        Use ``stp_config`` to pass a source-appropriate short-term plasticity preset::

            builder.connect_to_striatum(
                "cortex_sensory", CortexPopulation.L5_PYR,
                axonal_delay_ms=4.0, axonal_delay_std_ms=6.0,
                connectivity=0.3, weight_scale=0.0001,
                stp_config=CORTICOSTRIATAL_PRESET.configure(),
            )

        Args:
            source_region: Name of the source region.
            source_population: Source population identifier.
            axonal_delay_ms: Mean axonal delay in milliseconds (applied to all four connections).
            axonal_delay_std_ms: Std-dev for heterogeneous delays.
            connectivity: Connection probability for D1 and D2 MSNs.
            weight_scale: Initial weight scale for D1 MSNs (and TAN/FSI baseline).
            target_region: Name of the target striatal region (default: ``"striatum"``).
            d2_weight_scale: Weight scale for D2 MSNs; defaults to *weight_scale* if None.
            fsi_connectivity: Connection probability for FSI input (default 0.5).
            fsi_weight_scale: Weight scale for FSI; defaults to ``weight_scale * 20``
                              if None (FSI require stronger drive than MSNs).
            tan_connectivity: Connection probability for TAN input (default 0.3).
            stp_config: Short-term plasticity config applied to D1 and D2 only;
                        FSI and TAN always receive ``None``.

        Returns:
            Self for method chaining.
        """
        d2_ws = d2_weight_scale if d2_weight_scale is not None else weight_scale
        # FSI weight derivation:
        # If fsi_weight_scale is provided, use it directly.
        # If weight_scale is a ConductanceScaledSpec, auto-derive an FSI-specific spec
        # using FSI biophysics (g_L=0.10, tau_E=5ms): FSI are fast-spiking and should
        # fire reliably on afferent bursts (target_v_inf=1.10, slightly above threshold).
        # stp_utilization_factor=1.0 (default): auto-correction in _create_axonal_tract
        # will compute the correct u_eff from the stp_config for FSI.
        # If weight_scale is a plain float, fall back to the 10× multiplier heuristic.
        if fsi_weight_scale is not None:
            fsi_ws: Union[float, ConductanceScaledSpec] = fsi_weight_scale
        elif isinstance(weight_scale, ConductanceScaledSpec):
            fsi_ws = ConductanceScaledSpec(
                source_rate_hz=weight_scale.source_rate_hz,
                target_g_L=0.10,          # FSI leak conductance (fast-spiking, tau_m≈8ms)
                target_tau_E_ms=5.0,      # Standard AMPA
                target_v_inf=1.10,        # FSI fires above threshold for FFI role
                fraction_of_drive=weight_scale.fraction_of_drive,
                # stp_utilization_factor defaults to 1.0 — auto-correction handles STP
            )
        else:
            fsi_ws = weight_scale * 10.0  # type: ignore[operator]

        # TAN weight derivation:
        # TANs are large cholinergic neurons (g_L=0.04, tau_E=10ms) with intrinsic
        # pacemaking; afferent drive should bring them sub-threshold (target_v_inf=0.95)
        # so pacemaking + input together reach threshold.  Half the D1/D2 fraction.
        # stp_utilization_factor=1.0 (hardcoded): TAN receives stp_config=None; if we
        # inherited the parent's manual STP factor the weight would be inflated with no
        # STP depletion to compensate, massively overdriving TAN.
        if isinstance(weight_scale, ConductanceScaledSpec):
            tan_ws: Union[float, ConductanceScaledSpec] = ConductanceScaledSpec(
                source_rate_hz=weight_scale.source_rate_hz,
                target_g_L=0.04,          # TAN leak conductance (slow, tau_m≈25ms)
                target_tau_E_ms=10.0,     # Slower cholinergic AMPA
                target_v_inf=0.95,        # Sub-threshold; intrinsic pacemaking closes gap
                fraction_of_drive=weight_scale.fraction_of_drive * 0.5,
                # stp_utilization_factor defaults to 1.0 — TAN has no STP; never inherit
            )
        else:
            tan_ws = weight_scale * 1.5  # type: ignore[operator]

        for target_pop, conn, ws, used_stp_config in [
            (StriatumPopulation.D1,  connectivity,     weight_scale, stp_config),
            (StriatumPopulation.D2,  connectivity,     d2_ws,        stp_config),
            (StriatumPopulation.FSI, fsi_connectivity, fsi_ws,       stp_config),
            (StriatumPopulation.TAN, tan_connectivity, tan_ws,       None),
        ]:
            self.connect(
                synapse_id=SynapseId(
                    source_region=source_region,
                    source_population=source_population,
                    target_region=target_region,
                    target_population=target_pop,
                    receptor_type=ReceptorType.AMPA,
                ),
                axonal_delay_ms=axonal_delay_ms,
                axonal_delay_std_ms=axonal_delay_std_ms,
                connectivity=conn,
                weight_scale=ws,
                stp_config=used_stp_config,
            )
        return self

    def validate_conductance_budget(self) -> List["ConductanceBudgetEntry"]:
        """Analytically compute synaptic V_inf for each target population with ConductanceScaledSpec inputs.

        Works entirely from registered :class:`ConnectionSpec` and
        :class:`ExternalInputSpec` data — no simulation required.  Call before
        or after :meth:`build` to catch silent (V_inf < 0.85) or overdriven
        (V_inf > 1.35) populations before running a 500 ms diagnostic.

        Only connections using :class:`~thalia.brain.synapses.ConductanceScaledSpec`
        are analysed; plain-float ``weight_scale`` connections are skipped.
        Both excitatory (AMPA, NMDA) and inhibitory (GABA_A, GABA_B) receptor
        types are included.  For inhibitory connections the caller must encode
        the GABA reversal by setting ``target_E_E = -0.5`` (or whichever value)
        in the :class:`~thalia.brain.synapses.ConductanceScaledSpec`.

        This is purely *synaptic* V_inf — ``baseline_drive`` contributions in
        region configs are not visible here, so populations that rely heavily
        on ``baseline_drive`` will appear lower than their true operating point.

        Returns:
            One :class:`ConductanceBudgetEntry` per unique (target_region,
            target_population) pair that has at least one qualifying connection.
            Sorted by predicted V_inf ascending (worst first).
        """
        INHIBITORY_RECEPTORS = {ReceptorType.GABA_A, ReceptorType.GABA_B}

        # Collect all ConductanceScaledSpec connections (both excitatory and inhibitory)
        all_specs: List[Tuple[SynapseId, float, ConductanceScaledSpec, Optional[STPConfig]]] = []
        for cs in self._connection_specs:
            if isinstance(cs.weight_scale, ConductanceScaledSpec):
                all_specs.append((cs.synapse_id, cs.connectivity, cs.weight_scale, cs.stp_config))
        for es in self._external_input_specs:
            if isinstance(es.weight_scale, ConductanceScaledSpec):
                all_specs.append((es.synapse_id, es.connectivity, es.weight_scale, es.stp_config))

        # Group by (target_region, target_population)
        groups: Dict[Tuple[str, str], List[Tuple[SynapseId, float, ConductanceScaledSpec, Optional[STPConfig]]]] = {}
        for sid, conn, spec, stp in all_specs:
            key = (sid.target_region, str(sid.target_population))
            groups.setdefault(key, []).append((sid, conn, spec, stp))

        entries: List[ConductanceBudgetEntry] = []
        for (target_region, target_pop), group in groups.items():
            # Use target neuron params from first spec — g_L and E_L are neuron properties
            # consistent across all connections targeting the same population.
            ref_spec = group[0][2]
            g_L = ref_spec.target_g_L
            E_L = ref_spec.target_E_L

            # E_E reference: pull from first excitatory spec (for display + issue thresholds).
            exc_pairs = [(s, sp) for (s, _, sp, __) in group if s.receptor_type not in INHIBITORY_RECEPTORS]
            E_E_ref = exc_pairs[0][1].target_E_E if exc_pairs else ref_spec.target_E_E

            source_summaries: List[SourceContribution] = []
            g_exc_intended  = 0.0
            g_exc_effective = 0.0
            g_inh_intended  = 0.0
            g_inh_effective = 0.0

            for sid, conn, spec, stp_cfg in group:
                is_inhibitory = sid.receptor_type in INHIBITORY_RECEPTORS
                # E_reversal = spec.target_E_E: callers set ~3.0 for AMPA/NMDA, ~-0.5 for GABA.
                E_reversal = spec.target_E_E

                # g this source is designed to deliver at its reference operating point.
                # Formula: g = g_L * (V_inf - E_L) / (E_reversal - V_inf)
                # Works for both exc and inh — both numerator and denominator are same sign
                # when V_inf lies between E_L and E_reversal (the normal operating regime).
                denom = E_reversal - spec.target_v_inf
                g_ss_ref = (
                    spec.target_g_L * (spec.target_v_inf - spec.target_E_L) / denom
                    if abs(denom) > 1e-9 else 0.0
                )
                g_intended = g_ss_ref * spec.fraction_of_drive

                # Actual u_eff at source rate (with STP depletion accounted for)
                u_eff = stp_cfg.steady_state_utilization(spec.source_rate_hz) if stp_cfg is not None else 1.0

                # Mirror the _apply_stp_correction() logic used in _create_axonal_tract():
                # if stp_utilization_factor is the default (1.0) AND STP is present,
                # auto-correction will scale the weight up by 1/u_eff at build time,
                # so the runtime conductance equals g_intended.
                # Only if the caller explicitly set stp_utilization_factor != 1.0 (manual override)
                # does depletion genuinely reduce delivered conductance.
                auto_corrected = stp_cfg is not None and spec.stp_utilization_factor == 1.0
                if auto_corrected:
                    stp_factor = u_eff  # weight will be inflated; runtime g = g_intended
                else:
                    stp_factor = spec.stp_utilization_factor  # manual value — trust it
                g_effective = g_intended * (u_eff / stp_factor) if stp_factor > 0 else 0.0

                source_summaries.append(SourceContribution(
                    source_key=f"{sid.source_region}:{sid.source_population}",
                    source_rate_hz=spec.source_rate_hz,
                    fraction_of_drive=spec.fraction_of_drive,
                    u_eff=u_eff,
                    stp_correction_applied=auto_corrected,
                    g_intended=g_intended,
                    g_effective=g_effective,
                    is_inhibitory=is_inhibitory,
                    E_reversal=E_reversal,
                ))
                if is_inhibitory:
                    g_inh_intended  += g_intended
                    g_inh_effective += g_effective
                else:
                    g_exc_intended  += g_intended
                    g_exc_effective += g_effective

            g_intended_total  = g_exc_intended  + g_inh_intended
            g_effective_total = g_exc_effective + g_inh_effective

            # Generalized V_inf: (Σ g_i·E_rev_i + g_L·E_L) / (g_L + Σ g_i)
            _num_int = g_L * E_L + sum(s.g_intended * s.E_reversal for s in source_summaries)
            _den_int = g_L + g_intended_total
            v_inf_intended = _num_int / _den_int if _den_int > 0 else E_L

            _num_eff = g_L * E_L + sum(s.g_effective * s.E_reversal for s in source_summaries)
            _den_eff = g_L + g_effective_total
            v_inf_effective = _num_eff / _den_eff if _den_eff > 0 else E_L

            ei_ratio = (
                g_exc_effective / g_inh_effective
                if g_inh_effective > 0.0 else float("inf")
            )

            # Check for manual stp_utilization_factor that significantly under/over-corrects
            manual_stp_issues: List[str] = []
            for s in source_summaries:
                if not s.stp_correction_applied and s.u_eff < 1.0:
                    for _sid, _conn, _spec, _stp in group:
                        if f"{_sid.source_region}:{_sid.source_population}" == s.source_key:
                            manual_factor = _spec.stp_utilization_factor
                            if abs(manual_factor - s.u_eff) > 0.10 * s.u_eff:
                                manual_stp_issues.append(
                                    f"{s.source_key}: manual stp_utilization_factor={manual_factor:.3f} "
                                    f"but actual u_eff={s.u_eff:.3f} at {s.source_rate_hz:.0f}Hz"
                                )

            issues: List[str] = []
            stp_penalty = v_inf_intended - v_inf_effective
            if manual_stp_issues:
                for msg in manual_stp_issues:
                    issues.append(f"Manual STP factor mismatch — {msg}")
            if v_inf_effective < 0.85:
                issues.append(
                    f"V_inf={v_inf_effective:.3f} is likely sub-threshold "
                    "(synaptic only; add baseline_drive contribution mentally)"
                )
            if v_inf_effective > 1.35:
                issues.append(f"V_inf={v_inf_effective:.3f} may be overdriven (> 1.35)")
            if g_inh_effective > 0.0 and ei_ratio > 10.0:
                issues.append(
                    f"E/I={ei_ratio:.1f} — excitation strongly dominates "
                    "(check that inhibitory ConductanceScaledSpec connections are registered)"
                )
            if g_inh_effective > 0.0 and ei_ratio < 0.5:
                issues.append(f"E/I={ei_ratio:.2f} — inhibition may suppress firing")

            entries.append(ConductanceBudgetEntry(
                target_region=target_region,
                target_population=target_pop,
                sources=source_summaries,
                g_intended_total=g_intended_total,
                g_effective_total=g_effective_total,
                v_inf_intended=v_inf_intended,
                v_inf_effective=v_inf_effective,
                stp_penalty=stp_penalty,
                target_g_L=g_L,
                target_E_E=E_E_ref,
                target_E_L=E_L,
                g_exc_intended=g_exc_intended,
                g_exc_effective=g_exc_effective,
                g_inh_intended=g_inh_intended,
                g_inh_effective=g_inh_effective,
                ei_ratio=ei_ratio,
                issues=issues,
            ))

        entries.sort(key=lambda e: e.v_inf_effective)
        return entries

    def print_conductance_budget(
        self,
        *,
        show_sources: bool = False,
        only_issues: bool = False,
    ) -> None:
        """Print a formatted conductance budget report to stdout.

        Args:
            show_sources: If True, list individual source contributions per population.
            only_issues:  If True, only show populations with warnings.
        """
        entries = self.validate_conductance_budget()
        if not entries:
            print("No ConductanceScaledSpec connections found.")
            return

        print(f"\n{'═' * 100}")
        print(f"  CONDUCTANCE BUDGET  ({len(entries)} populations)")
        print(f"{'═' * 100}")
        print(
            f"  {'Population':<47} {'V_inf_eff':>10} {'V_inf_int':>10} {'E/I':>7}  Issues"
        )
        print(f"  {'-' * 47} {'-' * 10} {'-' * 10} {'-' * 7}  ------")

        n_issues = 0
        for e in entries:
            if only_issues and not e.issues:
                continue
            status = "⚠" if e.issues else " "
            label = f"{e.target_region}:{e.target_population}"
            ei_str = "∞" if e.ei_ratio == float("inf") else f"{e.ei_ratio:.1f}"
            print(
                f"  {status} {label:<45} "
                f"{e.v_inf_effective:>10.3f} "
                f"{e.v_inf_intended:>10.3f} "
                f"{ei_str:>7}  "
                + (e.issues[0] if e.issues else "")
            )
            if show_sources:
                for s in e.sources:
                    corr = "✓" if s.stp_correction_applied else "✗"
                    kind = "I" if s.is_inhibitory else "E"
                    print(
                        f"      {corr}[{kind}] {s.source_key:<37} "
                        f"rate={s.source_rate_hz:5.1f}Hz "
                        f"frac={s.fraction_of_drive:.2f} "
                        f"u_eff={s.u_eff:.3f} "
                        f"g_eff={s.g_effective:.5f}"
                    )
            if e.issues:
                n_issues += 1
                for issue in e.issues[1:]:
                    print(f"      → {issue}")

        print(f"{'═' * 100}")
        print(f"  {n_issues} populations with issues out of {len(entries)} analysed.")
        print(f"  Note: V_inf_eff excludes baseline_drive (region config).  E/I = ∞ means no inhibitory ConductanceScaledSpec found.")
        print(f"{'═' * 100}\n")

    def validate(self) -> List[str]:
        """Validate graph before building.

        Checks:
            - No isolated regions (warning)

        Returns:
            List of warning/error messages (empty if valid)
        """
        issues: List[str] = []

        # Check for isolated regions (no connections)
        connected_regions: set[str] = set()
        for conn in self._connection_specs:
            connected_regions.add(conn.synapse_id.source_region)
            connected_regions.add(conn.synapse_id.target_region)

        isolated: set[str] = set(self._region_specs.keys()) - connected_regions
        if isolated:
            issues.append(f"Warning: Isolated regions (no connections): {isolated}")

        return issues

    def _create_axonal_tract(
        self,
        conn_spec: ConnectionSpec,
        regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]],
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> AxonalTract:
        """Create a single-source :class:`AxonalTract` from one connection spec.

        Registers the corresponding input source on the target region so it
        can allocate synaptic weights and STP modules.

        Args:
            conn_spec: A single :class:`ConnectionSpec` (one source → one target).
            regions: Dict of instantiated regions.

        Returns:
            :class:`AxonalTract` instance.
        """
        synapse_id = conn_spec.synapse_id
        source_region = regions[synapse_id.source_region]
        target_region = regions[synapse_id.target_region]

        source_size = source_region.get_population_size(synapse_id.source_population)

        spec = AxonalTractSourceSpec(
            synapse_id=synapse_id,
            size=source_size,
            delay_ms=conn_spec.axonal_delay_ms,
            delay_std_ms=conn_spec.axonal_delay_std_ms,
        )

        weight_scale = _apply_stp_correction(conn_spec.weight_scale, conn_spec.stp_config)

        target_region.add_input_source(
            synapse_id=synapse_id,
            n_input=source_size,
            connectivity=conn_spec.connectivity,
            weight_scale=weight_scale,
            stp_config=conn_spec.stp_config,
            learning_strategy=conn_spec.learning_strategy,
            device=device,
        )

        # Initialize STP state to steady-state to prevent onset transient.
        # Weights are scaled for steady-state STP depletion; if STP starts
        # at rest (u=U, x=1) the first ~100ms deliver 5-20× the designed
        # conductance before depletion settles.  Pre-loading u_ss/x_ss makes
        # t=0 identical to any other timestep at the expected firing rate.
        if conn_spec.stp_config is not None and isinstance(weight_scale, ConductanceScaledSpec):
            if synapse_id in target_region.stp_modules:
                target_region.stp_modules[synapse_id].initialize_to_steady_state(
                    weight_scale.source_rate_hz
                )

        return AxonalTract(
            spec=spec,
            dt_ms=self.brain_config.dt_ms,
            device=device,
        )

    def build(self, device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE) -> Brain:
        """Build Brain from specifications.

        Steps:
            1. Validate graph
            2. Instantiate all regions from registry
            3. Instantiate all axonal tracts from connection specs
            4. Create Brain instance with regions and axonal tracts

        Returns:
            Constructed Brain instance

        Raises:
            ValueError: If validation fails
        """
        # Validate before building
        issues = self.validate()
        errors = [msg for msg in issues if msg.startswith("Error:")]
        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))

        # Instantiate regions
        regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]] = {}
        for name, spec in self._region_specs.items():
            config_class = NeuralRegionRegistry.get_config_class(spec.registry_name)

            if config_class is None:
                raise ValueError(
                    f"Region '{spec.registry_name}' has no config_class registered. "
                    f"Update registry with config_class metadata."
                )

            config = spec.config if spec.config is not None else config_class()

            config.seed = self.brain_config.seed
            config.dt_ms = self.brain_config.dt_ms

            region = NeuralRegionRegistry.create(
                spec.registry_name,
                config=config,
                population_sizes=spec.population_sizes,
                region_name=name,
                device=device,
            )

            regions[name] = region
            spec.instance = region

        # Create one AxonalTract per connection (single-source, keyed by SynapseId).
        # Each SynapseId is unique across all connections (same source/target/population
        # combination must not appear twice in well-formed brain graphs).
        axonal_tracts: Dict[SynapseId, AxonalTract] = {}
        for conn_spec in self._connection_specs:
            axonal_tract = self._create_axonal_tract(conn_spec, regions, device=device)
            synapse_id = axonal_tract.spec.synapse_id
            if synapse_id in axonal_tracts:
                raise ValueError(
                    f"Duplicate connection {synapse_id}. "
                    f"Each source→target→population combination must be unique."
                )
            axonal_tracts[synapse_id] = axonal_tract
            conn_spec.instance = axonal_tract

        # Register external input sources (if any)
        # These are provided by the training loop, not from other brain regions
        for ext_spec in self._external_input_specs:
            target_region = regions[ext_spec.synapse_id.target_region]
            weight_scale = _apply_stp_correction(ext_spec.weight_scale, ext_spec.stp_config)
            target_region.add_input_source(
                synapse_id=ext_spec.synapse_id,
                n_input=ext_spec.n_input,
                connectivity=ext_spec.connectivity,
                weight_scale=weight_scale,
                stp_config=ext_spec.stp_config,
                learning_strategy=ext_spec.learning_strategy,
                device=device,
            )
            if ext_spec.stp_config is not None and isinstance(weight_scale, ConductanceScaledSpec):
                sid = ext_spec.synapse_id
                if sid in target_region.stp_modules:
                    target_region.stp_modules[sid].initialize_to_steady_state(
                        weight_scale.source_rate_hz
                    )

        # Finalize initialization for regions that need post-connection setup
        # This allows regions to build components that depend on complete connectivity
        # (e.g., thalamus gap junctions that need all input sources)
        for region in regions.values():
            if hasattr(region, 'finalize_initialization'):
                region.finalize_initialization()

        # Create Brain
        brain = Brain(
            config=self.brain_config,
            regions=regions,
            axonal_tracts=axonal_tracts,
            device=device,
        )

        # Validate neuromodulator subscriptions: every channel declared in a region's
        # neuromodulator_subscriptions must be published by at least one other region.
        # Raises ValueError at build time rather than silently returning None at runtime.
        brain.neuromodulator_hub.validate()

        return brain

    # =============================================================================
    # Preset Architectures
    # =============================================================================

    @classmethod
    def register_preset(cls, name: str, description: str, builder_fn: PresetBuilderFn) -> None:
        """Register a preset architecture.

        Args:
            name: Preset name (e.g., "default")
            description: Human-readable description
            builder_fn: Function that configures a BrainBuilder
        """
        cls._presets[name] = PresetArchitecture(
            name=name,
            description=description,
            builder_fn=builder_fn,
        )

    @classmethod
    def preset_builder(
        cls,
        name: str,
        brain_config: Optional[BrainConfig] = None,
        **overrides: Any
    ) -> BrainBuilder:
        """Create builder initialized with preset architecture.

        Unlike preset(), this returns the builder so you can modify it
        before calling build().

        Args:
            name: Preset name (e.g., "default")
            brain_config: Brain configuration
            **overrides: Override default preset parameters

        Returns:
            BrainBuilder instance with preset applied

        Raises:
            KeyError: If preset name not found
        """
        if name not in cls._presets:
            available = list(cls._presets.keys())
            raise KeyError(f"Preset '{name}' not found. Available: {available}")

        preset = cls._presets[name]
        builder = cls(brain_config)
        preset.builder_fn(builder, **overrides)
        return builder

    @classmethod
    def preset(
        cls,
        name: str,
        brain_config: Optional[BrainConfig] = None,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
        **overrides: Any
    ) -> Brain:
        """Create brain from preset architecture.

        Args:
            name: Preset name (e.g., "default")
            brain_config: Brain configuration
            **overrides: Override default preset parameters

        Returns:
            Constructed Brain instance

        Raises:
            KeyError: If preset name not found
        """
        builder = cls.preset_builder(name, brain_config, **overrides)
        return builder.build(device=device)


# Type alias for preset builder functions
# Accepts BrainBuilder and optional keyword overrides
PresetBuilderFn = Callable[[BrainBuilder], None]


class PresetArchitecture:
    """Container for preset architecture definition."""

    def __init__(
        self,
        name: str,
        description: str,
        builder_fn: PresetBuilderFn,
    ):
        self.name = name
        self.description = description
        self.builder_fn = builder_fn


# ============================================================================
# Built-in Preset Architectures
# ============================================================================
# Import after class definitions to avoid circular imports.

from thalia.brain.presets.default import build as _build_default  # noqa: E402
from thalia.brain.presets.basal_ganglia import build as _build_bg  # noqa: E402
from thalia.brain.presets.medial_temporal_lobe import build as _build_mtl  # noqa: E402

BrainBuilder.register_preset(
    name="default",
    description="Default biologically realistic brain architecture",
    builder_fn=_build_default,
)
BrainBuilder.register_preset(
    name="basal_ganglia",
    description="Basal Ganglia circuit (Striatum + GPe + GPi + STN + SNr + LHb + RMTg)",
    builder_fn=_build_bg,
)
BrainBuilder.register_preset(
    name="medial_temporal_lobe",
    description="Medial Temporal Lobe circuit (Medial Septum + Entorhinal Cortex + Hippocampus, optional Subiculum)",
    builder_fn=_build_mtl,
)
