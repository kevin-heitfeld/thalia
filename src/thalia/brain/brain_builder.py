"""
Brain Builder - Fluent API for Brain Construction

This module provides a fluent, progressive API for building brain architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from thalia.brain.regions.population_names import (
    ExternalPopulation,
    BLAPopulation,
    CeAPopulation,
    CerebellumPopulation,
    CortexPopulation,
    DRNPopulation,
    ECPopulation,
    GPePopulation,
    HippocampusPopulation,
    LHbPopulation,
    LocusCoeruleusPopulation,
    MedialSeptumPopulation,
    NucleusBasalisPopulation,
    PrefrontalPopulation,
    RMTgPopulation,
    SNcPopulation,
    STNPopulation,
    StriatumPopulation,
    SubstantiaNigraPopulation,
    ThalamusPopulation,
    VTAPopulation,
)
from thalia.components import STPConfig, ConductanceScaledSpec
from thalia.components.synapses.stp import (
    CORTICOSTRIATAL_PRESET,
    MOSSY_FIBER_PRESET,
    SCHAFFER_COLLATERAL_PRESET,
    THALAMO_STRIATAL_PRESET,
)
from thalia.typing import (
    PopulationName,
    PopulationSizes,
    ReceptorType,
    RegionName,
    SynapseId,
)

from .axonal_tract import AxonalTract, AxonalTractSourceSpec
from .brain import DynamicBrain
from .configs import BrainConfig, NeuralRegionConfig
from .regions import NeuralRegionRegistry, NeuralRegion


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
        # The same STP utilization factor applies since FSI now receives the same stp_config.
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
                stp_utilization_factor=weight_scale.stp_utilization_factor,
            )
        else:
            fsi_ws = weight_scale * 10.0  # type: ignore[operator]

        # TAN weight derivation:
        # TANs are large cholinergic neurons (g_L=0.04, tau_E=10ms) with intrinsic
        # pacemaking; afferent drive should bring them sub-threshold (target_v_inf=0.95)
        # so pacemaking + input together reach threshold.  Half the D1/D2 fraction.
        if isinstance(weight_scale, ConductanceScaledSpec):
            tan_ws: Union[float, ConductanceScaledSpec] = ConductanceScaledSpec(
                source_rate_hz=weight_scale.source_rate_hz,
                target_g_L=0.04,          # TAN leak conductance (slow, tau_m≈25ms)
                target_tau_E_ms=10.0,     # Slower cholinergic AMPA
                target_v_inf=0.95,        # Sub-threshold; intrinsic pacemaking closes gap
                fraction_of_drive=weight_scale.fraction_of_drive * 0.5,
                stp_utilization_factor=weight_scale.stp_utilization_factor,
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

        target_region.add_input_source(
            synapse_id=synapse_id,
            n_input=source_size,
            connectivity=conn_spec.connectivity,
            weight_scale=conn_spec.weight_scale,
            stp_config=conn_spec.stp_config,
        )

        return AxonalTract(
            spec=spec,
            dt_ms=self.brain_config.dt_ms,
            device=self.brain_config.device,
        )

    def build(self) -> DynamicBrain:
        """Build DynamicBrain from specifications.

        Steps:
            1. Validate graph
            2. Instantiate all regions from registry
            3. Instantiate all axonal tracts from connection specs
            4. Create DynamicBrain instance with regions and axonal tracts

        Returns:
            Constructed DynamicBrain instance

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

            config.device = self.brain_config.device
            config.seed = self.brain_config.seed
            config.dt_ms = self.brain_config.dt_ms

            region = NeuralRegionRegistry.create(
                spec.registry_name,
                config=config,
                population_sizes=spec.population_sizes,
                region_name=name,
            )

            regions[name] = region
            spec.instance = region

        # Create one AxonalTract per connection (single-source, keyed by SynapseId).
        # Each SynapseId is unique across all connections (same source/target/population
        # combination must not appear twice in well-formed brain graphs).
        axonal_tracts: Dict[SynapseId, AxonalTract] = {}
        for conn_spec in self._connection_specs:
            axonal_tract = self._create_axonal_tract(conn_spec, regions)
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
            target_region.add_input_source(
                synapse_id=ext_spec.synapse_id,
                n_input=ext_spec.n_input,
                connectivity=ext_spec.connectivity,
                weight_scale=ext_spec.weight_scale,
                stp_config=ext_spec.stp_config,
            )

        # Finalize initialization for regions that need post-connection setup
        # This allows regions to build components that depend on complete connectivity
        # (e.g., thalamus gap junctions that need all input sources)
        for region in regions.values():
            if hasattr(region, 'finalize_initialization'):
                region.finalize_initialization()

        # Create DynamicBrain
        brain = DynamicBrain(
            config=self.brain_config,
            regions=regions,
            axonal_tracts=axonal_tracts,
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
        **overrides: Any
    ) -> DynamicBrain:
        """Create brain from preset architecture.

        Args:
            name: Preset name (e.g., "default")
            brain_config: Brain configuration
            **overrides: Override default preset parameters

        Returns:
            Constructed DynamicBrain instance

        Raises:
            KeyError: If preset name not found
        """
        builder = cls.preset_builder(name, brain_config, **overrides)
        return builder.build()


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

BrainBuilder.register_preset(
    name="default",
    description="Default biologically realistic brain architecture",
    builder_fn=_build_default,
)
