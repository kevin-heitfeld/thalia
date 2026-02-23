"""
Brain Builder - Fluent API for Brain Construction

This module provides a fluent, progressive API for building brain architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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
from thalia.components import STPConfig
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
    weight_scale: float
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
    weight_scale: float
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
        weight_scale: float,
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
        if weight_scale < 0:
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
        weight_scale: float,
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
        if weight_scale < 0:
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
        weight_scale: float,
        *,
        target_region: RegionName = "striatum",
        d2_weight_scale: Optional[float] = None,
        fsi_connectivity: float = 0.5,
        tan_connectivity: float = 0.3,
        stp_config: Optional[STPConfig] = None,
    ) -> "BrainBuilder":
        """Connect a source region to the striatum with explicit D1, D2, FSI, and TAN pathways.

        Creates four ConnectionSpecs (one per sub-population) at the same axonal
        delay but with biologically-motivated connectivity and weight differences:

        +----------+----------------+---------------------+--------------------+
        | Pop      | connectivity   | weight_scale        | STP                |
        +==========+================+=====================+====================+
        | D1       | *connectivity* | *weight_scale*      | *stp_config*       |
        | D2       | *connectivity* | *d2_weight_scale*   | *stp_config*       |
        | FSI      | *fsi_connectivity* | *weight_scale*  | None               |
        | TAN      | *tan_connectivity* | weight_scale×0.5 | None             |
        +----------+----------------+---------------------+--------------------+

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
            weight_scale: Initial weight scale for D1 MSNs (and FSI baseline).
            target_region: Name of the target striatal region (default: ``"striatum"``).
            d2_weight_scale: Weight scale for D2 MSNs; defaults to *weight_scale* if None.
            fsi_connectivity: Connection probability for FSI input (default 0.5).
            tan_connectivity: Connection probability for TAN input (default 0.3).
            stp_config: Short-term plasticity config applied to D1 and D2 only;
                        FSI and TAN always receive ``None``.

        Returns:
            Self for method chaining.
        """
        d2_ws = d2_weight_scale if d2_weight_scale is not None else weight_scale

        for target_pop, conn, ws, use_stp in [
            (StriatumPopulation.D1,  connectivity,     weight_scale,      True),
            (StriatumPopulation.D2,  connectivity,     d2_ws,             True),
            (StriatumPopulation.FSI, fsi_connectivity, weight_scale,      False),
            (StriatumPopulation.TAN, tan_connectivity, weight_scale * 0.5, False),
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
                stp_config=stp_config if use_stp else None,
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


def _build_default(builder: BrainBuilder, **overrides: Any) -> None:
    """Default biologically realistic brain architecture with empirically grounded population sizes and axonal delays."""
    # =============================================================================
    # Define default population sizes based on rodent brain estimates (scaled down for tractability)
    # Can be overridden via **overrides parameter
    # =============================================================================

    default_cerebellum_sizes: PopulationSizes = {
        CerebellumPopulation.GRANULE: 20000,  # Granule:Purkinje = 200:1 (biology: 1000:1, most numerous neurons in brain)
        CerebellumPopulation.PURKINJE: 100,   # Purkinje cells are sole output of cerebellar cortex, provide strong inhibition to DCN
        CerebellumPopulation.DCN: 100,        # DCN:Purkinje = 1:1 (biology ~1:1, DCN are sole cerebellar output neurons
    }
    default_bla_sizes: PopulationSizes = {
        BLAPopulation.PRINCIPAL: 2000,   # ~60% of BLA — glutamatergic fear/extinction engrams
        BLAPopulation.PV: 500,           # ~20% — fast-spiking, feedforward inhibition
        BLAPopulation.SOM: 300,          # ~10% — slow, dendritic inhibition (extinction)
    }
    default_cea_sizes: PopulationSizes = {
        CeAPopulation.LATERAL: 750,      # CeL — integrative (ON/OFF cell division)
        CeAPopulation.MEDIAL: 500,       # CeM — output nucleus (→ LC, LHb)
    }
    default_cortex_sensory_sizes: PopulationSizes = {
        CortexPopulation.L23_PYR: 1200,  # 40% (supragranular, associative)
        CortexPopulation.L4_PYR: 300,    # 10% (granular, thalamic input — prominent in sensory areas)
        CortexPopulation.L5_PYR: 450,    # 15% (output to subcortex)
        CortexPopulation.L6A_PYR: 150,   # 5% (corticothalamic type I)
        CortexPopulation.L6B_PYR: 900,   # 30% (corticothalamic type II)
    }
    default_cortex_association_sizes: PopulationSizes = {
        # Association cortex (e.g. IT, PPC, PMC) has proportionally *thin* L4
        # (less thalamic drive) and thick L2/3 (more corticocortical integration).
        # Sizes scaled to ~75% of sensory column to reflect smaller absolute area.
        CortexPopulation.L23_PYR: 900,   # 40% (thicker relative L2/3 — strong associative integration)
        CortexPopulation.L4_PYR: 150,    # 5%  (thin L4 — hallmark of association / agranular areas)
        CortexPopulation.L5_PYR: 375,    # 15% (strong subcortical output to striatum, STN)
        CortexPopulation.L6A_PYR: 120,   # 5%  (corticothalamic type I)
        CortexPopulation.L6B_PYR: 750,   # 30% (corticothalamic type II)
    }
    default_drn_sizes: PopulationSizes = {
        # Biology: ~100K-200K neurons in humans (~85% serotonergic, ~15% GABAergic).
        # Scaled to 5K for tractability; 10:1 ratio matches biological composition.
        DRNPopulation.SEROTONIN: 5000,   # Serotonergic projection neurons (2-4 Hz tonic pacemaker, I_h driven)
        DRNPopulation.GABA: 500,         # Local GABAergic interneurons (5-HT1A → feedback inhibition)
    }
    default_gpe_sizes: PopulationSizes = {
        GPePopulation.ARKYPALLIDAL: 700,   # ~25%, project back to striatum (global suppression)
        GPePopulation.PROTOTYPIC: 2000,    # ~75% of GPe, project to STN + SNr (~50 Hz tonic)
    }
    default_ec_sizes: PopulationSizes = {
        ECPopulation.EC_II: 400,    # Layer II stellate cells: grid/place → DG, CA3 (perforant path)
        ECPopulation.EC_III: 300,   # Layer III pyramidal time cells → CA1 (temporoammonic direct path)
        ECPopulation.EC_V: 200,     # Layer V output back-projection ← CA1 → neocortex
    }
    default_hippocampus_sizes: PopulationSizes = {
        HippocampusPopulation.DG: 500,   # cortex:hippocampus biological ratio (400:1)
        HippocampusPopulation.CA3: 250,  # Autoassociative recurrent memory
        HippocampusPopulation.CA2: 75,   # Small transitional zone (CA2 is tiny in biology)
        HippocampusPopulation.CA1: 375,  # Output to cortex (slightly larger than CA3)
    }
    default_lhb_sizes: PopulationSizes = {
        LHbPopulation.PRINCIPAL: 500,      # Glutamatergic, excited by SNr (bad outcome signal)
    }
    default_locus_coeruleus_sizes: PopulationSizes = {
        LocusCoeruleusPopulation.NE: 1600,
        LocusCoeruleusPopulation.GABA: 300,
    }
    default_medial_septum_sizes: PopulationSizes = {
        MedialSeptumPopulation.ACH: 200,
        MedialSeptumPopulation.GABA: 200,
    }
    default_nucleus_basalis_sizes: PopulationSizes = {
        NucleusBasalisPopulation.ACH: 3000,
        NucleusBasalisPopulation.GABA: 500,
    }
    default_prefrontal_sizes: PopulationSizes = {
        PrefrontalPopulation.EXECUTIVE: 800,
    }
    default_rmtg_sizes: PopulationSizes = {
        RMTgPopulation.GABA: 1000,         # GABAergic, inhibit VTA DA (dopamine pause)
    }
    default_striatum_sizes: PopulationSizes = {
        StriatumPopulation.D1: 200,
        StriatumPopulation.D2: 200,
        'n_actions': 10,
        'neurons_per_action': 10,
    }
    default_substantia_nigra_sizes: PopulationSizes = {
        SubstantiaNigraPopulation.VTA_FEEDBACK: 1000,  # SNr receives dense feedback from VTA dopamine neurons (inhibitory)
    }
    default_snc_sizes: PopulationSizes = {
        SNcPopulation.DA: 1500,    # DA:GABA roughly 3:1 (same calibration as VTA)
        SNcPopulation.GABA: 500,
    }
    default_stn_sizes: PopulationSizes = {
        STNPopulation.STN: 500,            # Glutamatergic pacemakers (~20 Hz autonomous)
    }
    default_thalamus_sizes: PopulationSizes = {
        ThalamusPopulation.RELAY: 400,  # Thalamic relay neurons (input from sensory pathways, output to cortex)
        ThalamusPopulation.TRN: 40,     # 10:1 relay:TRN ratio
    }
    default_vta_sizes: PopulationSizes = {
        # 2500 total DA neurons split into mesolimbic (~55%) and mesocortical (~35%)
        # per Lammel et al. (2008): ~55% project to ventral striatum/limbic, ~35% to PFC.
        # Remaining ~10% are glutamatergic (not modeled).
        VTAPopulation.DA_MESOLIMBIC: 1375,   # 55% \u2014 reward/motivation; D2 autoreceptors present
        VTAPopulation.DA_MESOCORTICAL: 875,  # 35% \u2014 executive/arousal; no D2 autoreceptors
        VTAPopulation.GABA: 1000,  # 40% of DA population for local inhibitory control
    }

    # Merge with overrides (user overrides take precedence)
    sizes_overrides: Dict[RegionName, PopulationSizes] = overrides.get('population_sizes', {})

    bla_sizes: PopulationSizes = {**default_bla_sizes, **sizes_overrides.get('basolateral_amygdala', {})}
    cea_sizes: PopulationSizes = {**default_cea_sizes, **sizes_overrides.get('central_amygdala', {})}
    cerebellum_sizes: PopulationSizes = {**default_cerebellum_sizes, **sizes_overrides.get('cerebellum', {})}
    cortex_sensory_sizes: PopulationSizes = {**default_cortex_sensory_sizes, **sizes_overrides.get('cortex_sensory', {})}
    cortex_association_sizes: PopulationSizes = {**default_cortex_association_sizes, **sizes_overrides.get('cortex_association', {})}
    drn_sizes: PopulationSizes = {**default_drn_sizes, **sizes_overrides.get('dorsal_raphe', {})}
    ec_sizes: PopulationSizes = {**default_ec_sizes, **sizes_overrides.get('entorhinal_cortex', {})}
    gpe_sizes: PopulationSizes = {**default_gpe_sizes, **sizes_overrides.get('globus_pallidus_externa', {})}
    hippocampus_sizes: PopulationSizes = {**default_hippocampus_sizes, **sizes_overrides.get('hippocampus', {})}
    lhb_sizes: PopulationSizes = {**default_lhb_sizes, **sizes_overrides.get('lateral_habenula', {})}
    locus_coeruleus_sizes: PopulationSizes = {**default_locus_coeruleus_sizes, **sizes_overrides.get('locus_coeruleus', {})}
    medial_septum_sizes: PopulationSizes = {**default_medial_septum_sizes, **sizes_overrides.get('medial_septum', {})}
    nucleus_basalis_sizes: PopulationSizes = {**default_nucleus_basalis_sizes, **sizes_overrides.get('nucleus_basalis', {})}
    prefrontal_sizes: PopulationSizes = {**default_prefrontal_sizes, **sizes_overrides.get('prefrontal', {})}
    rmtg_sizes: PopulationSizes = {**default_rmtg_sizes, **sizes_overrides.get('rostromedial_tegmentum', {})}
    striatum_sizes: PopulationSizes = {**default_striatum_sizes, **sizes_overrides.get('striatum', {})}
    substantia_nigra_sizes: PopulationSizes = {**default_substantia_nigra_sizes, **sizes_overrides.get('substantia_nigra', {})}
    snc_sizes: PopulationSizes = {**default_snc_sizes, **sizes_overrides.get('substantia_nigra_compacta', {})}
    stn_sizes: PopulationSizes = {**default_stn_sizes, **sizes_overrides.get('subthalamic_nucleus', {})}
    thalamus_sizes: PopulationSizes = {**default_thalamus_sizes, **sizes_overrides.get('thalamus', {})}
    vta_sizes: PopulationSizes = {**default_vta_sizes, **sizes_overrides.get('vta', {})}

    external_reward_size: int = sizes_overrides.get(SynapseId._EXTERNAL_REGION_NAME, {}).get(ExternalPopulation.REWARD, 100)
    external_sensory_size: Optional[int] = sizes_overrides.get(SynapseId._EXTERNAL_REGION_NAME, {}).get(ExternalPopulation.SENSORY, None)
    if external_sensory_size is None:
        external_sensory_size = thalamus_sizes[ThalamusPopulation.RELAY]

    # =============================================================================
    # Define regions with biologically realistic population sizes
    # =============================================================================

    builder.add_region("basolateral_amygdala", "basolateral_amygdala", population_sizes=bla_sizes)
    builder.add_region("central_amygdala", "central_amygdala", population_sizes=cea_sizes)
    builder.add_region("cerebellum", "cerebellum", population_sizes=cerebellum_sizes)
    builder.add_region("cortex_sensory", "cortical_column", population_sizes=cortex_sensory_sizes)
    builder.add_region("cortex_association", "cortical_column", population_sizes=cortex_association_sizes)
    builder.add_region("dorsal_raphe", "dorsal_raphe", population_sizes=drn_sizes)
    builder.add_region("entorhinal_cortex", "entorhinal_cortex", population_sizes=ec_sizes)
    builder.add_region("globus_pallidus_externa", "globus_pallidus_externa", population_sizes=gpe_sizes)
    builder.add_region("hippocampus", "hippocampus", population_sizes=hippocampus_sizes)
    builder.add_region("lateral_habenula", "lateral_habenula", population_sizes=lhb_sizes)
    builder.add_region("locus_coeruleus", "locus_coeruleus", population_sizes=locus_coeruleus_sizes)
    builder.add_region("medial_septum", "medial_septum", population_sizes=medial_septum_sizes)
    builder.add_region("nucleus_basalis", "nucleus_basalis", population_sizes=nucleus_basalis_sizes)
    builder.add_region("prefrontal", "prefrontal", population_sizes=prefrontal_sizes)
    builder.add_region("rostromedial_tegmentum", "rostromedial_tegmentum", population_sizes=rmtg_sizes)
    builder.add_region("striatum", "striatum", population_sizes=striatum_sizes)
    builder.add_region("substantia_nigra", "substantia_nigra", population_sizes=substantia_nigra_sizes)
    builder.add_region("substantia_nigra_compacta", "substantia_nigra_compacta", population_sizes=snc_sizes)
    builder.add_region("subthalamic_nucleus", "subthalamic_nucleus", population_sizes=stn_sizes)
    builder.add_region("thalamus", "thalamus", population_sizes=thalamus_sizes)
    builder.add_region("vta", "vta", population_sizes=vta_sizes)

    # =============================================================================
    # Define connections with biologically realistic axonal delays
    # =============================================================================

    # External Sensory Input → Thalamus: Ascending sensory pathways
    # This represents retinogeniculate (vision), cochlear (audition), or
    # somatosensory pathways (touch). Sensory input is provided by training
    # loop via region_inputs dict, not from a brain region.
    #
    # Biology:
    # - Very fast, heavily myelinated pathways
    # - Direct projection to thalamic relay neurons
    # - Parallel collateral to TRN for feedforward inhibition
    # - Distance: ~1-2cm, conduction velocity: ~20-30 m/s → ~0.5-1ms delay
    #   (included implicitly in simulation timestep, no additional delay needed)
    #
    # NOTE: Sensory input size can differ from relay_size for spatial
    # compression/expansion (e.g., 256 retinal ganglion cells → 128 LGN neurons)
    builder.add_external_input_source(
        synapse_id=SynapseId.external_sensory_to_thalamus_relay("thalamus"),
        n_input=external_sensory_size,
        connectivity=0.25,
        weight_scale=0.0005,
    )

    # External reward → VTA DA: Direct reward signal delivery
    # Population-coded spikes generated by DynamicBrain.deliver_reward() and
    # injected directly, bypassing a dedicated region.  n_input controls the
    # size of the spike vector expected by deliver_reward() at runtime.
    builder.add_external_input_source(
        synapse_id=SynapseId.external_reward_to_vta_da("vta"),
        n_input=external_reward_size,
        connectivity=0.7,
        weight_scale=0.0008,
    )

    # Thalamus → CorticalColumn L4: Thalamocortical projection with feedforward inhibition
    # Fast, heavily myelinated pathway
    # Distance: ~2-3cm, conduction velocity: ~10-20 m/s → 2-3ms delay
    # Uses 'relay' population: Only relay neurons project to cortex (TRN provides lateral inhibition)
    #
    # Biology: Thalamic afferents synapse onto BOTH pyramidal and PV interneurons
    # PV cells have lower thresholds (0.8 vs 1.4), so they receive weaker weights
    # This creates fast feedforward inhibition: PV fires before pyramidal

    # Thalamus → L4 Pyramidal: Main thalamocortical drive
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=5.0,
        connectivity=0.7,
        weight_scale=0.00008,
    )

    # Thalamus → L4 PV: Feedforward inhibition drive
    # Biology: Same axons branch to PV cells for fast inhibition
    # Weaker weights compensate for PV's lower threshold (0.8 vs 1.4)
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_INHIBITORY_PV,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=5.0,
        connectivity=0.7,
        weight_scale=0.00002,
    )

    # CorticalColumn L6a/L6b → Thalamus: Dual corticothalamic feedback pathways
    # L6a (type I) → TRN: Inhibitory modulation for selective attention (slow pathway, low gamma 25-35 Hz)
    # L6b (type II) → Relay: Excitatory modulation for precision processing (fast pathway, high gamma 60-80 Hz)
    #
    # Biologically realistic delays:
    # - Corticothalamic axons: 8-12ms (longer than thalamocortical due to less myelination and more synapses)
    # - L6a→TRN: ~10ms (type I pathway, standard corticothalamic)
    # - L6b→Relay: ~5ms (type II pathway, slightly faster direct projection)
    #
    # Total feedback loop timing includes:
    # - Axonal delays (specified here): 10ms + 5ms
    # - Synaptic delays (~1-2ms per synapse): multiple synapses in loop
    # - Neural integration (tau_E=5ms, tau_I=10ms): conductance buildup/decay
    # - Membrane dynamics (tau_mem~20ms): integration to threshold
    # - Refractory periods (tau_ref): post-spike delays
    # These neural dynamics add ~10-20ms to total loop period, producing gamma oscillations
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        axonal_delay_std_ms=15.0,
        connectivity=0.3,
        weight_scale=0.001,
    )
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L6B_PYR,
            target_region="thalamus",
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=0.0001,
    )

    # =========================================================================
    # CorticalColumn → EntorhinalCortex → Hippocampus: Bidirectional memory integration
    # All neocortical signals enter hippocampus via entorhinal cortex (EC) gateway.
    # =========================================================================

    # Sensory cortex L5 → EC_II  (spatial/sensory context → perforant path)
    # Moderately myelinated; distance ~3-5 cm, velocity ~5-10 m/s → 5-7 ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="entorhinal_cortex",
            target_population=ECPopulation.EC_II,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=9.0,
        connectivity=0.3,
        weight_scale=0.0006,
    )

    # Association cortex L2/3 → EC_II  (semantic / multi-modal context → perforant path)
    # Association cortex is the primary driver of EC layer II grid cells.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L23_PYR,
            target_region="entorhinal_cortex",
            target_population=ECPopulation.EC_II,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.35,
        weight_scale=0.0006,
    )

    # Association cortex L2/3 → EC_III  (temporal / semantic context → temporoammonic path)
    # EC layer III time cells receive strong association cortex input.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L23_PYR,
            target_region="entorhinal_cortex",
            target_population=ECPopulation.EC_III,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.30,
        weight_scale=0.0005,
    )

    # EC_II → Hippocampus DG  (perforant path — principal input, facilitating STP)
    # Outer molecular layer: sparse, 15-20 % connectivity.
    # MOSSY_FIBER facilitating STP approximates the burst-then-fade characteristic
    # of true perforant path EPSPs at DG granule cells.
    builder.connect(
        synapse_id=SynapseId(
            source_region="entorhinal_cortex",
            source_population=ECPopulation.EC_II,
            target_region="hippocampus",
            target_population=HippocampusPopulation.DG,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.25,
        weight_scale=0.0008,
        stp_config=MOSSY_FIBER_PRESET.configure(),
    )

    # EC_II → Hippocampus CA3  (direct perforant path to CA3 — less sparse)
    # Stratum lacunosum-moleculare projection; weaker than DG perforant.
    builder.connect(
        synapse_id=SynapseId(
            source_region="entorhinal_cortex",
            source_population=ECPopulation.EC_II,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.5,
        axonal_delay_std_ms=5.0,
        connectivity=0.20,
        weight_scale=0.0005,
        stp_config=MOSSY_FIBER_PRESET.configure(),
    )

    # EC_III → Hippocampus CA1  (temporoammonic direct path — depressing STP)
    # Distal apical dendrites of CA1; bypasses DG/CA3 (direct path).
    # Depression emphasizes novelty: strong initial pulse, then adapts.
    builder.connect(
        synapse_id=SynapseId(
            source_region="entorhinal_cortex",
            source_population=ECPopulation.EC_III,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.25,
        weight_scale=0.0006,
        stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),  # Depressing (U=0.5)
    )

    # Hippocampus CA1 → EC_V  (back-projection — memory index to layer V)
    # Subicular / CA1 axons ascend to entorhinal layer V.
    # ~3 ms synaptic latency; moderate connectivity.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="entorhinal_cortex",
            target_population=ECPopulation.EC_V,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.30,
        weight_scale=0.0004,
    )

    # EC_V → Association cortex L2/3  (memory indexing output → cortical consolidation)
    # EC layer V neurons broadcast the compressed hippocampal memory index back to
    # neocortex, driving systems consolidation during offline replay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="entorhinal_cortex",
            source_population=ECPopulation.EC_V,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=9.0,
        connectivity=0.25,
        weight_scale=0.0003,
    )

    # Association → PFC: Higher-level integrated representations drive executive control
    # Association cortex is the primary corticocortical driver of PFC — sensory percepts
    # are first integrated across modalities in association areas before reaching PFC.
    # Distance: ~5-10cm, conduction velocity: ~3-8 m/s → 10-15ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L23_PYR,
            target_region="prefrontal",
            target_population=PrefrontalPopulation.EXECUTIVE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.5,
        axonal_delay_std_ms=20.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # Multi-source → Striatum: Corticostriatal + hippocampostriatal + PFC inputs
    # connect_to_striatum() creates explicit D1, D2, FSI, and TAN pathways for each source,
    # so all populations receive actual synaptic input via dedicated axonal tracts.
    # Per-target delays model different myelination patterns (Gerfen & Surmeier 2011):
    # - CorticalColumn → Striatum: Fast, heavily myelinated, short distance (~2-4cm) → 3-5ms
    # - Hippocampus → Striatum: Moderate, longer distance (~4-6cm) → 7-10ms
    # - PFC → Striatum: Variable, longest distance (~6-10cm) → 12-18ms
    builder.connect_to_striatum(
        source_region="cortex_sensory",
        source_population=CortexPopulation.L5_PYR,
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.3,
        weight_scale=0.0001,
        stp_config=CORTICOSTRIATAL_PRESET.configure(),
    )
    builder.connect_to_striatum(
        source_region="hippocampus",
        source_population=HippocampusPopulation.CA1,
        axonal_delay_ms=8.5,
        axonal_delay_std_ms=13.0,
        connectivity=0.3,
        weight_scale=0.0002,
        stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
    )
    builder.connect_to_striatum(
        source_region="prefrontal",
        source_population=PrefrontalPopulation.EXECUTIVE,
        axonal_delay_ms=15.0,
        axonal_delay_std_ms=22.5,
        connectivity=0.3,
        weight_scale=0.0002,
        stp_config=CORTICOSTRIATAL_PRESET.configure(),
    )

    # Striatum → PFC: Basal ganglia gating of working memory
    # Via thalamus (MD/VA nuclei), total distance ~8-12cm (Haber 2003)
    # Includes striatum→thalamus→PFC relay → 15-20ms total delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="striatum",
            source_population=StriatumPopulation.D1,
            target_region="prefrontal",
            target_population=PrefrontalPopulation.EXECUTIVE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=17.5,
        axonal_delay_std_ms=26.0,
        connectivity=0.6,
        weight_scale=0.0003,
    )

    # Cerebellum: Motor/cognitive forward models
    # Receives multi-modal input (sensory + goals), outputs predictions
    # Corticopontocerebellar pathway: via pontine nuclei (Schmahmann 1996)
    # Distance: ~10-15cm total, includes relay → 20-30ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="cerebellum",
            target_population=CerebellumPopulation.GRANULE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=25.0,
        axonal_delay_std_ms=37.5,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # PFC → Cerebellum: similar pathway length
    # Goal/context input
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE,
            target_region="cerebellum",
            target_population=CerebellumPopulation.GRANULE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=25.0,
        axonal_delay_std_ms=37.5,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # Cerebellum → CorticalColumn L4: via thalamus (VL/VA nuclei), moderately fast
    # Distance: ~8-12cm, includes thalamic relay → 15-20ms delay
    # Forward model predictions
    # Split into pyramidal and PV for feedforward inhibition (same as thalamus)

    # Cerebellum → L4 Pyramidal: Motor predictions
    builder.connect(
        synapse_id=SynapseId(
            source_region="cerebellum",
            source_population=CerebellumPopulation.DCN,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=17.5,
        axonal_delay_std_ms=26.0,
        connectivity=0.3,
        weight_scale=0.0008,
    )

    # Cerebellum → L4 PV: Feedforward inhibition for motor predictions
    builder.connect(
        synapse_id=SynapseId(
            source_region="cerebellum",
            source_population=CerebellumPopulation.DCN,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_INHIBITORY_PV,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=17.5,
        axonal_delay_std_ms=26.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # PFC ⇄ Hippocampus: Bidirectional memory-executive integration
    # Critical for goal-directed memory retrieval and episodic future thinking

    # PFC → Hippocampus: Top-down memory retrieval, schema application
    # PFC guides hippocampal retrieval via direct monosynaptic pathway
    # Distance: ~5-7cm, conduction velocity: ~3-5 m/s → 12-18ms delay
    # Enables: goal-directed memory search, schema-based encoding modulation
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=15.0,
        axonal_delay_std_ms=22.5,
        connectivity=0.3,
        weight_scale=0.0003,
    )

    # Hippocampus → PFC: Memory-guided decision making
    # Hippocampal replay guides PFC working memory and planning
    # Distance: ~5-7cm, similar pathway to reverse direction → 10-15ms delay
    # Enables: episodic future thinking, memory-guided decisions
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="prefrontal",
            target_population=PrefrontalPopulation.EXECUTIVE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=18.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # PFC → CorticalColumn: Top-down attention and cognitive control
    # PFC biases cortical processing via direct feedback projections to L2/3
    # Corticocortical feedback: ~5-8cm, conduction velocity: ~3-6 m/s → 10-15ms
    # Biological accuracy: PFC feedback targets superficial layers (L2/3), NOT L4
    # This bypasses thalamic input and directly modulates cortical representations
    # Enables: attentional bias, cognitive control over perception, working memory maintenance
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=18.0,
        connectivity=0.3,
        weight_scale=0.0005,
    )

    # Thalamus → Hippocampus: Direct sensory-to-memory pathway (bypass cortex)
    # Nucleus reuniens provides direct thalamic input to hippocampus
    # Fast subcortical route for unfiltered sensory encoding
    # Distance: ~4-6cm, conduction velocity: ~5-8 m/s → 6-10ms delay
    # Enables: fast sensory encoding, subcortical memory formation
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="hippocampus",
            target_population=HippocampusPopulation.DG,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=12.0,
        connectivity=0.3,
        weight_scale=0.0003,
    )

    # Thalamus → Striatum: Thalamostriatal pathway for habitual responses
    # Direct sensory-action pathway bypassing cortex (Smith et al. 2004, 2009, 2014)
    # Fast subcortical route for stimulus-response habits
    # Distance: ~3-5cm, conduction velocity: ~6-10 m/s → 4-7ms delay
    # Enables: fast habitual responses, stimulus-response learning, subcortical reflexes
    builder.connect_to_striatum(
        source_region="thalamus",
        source_population=ThalamusPopulation.RELAY,
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=0.0002,
        stp_config=THALAMO_STRIATAL_PRESET.configure(),
    )

    # Medial Septum → Hippocampus CA3: Septal theta drive for emergent oscillations
    # GABAergic pacemaker neurons phase-lock hippocampal OLM interneurons
    # OLM dendritic inhibition creates emergent encoding/retrieval separation
    # Distance: ~1-2cm (local subcortical), well-myelinated → 2ms delay
    # CRITICAL: This connection enables emergent theta (replaces hardcoded sinusoid)
    builder.connect(
        synapse_id=SynapseId(
            source_region="medial_septum",
            source_population=MedialSeptumPopulation.GABA,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA3,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.15,
        weight_scale=0.0009,
    )

    # CRITICAL: Hippocampus → Septum feedback inhibition (CA1 → Septum GABA)
    # Biology: CA1 pyramidal cells project back to septum GABAergic neurons
    # When hippocampus is hyperactive, this feedback suppresses septal drive
    # This closes the loop: Septum drives hippocampus, hippocampus regulates septum
    # Distance: ~1-2cm, similar myelination → 2ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="medial_septum",
            target_population=MedialSeptumPopulation.GABA,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.2,
        weight_scale=0.0005,
    )

    # =============================================================================
    # SPIKING DOPAMINE SYSTEM (VTA + SNr + RewardEncoder)
    # =============================================================================
    # Implements biologically-accurate reinforcement learning through burst/pause dynamics.
    # Replaces scalar dopamine broadcast with spike-based volume transmission.
    #
    # Closed-loop TD learning: Striatum → SNr → VTA → Striatum
    # - SNr provides value estimate: V(s) ∝ 1/firing_rate
    # - VTA computes RPE: δ = reward - V(s)
    # - VTA DA neurons burst (δ>0) or pause (δ<0)
    # - Striatum receives DA spikes, converts to concentration via receptors
    # - Three-factor learning: Δw = eligibility × DA_concentration × lr

    # Striatum → SNr: Direct pathway (D1) and indirect pathway (D2)
    # D1 MSNs inhibit SNr (disinhibit thalamus → "Go" signal)
    # D2 MSNs excite SNr via GPe disinhibition (inhibit thalamus → "No-Go" signal)
    # Distance: ~1-2cm, well-myelinated → 2.5ms mean delay
    #
    # BIOLOGICAL ASYMMETRY:
    # - D1 (direct): Monosynaptic GABAergic inhibition → STRONG (weight=0.08)
    # - D2 (indirect): Polysynaptic via GPe→STN→SNr → WEAK (weight=0.01)
    # This 8:1 ratio reflects direct vs indirect pathway potency (direct is much stronger)
    #
    # BIOLOGY: SNR has strong intrinsic pacemaking (persistent Na+, T-type Ca2+)
    # producing 50-70 Hz baseline. Striatum MODULATES this (cannot completely silence).
    # With maximal D1, SNR drops to ~30-40 Hz. With maximal D2, SNR rises to ~80-90 Hz.
    builder.connect(
        synapse_id=SynapseId(
            source_region="striatum",
            source_population=StriatumPopulation.D1,
            target_region="substantia_nigra",
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=7.5,
        connectivity=0.6,
        weight_scale=0.0008,
    )
    # D2 MSNs (indirect, NoGo): Inhibit GPe PROTOTYPIC
    # Biology: D2-MSN → GPe is the first link of the INDIRECT pathway
    # D2 fires → suppresses GPe → disinhibits STN → STN bursts → excites SNr
    # Net effect: SNr fires more → thalamus suppressed → action cancelled
    # Distance: ~1-2cm, well-myelinated → 2-4ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="striatum",
            source_population=StriatumPopulation.D2,
            target_region="globus_pallidus_externa",
            target_population=GPePopulation.PROTOTYPIC,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.5,
        weight_scale=0.0006,
    )

    # =============================================================================
    # BASAL GANGLIA INDIRECT PATHWAY (GPe, STN)
    # =============================================================================
    # Complete biological indirect pathway: D2 → GPe → STN → SNr
    # Also: CorticalColumn (hyperdirect) → STN → SNr (fastest action-suppression pathway)

    # GPe PROTOTYPIC → STN: Inhibitory pacing of STN autonomous pacemaker
    # GPe → STN inhibition is the basis of the GPe-STN oscillatory loop
    # Biology: GABAergic, myelinated → 3-5ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="globus_pallidus_externa",
            source_population=GPePopulation.PROTOTYPIC,
            target_region="subthalamic_nucleus",
            target_population=STNPopulation.STN,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.5,
        weight_scale=0.000025,
    )

    # GPe PROTOTYPIC → STN: Slow GABA_B component (metabotropic K⁺ channel)
    # GABA_B provides late-onset (~100ms), long-duration (~400ms) inhibitory tail,
    # underlying sustained STN suppression and post-inhibitory rebound burst timing.
    # Critical for beta-oscillation power and the hyperdirect pathway suppression window.
    # Biology: Same GPe→STN axons release sufficient GABA to activate metabotropic receptors
    #          at high-frequency bursts; GABA_B requires higher [GABA] than GABA_A.
    builder.connect(
        synapse_id=SynapseId(
            source_region="globus_pallidus_externa",
            source_population=GPePopulation.PROTOTYPIC,
            target_region="subthalamic_nucleus",
            target_population=STNPopulation.STN,
            receptor_type=ReceptorType.GABA_B,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.4,
        weight_scale=0.000015,
    )

    # STN → GPe PROTOTYPIC: Excitatory feedback (closes GPe-STN loop)
    # STN bursts re-excite GPe, sustaining the oscillatory sub-second loop
    # Biology: Glutamatergic, moderately myelinated → 3-5ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="subthalamic_nucleus",
            source_population=STNPopulation.STN,
            target_region="globus_pallidus_externa",
            target_population=GPePopulation.PROTOTYPIC,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.5,
        weight_scale=0.0005,
    )

    # STN → SNr: Net excitatory output of indirect pathway
    # When D2-MSNs fire → GPe inhibited → STN disinhibited → STN bursts → SNr excited
    # Biology: Glutamatergic, moderately myelinated → 3-7ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="subthalamic_nucleus",
            source_population=STNPopulation.STN,
            target_region="substantia_nigra",
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.5,
        weight_scale=0.0004,
    )

    # GPe PROTOTYPIC → SNr: Pallido-nigral inhibitory pathway
    # GPe also directly projects to SNr providing additional inhibitory gating
    # Biology: GABAergic, myelinated → 3-5ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="globus_pallidus_externa",
            source_population=GPePopulation.PROTOTYPIC,
            target_region="substantia_nigra",
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.4,
        weight_scale=0.00003,
    )

    # CorticalColumn L5 → STN: HYPERDIRECT pathway (fastest cortex-basal ganglia route)
    # CorticalColumn sends 'hold' signal directly to STN while direct-pathway decision propagates
    # Arrives at SNr before striatal signals, enabling rapid action suppression
    # Biology: Corticospinal-type axons, heavily myelinated → 3-8ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="subthalamic_nucleus",
            target_population=STNPopulation.STN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=0.0004,
    )

    # =============================================================================
    # INTER-COLUMN CORTICOCORTICAL CONNECTIONS
    # =============================================================================
    # Two-column hierarchy: cortex_sensory → cortex_association → prefrontal.
    #
    # cortex_sensory  – receives thalamic/cerebellar drive, encodes perceptual features
    # cortex_association – integrates sensory with hippocampal memory; routes to PFC
    #                      and subcortex for goal-directed action selection
    #
    # Feedforward (sensory L2/3 → assoc. L4): ~5ms, well-myelinated, ~2-3cm
    # Feedback  (assoc. L6B → sensory L2/3): ~8ms, partly unmyelinated component
    # These implement the canonical predictive-coding FF/FB architecture
    # (Felleman & Van Essen 1991; Bastos et al. 2012).

    # Sensory L2/3 → Association L4: Feedforward percept transfer
    # Supragranular pyramidals project to granular layer of the next-higher area.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=0.0004,
    )

    # Association L6B → Sensory L2/3: Top-down prediction feedback
    # Deep-layer → superficial-layer of lower area (canonical FB pathway).
    # Carries predictions; suppresses expected patterns (predictive coding).
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6B_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=12.0,
        connectivity=0.2,
        weight_scale=0.0004,
    )

    # Hippocampus CA1 → Association L2/3: Retrieved episodic content to context
    # Complements sensory→DG encoding; retrieval arrives at association for integration.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=10.0,
        connectivity=0.3,
        weight_scale=0.0001,
    )

    # PFC → Association L2/3: Top-down executive modulation of higher representations
    # PFC biases both cortical columns; association column receives goal-context signal.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=18.0,
        connectivity=0.3,
        weight_scale=0.0005,
    )

    # Association L5 → Striatum D1/D2/FSI/TAN: Goal-directed corticostriatal projection
    # High-level context drives direct pathway for action selection.
    # Slightly longer delay than sensory→D1 (additional cortical processing).
    # connect_to_striatum() creates all four sub-population pathways explicitly.
    builder.connect_to_striatum(
        source_region="cortex_association",
        source_population=CortexPopulation.L5_PYR,
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=9.0,
        connectivity=0.3,
        weight_scale=0.0001,
        stp_config=CORTICOSTRIATAL_PRESET.configure(),
    )

    # Association L6A → Thalamus TRN: Corticothalamic attention control
    # Association cortex gates thalamic relay to shape L4 input in sensory column.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=18.0,
        connectivity=0.2,
        weight_scale=0.0008,
    )

    # =============================================================================
    # ANTI-REWARD / NEGATIVE RPE PATHWAY (LHb, RMTg → VTA)
    # =============================================================================
    # SNr (high activity = bad outcome) → LHb → RMTg → VTA DA pause
    # This implements the biological negative RPE pathway replacing the old
    # direct SNr→VTA value decoding.

    # SNr → LHb: High SNr activity = suppressed action = bad outcome → LHb excited
    # Biology: SNr projects excitatory input to LHb (glutamatergic via relay cells)
    # Distance: ~1-2cm (adjacent midbrain to epithalamus) → 2-4ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="substantia_nigra",
            source_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            target_region="lateral_habenula",
            target_population=LHbPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.5,
        weight_scale=0.00008,
    )

    # SNr → VTA DA_MESOLIMBIC: Direct value-feedback path for full TD error
    # This connection delivers SNr spikes to VTA so that VTA.forward() can decode
    # V(s) = 1 − SNr_rate / 2·baseline (inverse coding: high SNr = low value).
    # VTA reads the raw spike tensor but does NOT integrate the GABA_A conductance —
    # value decoding is done by the explicit scan in forward().
    # Biology: SNr collaterals to VTA are documented (Tepper et al. 1995; Paladini &
    # Tepper 1999); this pathway provides a direct value estimate bypassing the
    # multi-synapse LHb→RMTg→VTA delay chain (~8–12 ms vs ~2 ms here).
    # Distance: ~0.5–1 cm (same midbrain region) → 1–2 ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="substantia_nigra",
            source_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            target_region="vta",
            target_population=VTAPopulation.DA_MESOLIMBIC,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=2.0,
        connectivity=0.4,
        weight_scale=0.00001,  # Very weak — spikes are decoded, not integrated as conductance
    )

    # LHb → RMTg: Aversive signal drives GABAergic pause mediator
    # Biology: Glutamatergic LHb→RMTg (heaviest known projection from LHb)
    # Distance: ~0.5-1cm (adjacent brainstem structures) → 1-3ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="lateral_habenula",
            source_population=LHbPopulation.PRINCIPAL,
            target_region="rostromedial_tegmentum",
            target_population=RMTgPopulation.GABA,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.6,
        weight_scale=0.0006,
    )

    # RMTg → VTA DA_MESOLIMBIC: GABAergic pause = negative RPE (mesolimbic channel)
    # RMTg inhibits all VTA DA neurons; route to each sub-population separately so the
    # two populations have independent synaptic weight matrices.
    # Fast GABA_A-mediated, short distance → 1-2ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="rostromedial_tegmentum",
            source_population=RMTgPopulation.GABA,
            target_region="vta",
            target_population=VTAPopulation.DA_MESOLIMBIC,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=2.0,
        connectivity=0.7,
        weight_scale=0.0005,
    )

    # RMTg → VTA DA_MESOCORTICAL: same pause mechanism for mesocortical sub-population.
    # Mesocortical DA neurons recover faster (higher baseline, no D2 autoreceptors).
    builder.connect(
        synapse_id=SynapseId(
            source_region="rostromedial_tegmentum",
            source_population=RMTgPopulation.GABA,
            target_region="vta",
            target_population=VTAPopulation.DA_MESOCORTICAL,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=2.0,
        connectivity=0.7,
        weight_scale=0.0005,
    )

    # =============================================================================
    # SPIKING NOREPINEPHRINE SYSTEM (Locus Coeruleus)
    # =============================================================================
    # Implements biologically-accurate arousal/uncertainty signaling through NE volume transmission.
    # LC computes uncertainty from PFC variance and hippocampal novelty, broadcasts NE spikes.
    # NE neurons: 1-3 Hz baseline, synchronized bursts (500ms), gap junction coupling.
    # Receptors: τ_rise=8ms, τ_decay=150ms (NET reuptake).

    # PFC → LC: Prefrontal variance signals uncertainty
    # High PFC activity variance → high LC firing → NE release
    # Executive population variance indicates decision uncertainty
    # Distance: ~3-5cm, well-myelinated → 5-8ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=10.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # Hippocampus → LC: Novelty detection drives arousal
    # CA1 output variance indicates contextual novelty
    # Novel contexts → high LC firing → memory encoding enhancement
    # Distance: ~4-6cm, moderately myelinated → 8-12ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        axonal_delay_std_ms=15.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # =============================================================================
    # SPIKING ACETYLCHOLINE SYSTEM (Nucleus Basalis)
    # =============================================================================
    # Implements biologically-accurate attention/encoding signaling through ACh volume transmission.
    # NB computes prediction error from PFC activity changes, broadcasts brief ACh bursts.
    # ACh neurons: 2-5 Hz baseline, brief bursts (50-100ms), fast SK adaptation.
    # Receptors: τ_rise=5ms, τ_decay=50ms (AChE degradation).

    # PFC → NB: Prefrontal activity changes signal prediction errors
    # High PFC activity rate-of-change → high NB firing → ACh release
    # Unexpected events drive encoding mode in cortex/hippocampus
    # Distance: ~3-5cm, well-myelinated → 5-8ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE,
            target_region="nucleus_basalis",
            target_population=NucleusBasalisPopulation.ACH,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=10.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # BLA PRINCIPAL → NB: Emotional salience drives ACh encoding-mode bursts
    # BLA principal neurons respond to unexpected or aversive stimuli (US, threat).
    # High BLA activity signals a salient, emotionally significant event →
    # strong prediction error → NB bursts ACh → cortex/hippocampus encode the event.
    # This is the direct basal-forebrain projection of the BLA described in
    # Zaborszky et al. (2015) and Holland & Gallagher (1999).
    # Distance: amygdala → basal forebrain ~2-4 cm, moderately myelinated → 5-8ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="nucleus_basalis",
            target_population=NucleusBasalisPopulation.ACH,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=9.0,
        connectivity=0.2,
        weight_scale=0.0003,
    )

    # =============================================================================
    # SPIKING SEROTONIN SYSTEM (Dorsal Raphe Nucleus)
    # =============================================================================
    # DRN 5-HT neurons: 2-4 Hz tonic pacemaking (I_h driven), suppressed by LHb.
    # Serotonin broadcasts patience/mood signals to striatum, PFC, hippocampus, and
    # BLA via the '5ht' neuromodulator hub (wired automatically from neuromodulator_outputs).
    # This section provides the inhibitory input that drives 5-HT pauses.

    # LHb → DRN SEROTONIN: Punishment / negative RPE → 5-HT pause
    # LHb principal (glutamatergic) projects heavily to DRN.
    # In DorsalRapheNucleus.forward() the LHb spike rate is converted to a negative
    # serotonin drive via `lhb_inhibition_gain`: high LHb activity → 5-HT pause.
    # Biology: Vertes & Linley (2008), Stern et al. (2017) — one of the strongest
    #          LHb output projections; monosynaptically suppresses 5-HT neurons
    #          via local GABA interneurons.
    # Distance: ~1-2cm (adjacent midbrain structures) → 2-4ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="lateral_habenula",
            source_population=LHbPopulation.PRINCIPAL,
            target_region="dorsal_raphe",
            target_population=DRNPopulation.SEROTONIN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.5,
        weight_scale=0.001,
    )

    # =============================================================================
    # AMYGDALA (BLA + CeA) CONNECTIONS
    # =============================================================================
    # BLA: fear conditioning / extinction (CS–US association)
    # CeA: fear output (→ LC arousal, LHb aversive RPE)

    # CorticalColumn → BLA PRINCIPAL: CS representation (slow, detailed pathway)
    # Auditory/somatosensory cortex provides the conditioned stimulus (CS) signal.
    # Distance: ~3-5cm, moderately myelinated → 6-10ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=12.0,
        connectivity=0.15,
        weight_scale=0.0003,
    )

    # Thalamus → BLA PRINCIPAL: Fast CS pathway (thalamo-amygdalar shortcut)
    # Direct thalamic relay bypasses cortex (~12ms faster than cortical path).
    # Enables rapid fear conditioning before full cortical elaboration of CS.
    # Distance: ~2-3cm → ~5ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=8.0,
        connectivity=0.2,
        weight_scale=0.0003,
    )

    # Hippocampus → BLA PRINCIPAL: Context signal for contextual fear / extinction renewal
    # CA1 encodes spatial/temporal context; gates fear recall based on place-memory.
    # Distance: ~1-2cm (directly adjacent structures) → 3-5ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.2,
        weight_scale=0.0003,
    )

    # PFC → BLA SOM: Top-down extinction regulation
    # Infralimbic PFC → BLA SOM interneurons inhibits principal neurons → extinction.
    # Distance: ~4-6cm, well-myelinated → 6-10ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal",
            source_population=PrefrontalPopulation.EXECUTIVE,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.SOM,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=12.0,
        connectivity=0.2,
        weight_scale=0.0002,
    )

    # BLA PRINCIPAL → CeA LATERAL: Core fear signal transmission
    # LA/BA principal neurons project to CeL, driving fear-ON cells.
    # Distance: ~0.5-1cm (within amygdaloid complex) → 2-3ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="central_amygdala",
            target_population=CeAPopulation.LATERAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=3.0,
        connectivity=0.3,
        weight_scale=0.0005,
    )

    # BLA PRINCIPAL → CeA MEDIAL: Direct strong projection (bypasses CeL gating)
    # Some BLA principal neurons project directly to CeM for rapid fear output.
    # Distance: ~0.5-1cm → 2-3ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="central_amygdala",
            target_population=CeAPopulation.MEDIAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.0,
        connectivity=0.2,
        weight_scale=0.0004,
    )

    # CeA MEDIAL → LC: Fear-driven NE arousal
    # CeM activates LC during fear, driving NE release and sympathetic arousal.
    # Distance: ~3-5cm (amygdala → pons) → 5-10ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="central_amygdala",
            source_population=CeAPopulation.MEDIAL,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=7.0,
        axonal_delay_std_ms=10.0,
        connectivity=0.3,
        weight_scale=0.0004,
    )

    # CeA MEDIAL → LHb: Aversive prediction error signal
    # CeM output encodes expected punishment; drives LHb for negative RPE.
    # LHb will then activate RMTg → DA pause in VTA.
    # Distance: ~3-4cm → 5-8ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="central_amygdala",
            source_population=CeAPopulation.MEDIAL,
            target_region="lateral_habenula",
            target_population=LHbPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=8.0,
        connectivity=0.3,
        weight_scale=0.0004,
    )


# Register built-in presets
BrainBuilder.register_preset(
    name="default",
    description="Default preset",
    builder_fn=_build_default,
)
