"""
Brain Builder - Fluent API for Brain Construction

This module provides a fluent, progressive API for building brain architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from thalia.brain.regions.population_names import (
    CerebellumPopulation,
    CortexPopulation,
    HippocampusPopulation,
    LocusCoeruleusPopulation,
    MedialSeptumPopulation,
    NucleusBasalisPopulation,
    PrefrontalPopulation,
    RewardEncoderPopulation,
    StriatumPopulation,
    SubstantiaNigraPopulation,
    ThalamusPopulation,
    VTAPopulation,
)
from thalia.typing import (
    PopulationName,
    PopulationSizes,
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
        name: Instance name (e.g., "my_cortex", "visual_input")
        registry_name: Region type in registry (e.g., "cortex")
        population_sizes: Population size specifications (e.g., {"l23": 500, "l5": 300})
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
        source: Source region name
        source_population: Output population on source (e.g., 'l23', 'l5', 'relay', 'ca1', 'd1', 'executive', 'dcn')
        target: Target region name
        target_population: Input population on target (e.g., 'feedforward', 'top_down')
        axonal_delay_ms: Axonal conduction delay in milliseconds (mean)
        axonal_delay_std_ms: Standard deviation for heterogeneous delays (0 = uniform)
        connectivity: Connection probability (fraction of connections present, 0-1)
        weight_scale: Initial weight scale (normalized conductance)
        instance: Instantiated axonal tract (set after build())
    """

    source: RegionName
    source_population: PopulationName
    target: RegionName
    target_population: PopulationName
    axonal_delay_ms: float
    axonal_delay_std_ms: float
    connectivity: float
    weight_scale: float
    is_inhibitory: bool = False  # True for GABAergic long-range projections (e.g., SNr→Thalamus)
    instance: Optional[AxonalTract] = None


@dataclass
class ExternalInputSpec:
    """Specification for an external input source.

    Attributes:
        source_name: Name of external source (e.g., "sensory", "motor_command")
        target: Target region name
        target_population: Target population on target (e.g., "relay")
        n_input: Number of input neurons from external source
        connectivity: Connection probability (0-1)
        weight_scale: Initial weight scale
    """

    source_population: PopulationName
    target: RegionName
    target_population: PopulationName
    n_input: int
    connectivity: float
    weight_scale: float


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
            name: Instance name (e.g., "my_cortex", "thalamus")
            registry_name: Region type in registry (e.g., "cortex", "thalamus")
            population_sizes: Population size specifications (e.g., {"l23": 500, "l5": 300})
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
        source: RegionName,
        source_population: PopulationName,
        target: RegionName,
        target_population: PopulationName,
        axonal_delay_ms: float,
        axonal_delay_std_ms: float,
        connectivity: float,
        weight_scale: float,
        is_inhibitory: bool = False,
    ) -> BrainBuilder:
        """Connect two regions with an axonal tract.

        Args:
            source: Source region name
            source_population: Output population on source (e.g., 'l23', 'l5', 'relay', 'ca1', 'd1', 'executive', 'dcn')
            target: Target region name
            target_population: Target population on target (e.g., 'l23', 'trn', 'dg', 'ca1', 'executive')
            axonal_delay_ms: Axonal conduction delay in milliseconds (mean)
            axonal_delay_std_ms: Standard deviation for heterogeneous delays (0 = uniform delay)
            connectivity: Connection probability (fraction of connections present, 0-1)
            weight_scale: Initial weight scale (normalized conductance)
            is_inhibitory: If True, creates a GABAergic (inhibitory) projection. The delayed spikes
                will be routed through the target region's inhibitory conductance channel (g_inh).
                Default False (glutamatergic/excitatory).

        Returns:
            Self for method chaining

        Raises:
            ValueError: If source or target region doesn't exist
        """
        if source not in self._region_specs:
            raise ValueError(f"Source region '{source}' not found")
        if target not in self._region_specs:
            raise ValueError(f"Target region '{target}' not found")
        if weight_scale < 0:
            raise ValueError("Weight scale must be non-negative")

        spec = ConnectionSpec(
            source=source,
            source_population=source_population,
            target=target,
            target_population=target_population,
            axonal_delay_ms=axonal_delay_ms,
            axonal_delay_std_ms=axonal_delay_std_ms,
            connectivity=connectivity,
            weight_scale=weight_scale,
            is_inhibitory=is_inhibitory,
        )

        self._connection_specs.append(spec)
        return self

    def add_external_input(
        self,
        source_population: PopulationName,
        target: RegionName,
        target_population: PopulationName,
        n_input: int,
        connectivity: float,
        weight_scale: float,
    ) -> BrainBuilder:
        """Register an external input source (e.g., sensory input).

        External inputs are not brain regions - they're provided by the training loop
        via the `brain_inputs` dict passed to `brain.forward()`.

        This method registers the source with the target region and creates
        synaptic weights, just like connect() but without creating an AxonalTract.

        Args:
            source_population: Name of external source population (e.g., "sensory", "motor_command")
            target: Target region name
            target_population: Target population on target (e.g., "relay")
            n_input: Number of input neurons from external source
            connectivity: Connection probability (0-1)
            weight_scale: Initial weight scale

        Returns:
            Self for method chaining

        Raises:
            ValueError: If target region doesn't exist
        """
        if target not in self._region_specs:
            raise ValueError(f"Target region '{target}' not found")
        if weight_scale < 0:
            raise ValueError("Weight scale must be non-negative")

        spec = ExternalInputSpec(
            source_population=source_population,
            target=target,
            target_population=target_population,
            n_input=n_input,
            connectivity=connectivity,
            weight_scale=weight_scale,
        )

        self._external_input_specs.append(spec)
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
            connected_regions.add(conn.source)
            connected_regions.add(conn.target)

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
        source_region = regions[conn_spec.source]
        target_region = regions[conn_spec.target]

        source_size = source_region.get_population_size(conn_spec.source_population)

        synapse_id = SynapseId(
            source_region=conn_spec.source,
            source_population=conn_spec.source_population,
            target_region=conn_spec.target,
            target_population=conn_spec.target_population,
            is_inhibitory=conn_spec.is_inhibitory,
        )

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
            # TODO: Add STP config support to ConnectionSpec if needed.
            stp_config=None,
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
            target_region = regions[ext_spec.target]
            target_region.add_input_source(
                synapse_id=SynapseId(
                    source_region="external",
                    source_population=ext_spec.source_population,
                    target_region=ext_spec.target,
                    target_population=ext_spec.target_population,
                    # TODO: Assuming external inputs are excitatory by default; can be extended to support inhibitory if needed
                    is_inhibitory=False,
                ),
                n_input=ext_spec.n_input,
                connectivity=ext_spec.connectivity,
                weight_scale=ext_spec.weight_scale,
                # TODO: External inputs can also have STP if desired - add stp_config to ExternalInputSpec and pass it here
                stp_config=None,
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
        CerebellumPopulation.GRANULE.value: 20000,  # Granule:Purkinje = 200:1 (biology: 1000:1, most numerous neurons in brain)
        CerebellumPopulation.PURKINJE.value: 100,   # Purkinje cells are sole output of cerebellar cortex, provide strong inhibition to DCN
        CerebellumPopulation.DCN.value: 100,        # DCN:Purkinje = 1:1 (biology ~1:1, DCN are sole cerebellar output neurons
    }
    default_cortex_sizes: PopulationSizes = {
        CortexPopulation.L23_PYR.value: 1200,  # 40% (supragranular, associative)
        CortexPopulation.L4_PYR.value: 300,    # 10% (granular, thalamic input)
        CortexPopulation.L5_PYR.value: 450,    # 15% (output to subcortex)
        CortexPopulation.L6A_PYR.value: 150,   # 5% (corticothalamic type I)
        CortexPopulation.L6B_PYR.value: 900,   # 30% (corticothalamic type II)
    }
    default_hippocampus_sizes: PopulationSizes = {
        HippocampusPopulation.DG.value: 500,   # cortex:hippocampus biological ratio (400:1)
        HippocampusPopulation.CA3.value: 250,  # Autoassociative recurrent memory
        HippocampusPopulation.CA2.value: 75,   # Small transitional zone (CA2 is tiny in biology)
        HippocampusPopulation.CA1.value: 375,  # Output to cortex (slightly larger than CA3)
    }
    default_prefrontal_sizes: PopulationSizes = {
        PrefrontalPopulation.EXECUTIVE.value: 800,
    }
    default_striatum_sizes: PopulationSizes = {
        StriatumPopulation.D1.value: 200,
        StriatumPopulation.D2.value: 200,
        'n_actions': 10,
        'neurons_per_action': 10,
    }
    default_thalamus_sizes: PopulationSizes = {
        ThalamusPopulation.RELAY.value: 400,  # Thalamic relay neurons (input from sensory pathways, output to cortex)
        ThalamusPopulation.TRN.value: 40,     # 10:1 relay:TRN ratio
    }

    # Medial septum: Theta pacemaker for hippocampal circuits
    # Small subcortical region with cholinergic and GABAergic pacemaker neurons
    # Generates intrinsic ~8 Hz bursting that phase-locks hippocampal OLM cells
    default_medial_septum_sizes: PopulationSizes = {
        MedialSeptumPopulation.ACH.value: 200,
        MedialSeptumPopulation.GABA.value: 200,
    }

    # Spiking dopamine system: RewardEncoder, SNr, VTA
    # Implements biologically-accurate dopamine release with burst/pause dynamics
    # Replaces scalar dopamine broadcast with spike-based volume transmission
    default_reward_encoder_sizes: PopulationSizes = {
        RewardEncoderPopulation.REWARD_SIGNAL.value: 100,
    }
    default_substantia_nigra_sizes: PopulationSizes = {
        SubstantiaNigraPopulation.VTA_FEEDBACK.value: 1000,  # SNr receives dense feedback from VTA dopamine neurons (inhibitory)
    }
    default_locus_coeruleus_sizes: PopulationSizes = {
        LocusCoeruleusPopulation.NE.value: 1600,
        LocusCoeruleusPopulation.GABA.value: 300,
    }
    default_nucleus_basalis_sizes: PopulationSizes = {
        NucleusBasalisPopulation.ACH.value: 3000,
        NucleusBasalisPopulation.GABA.value: 500,
    }
    default_vta_sizes: PopulationSizes = {
        VTAPopulation.DA.value: 2500,
        VTAPopulation.GABA.value: 1000,  # 40% of DA population for local inhibitory control
    }

    # Merge with overrides (user overrides take precedence)
    sizes_overrides: Dict[RegionName, PopulationSizes] = overrides.get('population_sizes', {})

    cerebellum_sizes: PopulationSizes = {**default_cerebellum_sizes, **sizes_overrides.get('cerebellum', {})}
    cortex_sizes: PopulationSizes = {**default_cortex_sizes, **sizes_overrides.get('cortex', {})}
    hippocampus_sizes: PopulationSizes = {**default_hippocampus_sizes, **sizes_overrides.get('hippocampus', {})}
    prefrontal_sizes: PopulationSizes = {**default_prefrontal_sizes, **sizes_overrides.get('prefrontal', {})}
    striatum_sizes: PopulationSizes = {**default_striatum_sizes, **sizes_overrides.get('striatum', {})}
    thalamus_sizes: PopulationSizes = {**default_thalamus_sizes, **sizes_overrides.get('thalamus', {})}
    medial_septum_sizes: PopulationSizes = {**default_medial_septum_sizes, **sizes_overrides.get('medial_septum', {})}
    reward_encoder_sizes: PopulationSizes = {**default_reward_encoder_sizes, **sizes_overrides.get('reward_encoder', {})}
    substantia_nigra_sizes: PopulationSizes = {**default_substantia_nigra_sizes, **sizes_overrides.get('substantia_nigra', {})}
    locus_coeruleus_sizes: PopulationSizes = {**default_locus_coeruleus_sizes, **sizes_overrides.get('locus_coeruleus', {})}
    nucleus_basalis_sizes: PopulationSizes = {**default_nucleus_basalis_sizes, **sizes_overrides.get('nucleus_basalis', {})}
    vta_sizes: PopulationSizes = {**default_vta_sizes, **sizes_overrides.get('vta', {})}

    external_sensory_size: Optional[int] = sizes_overrides.get('external', {}).get('sensory', None)
    if external_sensory_size is None:
        external_sensory_size = thalamus_sizes[ThalamusPopulation.RELAY.value]

    # =============================================================================
    # Define regions with biologically realistic population sizes
    # =============================================================================

    builder.add_region("cerebellum", "cerebellum", population_sizes=cerebellum_sizes)
    builder.add_region("cortex", "cortex", population_sizes=cortex_sizes)
    builder.add_region("hippocampus", "hippocampus", population_sizes=hippocampus_sizes)
    builder.add_region("locus_coeruleus", "locus_coeruleus", population_sizes=locus_coeruleus_sizes)
    builder.add_region("medial_septum", "medial_septum", population_sizes=medial_septum_sizes)
    builder.add_region("nucleus_basalis", "nucleus_basalis", population_sizes=nucleus_basalis_sizes)
    builder.add_region("prefrontal", "prefrontal", population_sizes=prefrontal_sizes)
    builder.add_region("reward_encoder", "reward_encoder", population_sizes=reward_encoder_sizes)
    builder.add_region("striatum", "striatum", population_sizes=striatum_sizes)
    builder.add_region("substantia_nigra", "substantia_nigra", population_sizes=substantia_nigra_sizes)
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
    builder.add_external_input(
        source_population="sensory",
        target="thalamus",
        target_population=ThalamusPopulation.RELAY.value,
        n_input=external_sensory_size,
        connectivity=0.25,
        weight_scale=0.0005,
    )

    # Thalamus → Cortex L4: Thalamocortical projection with feedforward inhibition
    # Fast, heavily myelinated pathway
    # Distance: ~2-3cm, conduction velocity: ~10-20 m/s → 2-3ms delay
    # Uses 'relay' population: Only relay neurons project to cortex (TRN provides lateral inhibition)
    #
    # Biology: Thalamic afferents synapse onto BOTH pyramidal and PV interneurons
    # PV cells have lower thresholds (0.8 vs 1.4), so they receive weaker weights
    # This creates fast feedforward inhibition: PV fires before pyramidal

    # Thalamus → L4 Pyramidal: Main thalamocortical drive
    builder.connect(
        source="thalamus",
        source_population=ThalamusPopulation.RELAY.value,
        target="cortex",
        target_population=CortexPopulation.L4_PYR.value,
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=5.0,
        connectivity=0.7,
        weight_scale=0.00008,
    )

    # Thalamus → L4 PV: Feedforward inhibition drive
    # Biology: Same axons branch to PV cells for fast inhibition
    # Weaker weights compensate for PV's lower threshold (0.8 vs 1.4)
    builder.connect(
        source="thalamus",
        source_population=ThalamusPopulation.RELAY.value,
        target="cortex",
        target_population=CortexPopulation.L4_INHIBITORY_PV.value,
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=5.0,
        connectivity=0.7,
        weight_scale=0.00002,
    )

    # Cortex L6a/L6b → Thalamus: Dual corticothalamic feedback pathways
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
        source="cortex",
        source_population=CortexPopulation.L6A_PYR.value,
        target="thalamus",
        target_population=ThalamusPopulation.TRN.value,
        axonal_delay_ms=10.0,
        axonal_delay_std_ms=15.0,
        connectivity=0.3,
        weight_scale=0.001,
    )
    builder.connect(
        source="cortex",
        source_population=CortexPopulation.L6B_PYR.value,
        target="thalamus",
        target_population=ThalamusPopulation.RELAY.value,
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=0.0001,
    )

    # Cortex ⇄ Hippocampus: Bidirectional memory integration
    # Entorhinal cortex ↔ hippocampus: moderately myelinated
    # Distance: ~3-5cm, conduction velocity: ~5-10 m/s → 5-8ms delay
    builder.connect(
        source="cortex",
        source_population=CortexPopulation.L5_PYR.value,
        target="hippocampus",
        target_population=HippocampusPopulation.DG.value,
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=10.0,
        connectivity=0.3,
        weight_scale=0.0005,
    )
    builder.connect(
        source="hippocampus",
        source_population=HippocampusPopulation.CA1.value,
        target="cortex",
        target_population=CortexPopulation.L5_PYR.value,
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=10.0,
        connectivity=0.3,
        weight_scale=0.0001,
    )

    # Cortex → PFC: Executive control pathway
    # Corticocortical long-range connections
    # Distance: ~5-10cm, conduction velocity: ~3-8 m/s → 10-15ms delay
    builder.connect(
        source="cortex",
        source_population=CortexPopulation.L23_PYR.value,
        target="prefrontal",
        target_population=PrefrontalPopulation.EXECUTIVE.value,
        axonal_delay_ms=12.5,
        axonal_delay_std_ms=20.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # Multi-source → Striatum: Corticostriatal + hippocampostriatal + PFC inputs
    # Per-target delays model different myelination patterns (Gerfen & Surmeier 2011):
    # - Cortex → Striatum: Fast, heavily myelinated, short distance (~2-4cm) → 3-5ms
    # - Hippocampus → Striatum: Moderate, longer distance (~4-6cm) → 7-10ms
    # - PFC → Striatum: Variable, longest distance (~6-10cm) → 12-18ms
    builder.connect(
        source="cortex",
        source_population=CortexPopulation.L5_PYR.value,
        target="striatum",
        target_population=StriatumPopulation.D1.value,
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.3,
        weight_scale=0.0001,
    )
    builder.connect(
        source="hippocampus",
        source_population=HippocampusPopulation.CA1.value,
        target="striatum",
        target_population=StriatumPopulation.D1.value,
        axonal_delay_ms=8.5,
        axonal_delay_std_ms=13.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )
    builder.connect(
        source="prefrontal",
        source_population=PrefrontalPopulation.EXECUTIVE.value,
        target="striatum",
        target_population=StriatumPopulation.D1.value,
        axonal_delay_ms=15.0,
        axonal_delay_std_ms=22.5,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # Striatum → PFC: Basal ganglia gating of working memory
    # Via thalamus (MD/VA nuclei), total distance ~8-12cm (Haber 2003)
    # Includes striatum→thalamus→PFC relay → 15-20ms total delay
    builder.connect(
        source="striatum",
        source_population=StriatumPopulation.D1.value,
        target="prefrontal",
        target_population=PrefrontalPopulation.EXECUTIVE.value,
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
        source="cortex",
        source_population=CortexPopulation.L5_PYR.value,
        target="cerebellum",
        target_population=CerebellumPopulation.GRANULE.value,
        axonal_delay_ms=25.0,
        axonal_delay_std_ms=37.5,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # PFC → Cerebellum: similar pathway length
    # Goal/context input
    builder.connect(
        source="prefrontal",
        source_population=PrefrontalPopulation.EXECUTIVE.value,
        target="cerebellum",
        target_population=CerebellumPopulation.GRANULE.value,
        axonal_delay_ms=25.0,
        axonal_delay_std_ms=37.5,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # Cerebellum → Cortex L4: via thalamus (VL/VA nuclei), moderately fast
    # Distance: ~8-12cm, includes thalamic relay → 15-20ms delay
    # Forward model predictions
    # Split into pyramidal and PV for feedforward inhibition (same as thalamus)

    # Cerebellum → L4 Pyramidal: Motor predictions
    builder.connect(
        source="cerebellum",
        source_population=CerebellumPopulation.DCN.value,
        target="cortex",
        target_population=CortexPopulation.L4_PYR.value,
        axonal_delay_ms=17.5,
        axonal_delay_std_ms=26.0,
        connectivity=0.3,
        weight_scale=0.0008,
    )

    # Cerebellum → L4 PV: Feedforward inhibition for motor predictions
    builder.connect(
        source="cerebellum",
        source_population=CerebellumPopulation.DCN.value,
        target="cortex",
        target_population=CortexPopulation.L4_INHIBITORY_PV.value,
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
        source="prefrontal",
        source_population=PrefrontalPopulation.EXECUTIVE.value,
        target="hippocampus",
        target_population=HippocampusPopulation.CA1.value,
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
        source="hippocampus",
        source_population=HippocampusPopulation.CA1.value,
        target="prefrontal",
        target_population=PrefrontalPopulation.EXECUTIVE.value,
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=18.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # PFC → Cortex: Top-down attention and cognitive control
    # PFC biases cortical processing via direct feedback projections to L2/3
    # Corticocortical feedback: ~5-8cm, conduction velocity: ~3-6 m/s → 10-15ms
    # Biological accuracy: PFC feedback targets superficial layers (L2/3), NOT L4
    # This bypasses thalamic input and directly modulates cortical representations
    # Enables: attentional bias, cognitive control over perception, working memory maintenance
    builder.connect(
        source="prefrontal",
        source_population=PrefrontalPopulation.EXECUTIVE.value,
        target="cortex",
        target_population=CortexPopulation.L23_PYR.value,
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
        source="thalamus",
        source_population=ThalamusPopulation.RELAY.value,
        target="hippocampus",
        target_population=HippocampusPopulation.DG.value,
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
    builder.connect(
        source="thalamus",
        source_population=ThalamusPopulation.RELAY.value,
        target="striatum",
        target_population=StriatumPopulation.D1.value,
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=0.0002,
    )

    # Medial Septum → Hippocampus CA3: Septal theta drive for emergent oscillations
    # GABAergic pacemaker neurons phase-lock hippocampal OLM interneurons
    # OLM dendritic inhibition creates emergent encoding/retrieval separation
    # Distance: ~1-2cm (local subcortical), well-myelinated → 2ms delay
    # CRITICAL: This connection enables emergent theta (replaces hardcoded sinusoid)
    # STRENGTHENED: 0.03 → 0.09 to propagate 8 Hz rhythm and break 1 Hz delta lock
    builder.connect(
        source="medial_septum",
        source_population=MedialSeptumPopulation.GABA.value,
        target="hippocampus",
        target_population=HippocampusPopulation.CA3.value,
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
        source="hippocampus",
        source_population=HippocampusPopulation.CA1.value,
        target="medial_septum",
        target_population=MedialSeptumPopulation.GABA.value,
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.2,
        weight_scale=0.0005,
    )

    # Medial Septum → Hippocampus CA1: Cholinergic modulation
    # Biology: Septum ACh neurons enhance CA1 plasticity and attention
    # Complements GABAergic theta drive to CA3 with cholinergic modulation to CA1
    # ACh phase-locked to theta peaks (encoding mode), modulates LTP/LTD
    # Distance: ~1-2cm, similar pathway → 2ms delay
    # STRENGTHENED: 0.008 → 0.032 to enhance theta entrainment in CA1
    builder.connect(
        source="medial_septum",
        source_population=MedialSeptumPopulation.ACH.value,
        target="hippocampus",
        target_population=HippocampusPopulation.CA1.value,
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.15,
        weight_scale=0.0003,
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
        source="striatum",
        source_population=StriatumPopulation.D1.value,
        target="substantia_nigra",
        target_population=SubstantiaNigraPopulation.VTA_FEEDBACK.value,
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=7.5,
        connectivity=0.6,
        weight_scale=0.0008,
    )
    builder.connect(
        source="striatum",
        source_population=StriatumPopulation.D2.value,
        target="substantia_nigra",
        target_population=SubstantiaNigraPopulation.VTA_FEEDBACK.value,
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=7.5,
        connectivity=0.6,
        weight_scale=0.0001,
    )

    # SNr → VTA: Value estimate feedback for TD learning
    # SNr provides inhibitory feedback encoding value: V(s) ∝ 1/firing_rate
    # High SNr firing (low D1) → high inhibition → low value signal to VTA
    # Low SNr firing (high D1) → low inhibition → high value signal to VTA
    # Distance: ~0.5-1cm (adjacent midbrain nuclei) → 1.5ms mean delay
    builder.connect(
        source="substantia_nigra",
        source_population=SubstantiaNigraPopulation.VTA_FEEDBACK.value,
        target="vta",
        target_population=VTAPopulation.DA.value,
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=4.5,
        connectivity=0.6,
        weight_scale=0.0001,
    )

    # RewardEncoder → VTA: External reward signal delivery
    # Population-coded reward spikes from environment/task
    # VTA decodes to scalar reward: r = mean(spikes) / max_rate
    # Note: RewardEncoder receives external input via Brain.forward()
    # Distance: Conceptual (external input pathway) → minimal delay
    builder.connect(
        source="reward_encoder",
        source_population=RewardEncoderPopulation.REWARD_SIGNAL.value,
        target="vta",
        target_population=VTAPopulation.DA.value,
        axonal_delay_ms=0.5,
        axonal_delay_std_ms=1.0,
        connectivity=0.7,
        weight_scale=0.0008,
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
        source="prefrontal",
        source_population=PrefrontalPopulation.EXECUTIVE.value,
        target="locus_coeruleus",
        target_population=LocusCoeruleusPopulation.NE.value,
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
        source="hippocampus",
        source_population=HippocampusPopulation.CA1.value,
        target="locus_coeruleus",
        target_population=LocusCoeruleusPopulation.NE.value,
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
        source="prefrontal",
        source_population=PrefrontalPopulation.EXECUTIVE.value,
        target="nucleus_basalis",
        target_population=NucleusBasalisPopulation.ACH.value,
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=10.0,
        connectivity=0.3,
        weight_scale=0.0002,
    )


# Register built-in presets
BrainBuilder.register_preset(
    name="default",
    description=(
        "Default biologically realistic brain architecture with 12 regions, "
        "population sizes based on primate data, and axonal delays reflecting "
        "conduction velocities and distances. Includes spiking dopamine, norepinephrine, "
        "and acetylcholine systems for neuromodulation. Recurrent connections are "
        "externalized for biological accuracy. Suitable for modeling a wide range of "
        "cognitive tasks with realistic neural dynamics."
    ),
    builder_fn=_build_default,
)
