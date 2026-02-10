"""
Brain Builder - Fluent API for Brain Construction

This module provides a fluent, progressive API for building brain architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from thalia.typing import (
    SpikesSourceKey,
    PopulationName,
    PopulationSizes,
    RegionName,
    compound_key,
)

from .axonal_projection import AxonalProjection, AxonalProjectionSourceSpec
from .brain import DynamicBrain
from .configs import BrainConfig, NeuralRegionConfig
from .regions import NeuralRegionRegistry, NeuralRegion


@dataclass
class RegionSpec:
    """Specification for a brain region.

    Attributes:
        name: Instance name (e.g., "my_cortex", "visual_input")
        registry_name: Region type in registry (e.g., "cortex")
        population_sizes: Semantic input sizes (e.g., relay_size, layer_sizes)
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
        target: Target region name
        source_population: Output population on source (e.g., 'l23', 'l5', 'relay', 'ca1', 'd1', 'executive', 'prediction')
        target_population: Input population on target (e.g., 'feedforward', 'top_down')
        config_params: Pathway configuration parameters
        instance: Instantiated pathway (set after build())
    """

    source: RegionName
    target: RegionName
    source_population: PopulationName
    target_population: PopulationName
    axonal_delay_ms: float
    instance: Optional[AxonalProjection] = None

    def compound_key(self) -> SpikesSourceKey:
        """Get compound key for this source (region:population)."""
        return compound_key(self.source, self.source_population)


class BrainBuilder:
    """Fluent API for progressive brain construction.

    Supports:
        - Incremental region addition via method chaining
        - Connection definition with automatic pathway creation
        - Preset architectures for common use cases
        - Validation before building
        - Save/load graph specifications to JSON
    """

    # Registry of preset architectures
    _presets: Dict[str, PresetArchitecture] = {}

    def __init__(self, brain_config: BrainConfig):
        """Initialize builder with brain configuration.

        Args:
            brain_config: Brain configuration (device, dt_ms, oscillators, etc.)
        """
        # DISABLE GRADIENTS
        # Thalia uses local learning rules (STDP, BCM, Hebbian, three-factor)
        # that do NOT require backpropagation. Disabling gradients provides:
        # - Performance boost (no autograd overhead)
        # - Memory savings (no gradient storage)
        # - Biological plausibility (no non-local error signals)
        torch.set_grad_enabled(False)

        self.brain_config = brain_config
        self._region_specs: Dict[RegionName, RegionSpec] = {}
        self._connection_specs: List[ConnectionSpec] = []

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
            population_sizes: Layer size specifications (e.g., {"l23_size": 500, "l5_size": 300})
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

        # Create region spec
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
        target: RegionName,
        source_population: PopulationName,
        target_population: PopulationName,
        axonal_delay_ms: float,
    ) -> BrainBuilder:
        """Connect two regions with a pathway.

        Args:
            source: Source region name
            target: Target region name
            source_population: Output population on source (e.g., 'l23', 'l5', 'relay', 'ca1', 'd1', 'executive', 'prediction')
            target_population: Target population on target (e.g., 'l23', 'trn', 'dg', 'ca1', 'executive')
            axonal_delay_ms: Axonal delay in milliseconds

        Returns:
            Self for method chaining

        Raises:
            ValueError: If source or target region doesn't exist, or if populations not specified
        """
        # Validate regions exist
        if source not in self._region_specs:
            raise ValueError(f"Source region '{source}' not found")
        if target not in self._region_specs:
            raise ValueError(f"Target region '{target}' not found")

        # Create connection spec
        spec = ConnectionSpec(
            source=source,
            target=target,
            source_population=source_population,
            target_population=target_population,
            axonal_delay_ms=axonal_delay_ms,
        )

        self._connection_specs.append(spec)
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

    def _get_pathway_source_size(
        self,
        source_region: NeuralRegion[NeuralRegionConfig],
        source_population: PopulationName,
    ) -> int:
        """Get output size for pathway from source region and population.

        Args:
            source_region: Source region instance
            source_population: Output population specification

        Returns:
            Output size for pathway
        """
        size_attr = source_region.OUTPUT_POPULATIONS.get(source_population, None)
        if size_attr is not None:
            size = getattr(source_region, size_attr, None)
            if size is not None and isinstance(size, int):
                return size
            else:
                raise ValueError(
                    f"Region '{type(source_region).__name__}' has OUTPUT_POPULATIONS entry for population '{source_population}' "
                    f"but size attribute '{size_attr}' is missing or not an int"
                )
        else:
            raise ValueError(
                f"Source population '{source_population}' not found in OUTPUT_POPULATIONS of region type '{type(source_region).__name__}'"
            )

    def _create_axonal_projection(
        self,
        target_specs: List[ConnectionSpec],
        regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]],
    ) -> AxonalProjection:
        """Create AxonalProjection from connection specs with per-target delay support.

        AxonalProjection has different initialization than standard pathways:
        - Takes list of (region_name, population, size, delay_ms[, target_delays]) tuples
        - Handles multi-source concatenation internally
        - Supports per-target delay variation for realistic axonal branching

        Args:
            target_specs: List of ConnectionSpec for this target
            regions: Dict of instantiated regions

        Returns:
            AxonalProjection instance with target-specific delays
        """
        assert len(target_specs) > 0, "At least one ConnectionSpec is required to create an AxonalProjection"

        # Build sources list: [(region_name, population, size, delay_ms[, target_delays]), ...]
        # and register input sources with target region for size inference and routing
        sources: List[AxonalProjectionSourceSpec] = []
        target_population = target_specs[0].target_population  # All specs in this group share the same target layer
        for spec in target_specs:
            source_size = self._get_pathway_source_size(regions[spec.source], spec.source_population)

            sources.append(AxonalProjectionSourceSpec(
                region_name=spec.source,
                population=spec.source_population,
                size=source_size,
                delay_ms=spec.axonal_delay_ms,
            ))

            # Register this source with target region
            target_region = regions[spec.target]
            target_region.add_input_source(
                source_name=spec.compound_key(),
                target_population=target_population,
                n_input=source_size,
                sparsity=0.5, # Moderate sparsity for axonal projections
                weight_scale=6.0,  # Strong initial weights to ensure effective signal propagation through long-range axonal pathways
            )

        projection = AxonalProjection(
            sources=sources,
            dt_ms=self.brain_config.dt_ms,
            device=self.brain_config.device,
        )

        return projection

    def build(self) -> DynamicBrain:
        """Build DynamicBrain from specifications.

        Steps:
            1. Infer region input sizes from connections
            2. Validate graph
            3. Instantiate all regions from registry
            4. Instantiate all pathways from registry
            5. Create DynamicBrain with graph

        Returns:
            Constructed DynamicBrain instance

        Raises:
            ValueError: If validation fails or size inference fails
        """
        # Validate before building
        issues = self.validate()
        errors = [msg for msg in issues if msg.startswith("Error:")]
        if errors:
            raise ValueError("Validation failed:\n" + "\n".join(errors))

        # Instantiate regions
        regions: Dict[RegionName, NeuralRegion[NeuralRegionConfig]] = {}
        for name, spec in self._region_specs.items():
            # Get config class from registry
            config_class = NeuralRegionRegistry.get_config_class(spec.registry_name)

            if config_class is None:
                # Region missing config class metadata in registry
                raise ValueError(
                    f"Region '{spec.registry_name}' has no config_class registered. "
                    f"Update registry with config_class metadata."
                )

            # Use provided config or default
            config = spec.config if spec.config is not None else config_class()

            # Set global brain config parameters
            config.device = self.brain_config.device
            config.seed = self.brain_config.seed
            config.dt_ms = self.brain_config.dt_ms

            # Create region from registry
            region = NeuralRegionRegistry.create(
                spec.registry_name,
                config=config,
                population_sizes=spec.population_sizes
            )

            regions[name] = region
            spec.instance = region

        # === MULTI-SOURCE PATHWAY CONSTRUCTION ===
        # Instantiate pathways - GROUP BY (TARGET, TARGET_POPULATION) for multi-source pathways
        # This allows multiple independent pathways to the same target (e.g., L6a and L6b to thalamus)
        connections: Dict[Tuple[RegionName, SpikesSourceKey], AxonalProjection] = {}

        # Group connections by (target, target_population) to create multi-source pathways
        # Key is (target_name, target_population) so L6a and L6b are separate groups
        connections_by_target_layer: Dict[Tuple[RegionName, PopulationName], List[ConnectionSpec]] = {}
        for conn_spec in self._connection_specs:
            group_key = (conn_spec.target, conn_spec.target_population)
            if group_key not in connections_by_target_layer:
                connections_by_target_layer[group_key] = []
            connections_by_target_layer[group_key].append(conn_spec)

        # Create one pathway per (target, target_population) group (multi-source if multiple inputs)
        for (target_name, target_population), target_specs in connections_by_target_layer.items():
            pathway: AxonalProjection = self._create_axonal_projection(target_specs, regions)
            conn_key: Tuple[RegionName, SpikesSourceKey] = (target_specs[0].source, f"{target_name}:{target_population}")
            connections[conn_key] = pathway
            for conn_spec in target_specs:
                conn_spec.instance = pathway

        # Create DynamicBrain
        brain = DynamicBrain(
            config=self.brain_config,
            regions=regions,
            connections=connections,
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
    def list_presets(cls) -> List[Tuple[str, str]]:
        """List available preset architectures.

        Returns:
            List of (name, description) tuples
        """
        return [(name, preset.description) for name, preset in cls._presets.items()]

    @classmethod
    def preset_builder(cls, name: str, brain_config: BrainConfig, **overrides: Any) -> BrainBuilder:
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
    def preset(cls, name: str, brain_config: BrainConfig, **overrides: Any) -> DynamicBrain:
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
    """Default preset architecture with biologically realistic regions, layers, and connections."""
    # =============================================================================
    # Define biologically realistic layer sizes and neuron counts based on literature
    # TODO: Default sizes (can be overridden)
    # =============================================================================

    cerebellum_sizes = {
        'purkinje_size': 200,
        'granule_size': 800,
    }
    cortex_sizes = {
        'l23_size': 600,
        'l4_size': 300,
        'l5_size': 300,
        'l6a_size': 40,
        'l6b_size': 200,
        'vta_da': 1000,
    }
    hippocampus_sizes = {
        'dg_size': 4000,
        'ca3_size': 2000,
        'ca2_size': 1000,
        'ca1_size': 2000,
        'vta_da': 1000,
    }
    pfc_sizes = {
        'executive_size': 600,
        'vta_da': 1000,
    }
    striatum_sizes = {
        'd1_size': 100,
        'd2_size': 100,
        'n_actions': 10,
        'neurons_per_action': 10,
        'vta_da': 1000,
    }
    thalamus_sizes = {
        'relay_size': 200,
        'trn_size': 60,
    }

    # Medial septum: Theta pacemaker for hippocampal circuits
    # Small subcortical region with cholinergic and GABAergic pacemaker neurons
    # Generates intrinsic ~8 Hz bursting that phase-locks hippocampal OLM cells
    medial_septum_sizes = {
        "n_ach": 100,
        "n_gaba": 100,
    }

    # Spiking dopamine system: RewardEncoder, SNr, VTA
    # Implements biologically-accurate dopamine release with burst/pause dynamics
    # Replaces scalar dopamine broadcast with spike-based volume transmission
    reward_encoder_sizes = {
        "n_neurons": 100,
    }
    snr_sizes = {
        "n_neurons": 1000,
        "d1_input": 100,
        "d2_input": 100,
    }
    vta_sizes = {
        "da_neurons": 1000,
        "gaba_neurons": 400,
        "snr_value": 1000,
        "reward_signal": 100,
    }

    # =============================================================================
    # Define regions with biologically realistic layer sizes and neuron counts
    # =============================================================================

    builder.add_region("thalamus", "thalamus", population_sizes=thalamus_sizes)
    builder.add_region("cortex", "cortex", population_sizes=cortex_sizes)
    builder.add_region("hippocampus", "hippocampus", population_sizes=hippocampus_sizes)
    builder.add_region("pfc", "prefrontal", population_sizes=pfc_sizes)
    builder.add_region("striatum", "striatum", population_sizes=striatum_sizes)
    builder.add_region("cerebellum", "cerebellum", population_sizes=cerebellum_sizes)
    builder.add_region("medial_septum", "medial_septum", population_sizes=medial_septum_sizes)
    builder.add_region("reward_encoder", "reward_encoder", population_sizes=reward_encoder_sizes)
    builder.add_region("snr", "snr", population_sizes=snr_sizes)
    builder.add_region("vta", "vta", population_sizes=vta_sizes)

    # =============================================================================
    # Define connections with biologically realistic axonal delays
    # =============================================================================

    # Thalamus → Cortex: Thalamocortical projection
    # Fast, heavily myelinated pathway (Jones 2007, Sherman & Guillery 2006)
    # Distance: ~2-3cm, conduction velocity: ~10-20 m/s → 2-3ms delay
    # Uses 'relay' population: Only relay neurons project to cortex (TRN provides lateral inhibition)
    builder.connect(
        source="thalamus",
        target="cortex",
        source_population="relay",
        target_population="l4",
        axonal_delay_ms=2.5,
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
        target="thalamus",
        source_population="l6a",
        target_population="trn",
        axonal_delay_ms=10.0,
    )
    builder.connect(
        source="cortex",
        target="thalamus",
        source_population="l6b",
        target_population="relay",
        axonal_delay_ms=5.0,
    )

    # Cortex ⇄ Hippocampus: Bidirectional memory integration
    # Entorhinal cortex ↔ hippocampus: moderately myelinated
    # Distance: ~3-5cm, conduction velocity: ~5-10 m/s → 5-8ms delay
    builder.connect(
        source="cortex",
        target="hippocampus",
        source_population="l5",
        target_population="dg",
        axonal_delay_ms=6.5,
    )
    builder.connect(
        source="hippocampus",
        target="cortex",
        source_population="ca1",
        target_population="l5",
        axonal_delay_ms=6.5,
    )

    # Cortex → PFC: Executive control pathway
    # Corticocortical long-range connections
    # Distance: ~5-10cm, conduction velocity: ~3-8 m/s → 10-15ms delay
    builder.connect(
        source="cortex",
        target="pfc",
        source_population="l23",
        target_population="executive",
        axonal_delay_ms=12.5,
    )

    # Multi-source → Striatum: Corticostriatal + hippocampostriatal + PFC inputs
    # Per-target delays model different myelination patterns (Gerfen & Surmeier 2011):
    # - Cortex → Striatum: Fast, heavily myelinated, short distance (~2-4cm) → 3-5ms
    # - Hippocampus → Striatum: Moderate, longer distance (~4-6cm) → 7-10ms
    # - PFC → Striatum: Variable, longest distance (~6-10cm) → 12-18ms
    builder.connect(
        source="cortex",
        target="striatum",
        source_population="l5",
        target_population="d1",
        axonal_delay_ms=4.0,
    )
    builder.connect(
        source="hippocampus",
        target="striatum",
        source_population="ca1",
        target_population="d1",
        axonal_delay_ms=8.5,
    )
    builder.connect(
        source="pfc",
        target="striatum",
        source_population="executive",
        target_population="d1",
        axonal_delay_ms=15.0,
    )

    # Striatum → PFC: Basal ganglia gating of working memory
    # Via thalamus (MD/VA nuclei), total distance ~8-12cm (Haber 2003)
    # Includes striatum→thalamus→PFC relay → 15-20ms total delay
    builder.connect(
        source="striatum",
        target="pfc",
        source_population="d1",
        target_population="executive",
        axonal_delay_ms=17.5,
    )

    # Cerebellum: Motor/cognitive forward models
    # Receives multi-modal input (sensory + goals), outputs predictions
    # Corticopontocerebellar pathway: via pontine nuclei (Schmahmann 1996)
    # Distance: ~10-15cm total, includes relay → 20-30ms delay
    builder.connect(
        source="cortex",
        target="cerebellum",
        source_population="l5",
        target_population="granule",
        axonal_delay_ms=25.0,
    )

    # PFC → Cerebellum: similar pathway length
    # Goal/context input
    builder.connect(
        source="pfc",
        target="cerebellum",
        source_population="executive",
        target_population="granule",
        axonal_delay_ms=25.0,
    )

    # Cerebellum → Cortex: via thalamus (VL/VA nuclei), moderately fast
    # Distance: ~8-12cm, includes thalamic relay → 15-20ms delay
    # Forward model predictions
    builder.connect(
        source="cerebellum",
        target="cortex",
        source_population="prediction",
        target_population="l4",
        axonal_delay_ms=17.5,
    )

    # PFC ⇄ Hippocampus: Bidirectional memory-executive integration
    # Critical for goal-directed memory retrieval and episodic future thinking

    # PFC → Hippocampus: Top-down memory retrieval, schema application
    # PFC guides hippocampal retrieval via direct monosynaptic pathway
    # Distance: ~5-7cm, conduction velocity: ~3-5 m/s → 12-18ms delay
    # Enables: goal-directed memory search, schema-based encoding modulation
    builder.connect(
        source="pfc",
        target="hippocampus",
        source_population="executive",
        target_population="ca1",
        axonal_delay_ms=15.0,
    )

    # Hippocampus → PFC: Memory-guided decision making
    # Hippocampal replay guides PFC working memory and planning
    # Distance: ~5-7cm, similar pathway to reverse direction → 10-15ms delay
    # Enables: episodic future thinking, memory-guided decisions
    builder.connect(
        source="hippocampus",
        target="pfc",
        source_population="ca1",
        target_population="executive",
        axonal_delay_ms=12.0,
    )

    # PFC → Cortex: Top-down attention and cognitive control
    # PFC biases cortical processing via direct feedback projections to L2/3
    # Corticocortical feedback: ~5-8cm, conduction velocity: ~3-6 m/s → 10-15ms
    # Biological accuracy: PFC feedback targets superficial layers (L2/3), NOT L4
    # This bypasses thalamic input and directly modulates cortical representations
    # Enables: attentional bias, cognitive control over perception, working memory maintenance
    builder.connect(
        source="pfc",
        target="cortex",
        source_population="executive",
        target_population="l23",
        axonal_delay_ms=12.0,
    )

    # Thalamus → Hippocampus: Direct sensory-to-memory pathway (bypass cortex)
    # Nucleus reuniens provides direct thalamic input to hippocampus
    # Fast subcortical route for unfiltered sensory encoding
    # Distance: ~4-6cm, conduction velocity: ~5-8 m/s → 6-10ms delay
    # Enables: fast sensory encoding, subcortical memory formation
    builder.connect(
        source="thalamus",
        target="hippocampus",
        source_population="relay",
        target_population="dg",
        axonal_delay_ms=8.0,
    )

    # Thalamus → Striatum: Thalamostriatal pathway for habitual responses
    # Direct sensory-action pathway bypassing cortex (Smith et al. 2004, 2009, 2014)
    # Fast subcortical route for stimulus-response habits
    # Distance: ~3-5cm, conduction velocity: ~6-10 m/s → 4-7ms delay
    # Enables: fast habitual responses, stimulus-response learning, subcortical reflexes
    builder.connect(
        source="thalamus",
        target="striatum",
        source_population="relay",
        target_population="d1",
        axonal_delay_ms=5.0,
    )

    # Medial Septum → Hippocampus: Septal theta drive for emergent oscillations
    # GABAergic pacemaker neurons phase-lock hippocampal OLM interneurons
    # OLM dendritic inhibition creates emergent encoding/retrieval separation
    # Verga et al. 2014: Medial septum critically gates hippocampal theta
    # Distance: ~1-2cm (local subcortical), well-myelinated → 2ms delay
    # CRITICAL: This connection enables emergent theta (replaces hardcoded sinusoid)
    builder.connect(
        source="medial_septum",
        target="hippocampus",
        source_population="gaba",
        target_population="ca3",
        axonal_delay_ms=2.0,
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
    # Distance: ~1-2cm, well-myelinated → 2-3ms delay
    builder.connect(
        source="striatum",
        target="snr",
        source_population="d1",
        target_population="d1_input",
        axonal_delay_ms=2.5,
    )
    builder.connect(
        source="striatum",
        target="snr",
        source_population="d2",
        target_population="d2_input",
        axonal_delay_ms=2.5,
    )

    # SNr → VTA: Value estimate feedback for TD learning
    # SNr provides inhibitory feedback encoding value: V(s) ∝ 1/firing_rate
    # High SNr firing (low D1) → high inhibition → low value signal to VTA
    # Low SNr firing (high D1) → low inhibition → high value signal to VTA
    # Distance: ~0.5-1cm (adjacent midbrain nuclei) → 1-2ms delay
    builder.connect(
        source="snr",
        target="vta",
        source_population="value",
        target_population="snr_value",
        axonal_delay_ms=1.5,
    )

    # RewardEncoder → VTA: External reward signal delivery
    # Population-coded reward spikes from environment/task
    # VTA decodes to scalar reward: r = mean(spikes) / max_rate
    # Note: RewardEncoder receives external input via Brain.forward({"reward_encoder": reward_spikes})
    # Distance: Conceptual (external input pathway) → minimal delay
    builder.connect(
        source="reward_encoder",
        target="vta",
        source_population="reward_signal",
        target_population="reward_signal",
        axonal_delay_ms=0.5,
    )

    # VTA → Striatum: Dopamine release for three-factor learning
    # DA neuron spikes converted to concentration by D1/D2 receptors
    # Modulates synaptic plasticity: Δw = eligibility × DA × lr
    # D1 receptors: Gs-coupled → facilitate LTP
    # D2 receptors: Gi-coupled → facilitate LTD
    # Distance: ~2-3cm (mesolimbic pathway), well-myelinated → 3-5ms delay
    builder.connect(
        source="vta",
        target="striatum",
        source_population="da_output",
        target_population="vta_da",
        axonal_delay_ms=4.0,
    )

    # VTA → PFC: Dopamine release for working memory gating
    # DA neuron spikes converted to concentration by NeuromodulatorReceptor
    # Modulates working memory gate: high DA → update WM, low DA → maintain WM
    # Also modulates D1/D2 receptor subtypes for differential excitability
    # Distance: ~2-3cm (mesocortical pathway), well-myelinated → 3-5ms delay
    builder.connect(
        source="vta",
        target="pfc",
        source_population="da_output",
        target_population="vta_da",
        axonal_delay_ms=4.0,
    )

    # VTA → Hippocampus: Dopamine release for novelty/salience modulation
    # DA enhances LTP for novel patterns and modulates consolidation
    # Minimal 10% projection strength (applied at receptor processing)
    # Distance: ~2-3cm, well-myelinated → 3-5ms delay
    builder.connect(
        source="vta",
        target="hippocampus",
        source_population="da_output",
        target_population="vta_da",
        axonal_delay_ms=4.0,
    )

    # VTA → Cortex: Dopamine release for Layer 5 action selection
    # DA modulates L5 pyramidal cells that project to striatum/brainstem
    # Sparse 30% projection to deep layers (applied at receptor processing)
    # Distance: ~2-3cm, well-myelinated → 3-5ms delay
    builder.connect(
        source="vta",
        target="cortex",
        source_population="da_output",
        target_population="vta_da",
        axonal_delay_ms=4.0,
    )

    # =============================================================================
    # RECURRENT CONNECTIONS (Externalized for Biological Accuracy)
    # =============================================================================

    # Hippocampus CA3 → CA3: Autoassociative recurrent memory
    # Local recurrent collaterals within CA3 for pattern completion
    # Distance: ~200-500μm (local), unmyelinated collaterals → 1-3ms delay
    # STP: Fast depression prevents frozen attractors (tau_d ~200ms)
    # Phase diversity: ±15% weight variation seeds temporal coding
    # ACh modulation applied at region level (encoding vs retrieval)
    builder.connect(
        source="hippocampus",
        target="hippocampus",
        source_population="ca3",
        target_population="ca3",
        axonal_delay_ms=2.0,  # Local recurrent collaterals
    )

    # Cortex L2/3 → L2/3: Lateral recurrent processing
    # Horizontal connections within L2/3 for associative computation
    # Distance: ~1-3mm (local horizontal), mixed myelination → 1-3ms delay
    # STP: Depression prevents runaway recurrent activity
    # Signed weights: E/I mix for lateral excitation and inhibition
    # ACh modulation applied at region level (encoding vs retrieval)
    builder.connect(
        source="cortex",
        target="cortex",
        source_population="l23",
        target_population="l23",
        axonal_delay_ms=2.0,  # Horizontal connections
    )

    # Striatum D1 → D1: Lateral inhibition for action selection
    # MSN→MSN GABAergic collaterals create winner-take-all dynamics
    # Distance: ~100-300μm (local), unmyelinated → 1-2ms delay
    # Enables action-specific competition (Moyer et al. 2014)
    builder.connect(
        source="striatum",
        target="striatum",
        source_population="d1",
        target_population="d1",
        axonal_delay_ms=1.5,  # Fast local MSN collaterals
    )

    # Striatum D2 → D2: Lateral inhibition for NoGo pathway
    # Similar MSN→MSN collaterals in indirect pathway
    builder.connect(
        source="striatum",
        target="striatum",
        source_population="d2",
        target_population="d2",
        axonal_delay_ms=1.5,
    )

    # Thalamus TRN → Relay: Inhibitory attentional gating
    # TRN provides feedforward and lateral inhibition to relay neurons
    # Distance: ~50-200μm (local), unmyelinated → 0.5-1ms delay
    # Implements searchlight attention mechanism (Guillery & Harting 2003)
    builder.connect(
        source="thalamus",
        target="thalamus",
        source_population="trn",
        target_population="relay",
        axonal_delay_ms=1.0,  # Very fast local inhibition
    )


# Register built-in presets
BrainBuilder.register_preset(
    name="default",
    description=(
        "Default 10-region architecture with thalamus, cortex, hippocampus, PFC, striatum, cerebellum, "
        "medial septum, and spiking dopamine system (RewardEncoder, SNr, VTA). "
        "Includes biologically realistic connections and delays for general-purpose learning, "
        "emergent theta oscillations, and reinforcement learning with burst/pause dopamine dynamics."
    ),
    builder_fn=_build_default,
)
