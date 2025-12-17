"""
Dynamic Brain - Component Graph Executor

This module implements a flexible brain architecture where the brain is
treated as a directed graph of neural components (regions, pathways, modules).

Unlike EventDrivenBrain (hardcoded 6 regions), DynamicBrain supports:
- Arbitrary number of components
- Flexible topologies (not limited to fixed connectivity)
- User-defined custom components via ComponentRegistry
- Dynamic component addition/removal
- Plugin architecture for external extensions
- Event-driven execution with axonal delays
- Optional parallel execution across multiple CPU cores

Architecture:
    DynamicBrain = Graph of NeuralComponents
    - nodes: regions, pathways, modules (all inherit from NeuralComponent)
    - edges: data flow between components
    - execution: event-driven via EventScheduler OR parallel via ParallelExecutor

Author: Thalia Project
Date: December 15, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING

import torch
import torch.nn as nn

from thalia.regions.base import NeuralComponent
from thalia.events.system import EventScheduler, Event, EventType, SpikePayload
from thalia.events.adapters.base import EventRegionConfig
from thalia.core.diagnostics import (
    StriatumDiagnostics,
    HippocampusDiagnostics,
    BrainSystemDiagnostics,
)

if TYPE_CHECKING:
    from thalia.config import GlobalConfig, ThaliaConfig
    from thalia.managers.component_registry import ComponentRegistry
    from thalia.events.adapters import EventDrivenRegionBase


@dataclass
class ComponentSpec:
    """Specification for a brain component.

    Used by BrainBuilder to define components before instantiation.
    """
    name: str
    """Instance name (e.g., 'cortex', 'my_visual_cortex')"""

    component_type: str
    """Registry type: 'region', 'pathway', or 'module'"""

    registry_name: str
    """Registry key (e.g., 'layered_cortex', 'spiking_stdp')"""

    config_params: Dict[str, Any] = field(default_factory=dict)
    """Configuration parameters for the component"""

    instance: Optional[NeuralComponent] = None
    """Instantiated component (set during build)"""


@dataclass
class ConnectionSpec:
    """Specification for a connection between components.

    Represents an edge in the component graph.

    Supports port-based routing for layer-specific outputs and
    multiple input types (feedforward, top-down, sensory, etc.).
    """
    source: str
    """Source component name"""

    target: str
    """Target component name"""

    pathway_type: str
    """Pathway registry name (e.g., 'spiking_stdp', 'attention')"""

    source_port: Optional[str] = None
    """Source output port (e.g., 'l23', 'l5', 'default')"""

    target_port: Optional[str] = None
    """Target input port (e.g., 'feedforward', 'top_down', 'ec_l3')"""

    config_params: Dict[str, Any] = field(default_factory=dict)
    """Pathway configuration parameters"""

    instance: Optional[NeuralComponent] = None
    """Instantiated pathway (set during build)"""


class DynamicBrain(nn.Module):
    """Dynamic brain constructed from component graph.

    Unlike EventDrivenBrain (hardcoded 6 regions), DynamicBrain builds
    arbitrary topologies from registered components.

    Key Differences from EventDrivenBrain:
    - Flexible component graph vs. hardcoded regions
    - User-extensible via ComponentRegistry
    - Arbitrary connectivity patterns
    - Plugin support for external components

    Architecture:
        - components: Dict[name -> NeuralComponent] (nodes)
        - connections: Dict[(source, target) -> Pathway] (edges)
        - topology: Directed graph adjacency list
        - execution: Topological ordering or parallel

    Example:
        components = {
            "thalamus": ThalamicRelay(config),
            "cortex": LayeredCortex(config),
            "hippocampus": Hippocampus(config),
        }

        connections = {
            ("thalamus", "cortex"): SpikingPathway(config),
            ("cortex", "hippocampus"): SpikingPathway(config),
        }

        brain = DynamicBrain(components, connections, global_config)

    Supports:
        - Custom user regions/pathways (via ComponentRegistry)
        - Arbitrary connectivity patterns
        - Dynamic component addition/removal
        - Checkpoint save/load of arbitrary graphs
    """

    def __init__(
        self,
        components: Dict[str, NeuralComponent],
        connections: Dict[Tuple[str, str], NeuralComponent],
        global_config: "GlobalConfig",
        use_parallel: bool = False,
        n_workers: Optional[int] = None,
        connection_specs: Optional[Dict[Tuple[str, str], Any]] = None,
    ):
        """Initialize DynamicBrain from component graph.

        Args:
            components: Dict mapping component names to instances
            connections: Dict mapping (source, target) tuples to pathways
            global_config: Global configuration (device, dt_ms, etc.)
            use_parallel: Enable parallel execution (multi-core CPU)
            n_workers: Number of worker processes (default: number of regions)
            connection_specs: Optional dict with source_port/target_port info per connection
        """
        super().__init__()

        self.global_config = global_config
        self.device = torch.device(global_config.device)

        # Minimal config for checkpoint compatibility
        # Note: Sizes will be added after components are known
        from types import SimpleNamespace
        self.config = SimpleNamespace(device=global_config.device)

        # Store components as nn.ModuleDict for proper parameter tracking
        self.components = nn.ModuleDict(components)

        # Update config with component sizes (for CheckpointManager compatibility)
        self._update_config_sizes()

        # Store connections with tuple keys for easy lookup
        # Also register in ModuleDict for parameter tracking
        self.connections: Dict[Tuple[str, str], NeuralComponent] = connections
        self._connection_modules = nn.ModuleDict({
            f"{src}_to_{tgt}": pathway
            for (src, tgt), pathway in connections.items()
        })

        # Store connection specs (for port-based routing)
        self._connection_specs = connection_specs or {}

        # Build topology graph for execution order
        self._topology = self._build_topology_graph()
        self._execution_order: Optional[List[str]] = None

        # Current simulation time
        self._current_time: float = 0.0

        # =================================================================
        # EVENT-DRIVEN EXECUTION
        # =================================================================
        # Use EventScheduler for event-driven execution with delays
        self._scheduler = EventScheduler()

        # Event adapters (lazy initialization - created when needed for event-driven execution)
        self._event_adapters: Optional[Dict[str, "EventDrivenRegionBase"]] = None

        # === MULTI-SOURCE PATHWAY BUFFERING ===
        # Buffer for multi-source pathway inputs
        # Maps target_name -> {source_name -> latest_output}
        self._multi_source_buffers: Dict[str, Dict[str, torch.Tensor]] = {}

        # =================================================================
        # PARALLEL EXECUTION (Optional)
        # =================================================================
        # Parallel execution distributes components across worker processes
        # for true multi-core CPU parallelism. GPU tensors are automatically
        # serialized to CPU for inter-process communication.
        self.use_parallel = use_parallel
        self._parallel_executor = None
        self.n_workers = n_workers

        if use_parallel:
            self._init_parallel_executor()

        # =================================================================
        # PATHWAY MANAGER (Phase 1.7.1)
        # =================================================================
        # Centralized pathway management for diagnostics and growth coordination
        from thalia.pathways.dynamic_pathway_manager import DynamicPathwayManager

        self.pathway_manager = DynamicPathwayManager(
            connections=self.connections,
            topology=self._topology,
            device=self.device,
            dt_ms=self.global_config.dt_ms,
        )

        # Backward compatibility: expose pathways dict
        self.pathways = self.pathway_manager.get_all_pathways()

        # =================================================================
        # OSCILLATOR MANAGER (Phase 1.7.2)
        # =================================================================
        # Rhythmic coordination via all brain oscillations (delta, theta, alpha, beta, gamma, ripple)
        # Provides theta-driven encoding/retrieval, gamma feature binding, cross-frequency coupling
        from thalia.coordination.oscillator import OscillatorManager, OSCILLATOR_DEFAULTS

        # Get frequencies from config if available, otherwise use defaults
        delta_freq = getattr(global_config, 'delta_frequency_hz', OSCILLATOR_DEFAULTS['delta'])
        theta_freq = getattr(global_config, 'theta_frequency_hz', OSCILLATOR_DEFAULTS['theta'])
        alpha_freq = getattr(global_config, 'alpha_frequency_hz', OSCILLATOR_DEFAULTS['alpha'])
        beta_freq = getattr(global_config, 'beta_frequency_hz', OSCILLATOR_DEFAULTS['beta'])
        gamma_freq = getattr(global_config, 'gamma_frequency_hz', OSCILLATOR_DEFAULTS['gamma'])
        ripple_freq = getattr(global_config, 'ripple_frequency_hz', OSCILLATOR_DEFAULTS['ripple'])

        self.oscillators = OscillatorManager(
            dt_ms=self.global_config.dt_ms,
            device=self.global_config.device,
            delta_freq=delta_freq,
            theta_freq=theta_freq,
            alpha_freq=alpha_freq,
            beta_freq=beta_freq,
            gamma_freq=gamma_freq,
            ripple_freq=ripple_freq,
            couplings=None,  # Use default couplings (theta-gamma, etc.)
        )

        # =================================================================
        # NEUROMODULATOR SYSTEMS (Phase 1.6.3)
        # =================================================================
        # VTA (dopamine), LC (norepinephrine), NB (acetylcholine)
        # Provides centralized neuromodulation
        from thalia.neuromodulation.manager import NeuromodulatorManager

        self.neuromodulator_manager = NeuromodulatorManager()

        # =================================================================
        # REINFORCEMENT LEARNING STATE (Phase 1.6.2)
        # =================================================================
        self._last_action: Optional[int] = None
        self._last_confidence: Optional[float] = None

        # Mutable container for shared last_action state (needed by ConsolidationManager)
        self._last_action_container = [None]

        # =================================================================
        # SPIKE COUNTING
        # =================================================================
        # Track total spikes per component for diagnostics
        self._spike_counts: Dict[str, int] = {name: 0 for name in components.keys()}

        # =================================================================
        # GROWTH HISTORY TRACKING (Phase 1.7.7)
        # =================================================================
        # Track all growth events for analysis and debugging
        self._growth_history: List[Any] = []  # List[GrowthEvent] from coordination.growth

        # =================================================================
        # CONSOLIDATION MANAGER (Phase 1.7.4)
        # =================================================================
        # Manages memory consolidation and offline replay
        # Initialize if brain has required components (hippocampus, striatum, cortex, pfc)
        self.consolidation_manager = None

        if all(comp in self.components for comp in ['hippocampus', 'striatum', 'cortex', 'pfc']):
            from thalia.memory.consolidation.manager import ConsolidationManager

            self.consolidation_manager = ConsolidationManager(
                hippocampus=self.components['hippocampus'],
                striatum=self.components['striatum'],
                cortex=self.components['cortex'],
                pfc=self.components['pfc'],
                config=self.config,
                deliver_reward_fn=self.deliver_reward,
            )

            # Set cortex L5 size if LayeredCortex
            if hasattr(self.components['cortex'], 'l5_size'):
                self.consolidation_manager.set_cortex_l5_size(self.components['cortex'].l5_size)
            elif hasattr(self.components['cortex'], 'config') and hasattr(self.components['cortex'].config, 'l5_size'):
                self.consolidation_manager.set_cortex_l5_size(self.components['cortex'].config.l5_size)

        # =================================================================
        # CHECKPOINT MANAGER (Phase 1.7.4)
        # =================================================================
        # Centralized checkpoint save/load with compression and validation
        from thalia.io.checkpoint_manager import CheckpointManager

        self.checkpoint_manager = CheckpointManager(
            brain=self,
            default_compression='zstd',  # Default compression format
        )

        # =================================================================
        # PLANNING SYSTEMS (Phase 1.7.5)
        # =================================================================
        # Mental simulation and Dyna planning for model-based RL
        # Only initialize if use_model_based_planning enabled in global_config
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from thalia.planning import MentalSimulationCoordinator, DynaPlanner

        self.mental_simulation: Optional["MentalSimulationCoordinator"] = None
        self.dyna_planner: Optional["DynaPlanner"] = None

        # Check if planning is enabled (either direct flag or via brain config)
        planning_enabled = (
            getattr(global_config, 'use_model_based_planning', False) or
            (hasattr(global_config, 'brain') and getattr(global_config.brain, 'use_model_based_planning', False))
        )

        if planning_enabled:
            # Only import if enabled (optional dependency)
            from thalia.planning import (
                MentalSimulationCoordinator,
                SimulationConfig,
                DynaPlanner,
                DynaConfig,
            )

            # Check that required components exist
            if all(name in self.components for name in ['pfc', 'hippocampus', 'striatum', 'cortex']):
                # Create mental simulation coordinator
                self.mental_simulation = MentalSimulationCoordinator(
                    pfc=self.components['pfc'],
                    hippocampus=self.components['hippocampus'],
                    striatum=self.components['striatum'],
                    cortex=self.components['cortex'],
                    config=SimulationConfig(),
                )

                # Create Dyna planner for background planning
                self.dyna_planner = DynaPlanner(
                    coordinator=self.mental_simulation,
                    striatum=self.components['striatum'],
                    hippocampus=self.components['hippocampus'],
                    config=DynaConfig(),
                )

        # Initialize spike count tracking (for diagnostics and intrinsic rewards)
        self._spike_counts: Dict[str, int] = {name: 0 for name in components.keys()}

        # =================================================================
        # HEALTH & CRITICALITY MONITORING (Phase 1.7.6)
        # =================================================================
        # Monitor network health and criticality for training diagnostics
        from thalia.diagnostics import HealthMonitor, CriticalityMonitor

        # Initialize health monitor (always enabled)
        self.health_monitor = HealthMonitor(
            enable_oscillator_monitoring=True  # We have oscillators
        )

        # Initialize criticality monitor (optional, enabled by config)
        self.criticality_monitor: Optional[CriticalityMonitor] = None
        criticality_enabled = getattr(global_config, 'monitor_criticality', False)

        if criticality_enabled:
            # CriticalityMonitor doesn't need component sizes - it tracks spike counts directly
            self.criticality_monitor = CriticalityMonitor()

        # =================================================================
        # COMPONENT METADATA
        # =================================================================
        self._component_specs: Dict[str, ComponentSpec] = {}
        """Component specifications (set by BrainBuilder after build)"""

        self._registry: Optional["ComponentRegistry"] = None
        """Component registry reference (for component type lookup)"""

    def _update_config_sizes(self) -> None:
        """Update config with component sizes for CheckpointManager compatibility.

        EventDrivenBrain has config.input_size, config.cortex_size, etc.
        DynamicBrain extracts these from components if they exist.
        """
        # Try to extract common size attributes for compatibility
        if "thalamus" in self.components:
            thalamus = self.components["thalamus"]
            # Try n_input first, then n_output as fallback
            if hasattr(thalamus.config, "n_input"):
                self.config.input_size = thalamus.config.n_input
            elif hasattr(thalamus.config, "n_output"):
                self.config.input_size = thalamus.config.n_output

        # Extract region sizes
        for region_name in ["cortex", "hippocampus", "pfc", "striatum", "cerebellum"]:
            if region_name in self.components:
                component = self.components[region_name]
                if hasattr(component.config, "n_output"):
                    setattr(self.config, f"{region_name}_size", component.config.n_output)

        # Extract n_actions from striatum
        if "striatum" in self.components:
            striatum = self.components["striatum"]
            if hasattr(striatum.config, "n_actions"):
                self.config.n_actions = striatum.config.n_actions

        # Set defaults for any missing attributes (for CheckpointManager compatibility)
        if not hasattr(self.config, "input_size"):
            self.config.input_size = None
        if not hasattr(self.config, "n_actions"):
            self.config.n_actions = None
        if not hasattr(self.config, 'hippocampus_size'):
            self.config.hippocampus_size = getattr(
                self.components.get('hippocampus'), 'n_output', 128
            ) if 'hippocampus' in self.components else 128
        if not hasattr(self.config, 'pfc_size'):
            self.config.pfc_size = getattr(
                self.components.get('pfc'), 'n_output', 64
            ) if 'pfc' in self.components else 64
        if not hasattr(self.config, 'device'):
            self.config.device = str(self.device)

    @property
    def current_time(self) -> float:
        """Get current simulation time in milliseconds.

        Returns:
            Current simulation time in ms
        """
        return self._current_time



    @property
    def theta_oscillator(self):
        """Get theta oscillator (compatibility property for monitoring).

        Returns wrapper object with get_frequency() method.
        """
        class ThetaWrapper:
            def __init__(self, theta_osc):
                self._theta = theta_osc

            def get_frequency(self):
                return self._theta.frequency_hz

        return ThetaWrapper(self.oscillators.theta)

    def measure_phase_locking(self) -> float:
        """Measure gamma-theta phase locking.

        Computes the phase-amplitude coupling between gamma amplitude
        and theta phase, a key metric of oscillator coordination.

        Returns:
            Phase locking value [0, 1], where higher is better coupling
        """
        # Get current phases
        theta_phase = self.oscillators.theta.phase
        gamma_phase = self.oscillators.gamma.phase

        # Get coupling amplitude if available
        coupled_amps = self.oscillators.get_coupled_amplitudes()
        if 'gamma' in coupled_amps:
            # Use coupling amplitude as proxy for phase locking
            # (higher coupling = better phase locking)
            return coupled_amps['gamma']

        # Fallback: compute simple phase coherence
        # Gamma should be at ~40 Hz, theta at ~8 Hz
        # Ideal ratio is 5:1 (5 gamma cycles per theta cycle)
        expected_ratio = 5.0
        actual_ratio = self.oscillators.gamma.frequency_hz / self.oscillators.theta.frequency_hz

        # Phase locking = how close to expected ratio
        ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio
        phase_locking = max(0.0, 1.0 - ratio_error)

        return phase_locking

    @classmethod
    def from_thalia_config(cls, config: "ThaliaConfig") -> "DynamicBrain":
        """Create DynamicBrain from ThaliaConfig (backward compatibility).

        This factory method enables drop-in replacement of EventDrivenBrain
        with DynamicBrain using the same ThaliaConfig format.

        Args:
            config: ThaliaConfig with global_, brain, and region configs

        Returns:
            DynamicBrain instance matching EventDrivenBrain architecture

        Example:
            config = ThaliaConfig(
                global_=GlobalConfig(device="cpu", dt_ms=1.0),
                brain=BrainConfig(sizes=RegionSizes(...)),
            )
            brain = DynamicBrain.from_thalia_config(config)

        Note:
            This manually builds the sensorimotor topology with sizes
            from config.brain.sizes to ensure exact compatibility.

            Parallel execution can be enabled via config.brain.parallel=True.
        """
        from thalia.core.brain_builder import BrainBuilder

        sizes = config.brain.sizes

        # Check if parallel mode requested
        use_parallel = getattr(config.brain, "parallel", False)
        n_workers = getattr(config.brain, "n_workers", None)

        builder = BrainBuilder(config.global_)

        # Add regions with sizes from config
        # Thalamus is input interface
        builder.add_component(
            "thalamus", "thalamus",
            n_input=sizes.input_size,
            n_output=sizes.thalamus_size
        )

        # Cortex - use cortex_type from config (PREDICTIVE or LAYERED)
        # Both types expose l23_size/l5_size for port-based routing
        cortex_registry_name = "predictive_cortex" if config.brain.cortex_type.value == "predictive" else "cortex"

        # Pass explicit layer sizes if available (all-or-nothing per strict validation)
        cortex_config = {"n_output": sizes.cortex_size}
        if sizes._cortex_l4_size is not None and sizes._cortex_l23_size is not None and sizes._cortex_l5_size is not None:
            cortex_config.update({
                "l4_size": sizes.cortex_l4_size,
                "l23_size": sizes.cortex_l23_size,
                "l5_size": sizes.cortex_l5_size,
            })

        builder.add_component("cortex", cortex_registry_name, **cortex_config)
        builder.add_component("hippocampus", "hippocampus", n_output=sizes.hippocampus_size)
        builder.add_component("pfc", "prefrontal", n_output=sizes.pfc_size)
        builder.add_component("striatum", "striatum", n_output=sizes.n_actions)
        # Cerebellum outputs n_actions (motor commands)
        builder.add_component("cerebellum", "cerebellum", n_output=sizes.n_actions)

        # Add connections (standard sensorimotor topology)
        # Thalamus → Cortex → Hippocampus/PFC → Striatum → Cerebellum
        builder.connect("thalamus", "cortex", "spiking")
        builder.connect("cortex", "hippocampus", "spiking", source_port="l23")  # Cortico-cortical
        builder.connect("hippocampus", "cortex", "spiking")  # Bidirectional (replay pathway)
        builder.connect("cortex", "pfc", "spiking", source_port="l23")  # Cortico-cortical
        builder.connect("hippocampus", "pfc", "spiking")
        # Striatum receives from 3 sources: cortex L5, hippocampus, pfc
        builder.connect("cortex", "striatum", "spiking", source_port="l5")  # Subcortical
        builder.connect("hippocampus", "striatum", "spiking")
        builder.connect("pfc", "striatum", "spiking")
        builder.connect("striatum", "pfc", "spiking")  # Bidirectional
        builder.connect("pfc", "cerebellum", "spiking")  # PFC→cerebellum (motor refinement)

        return builder.build(use_parallel=use_parallel, n_workers=n_workers)

    def _build_topology_graph(self) -> Dict[str, List[str]]:
        """Build adjacency list of component dependencies.

        Returns:
            Dict mapping component names to list of downstream dependencies

        Example:
            {"thalamus": ["cortex"], "cortex": ["hippocampus", "striatum"]}
        """
        graph = {name: [] for name in self.components.keys()}

        # Extract connections from tuple keys
        for (src, tgt) in self.connections.keys():
            if src in graph:
                graph[src].append(tgt)

        return graph

    def forward(
        self,
        sensory_input: Optional[torch.Tensor] = None,
        n_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute brain for n_timesteps (EventDrivenBrain-compatible signature).

        Args:
            sensory_input: Input pattern [input_size], or None for maintenance
                          Will be routed to thalamus automatically
            n_timesteps: Number of timesteps to process (default: 10 if not configured)

        Returns:
            Dict containing:
                - "outputs": Dict[component_name -> final output]
                - "history": Optional list of timestep outputs
                - "time": Final simulation time

        Example:
            brain = DynamicBrain.from_thalia_config(config)

            sensory_input = torch.randn(128, device=device)
            result = brain.forward(sensory_input, n_timesteps=10)

            # Alternative: Dict-based input (legacy)
            result = brain.forward({"thalamus": sensory_input}, n_timesteps=10)
        """
        # Handle default n_timesteps
        if n_timesteps is None:
            n_timesteps = 10  # Reasonable default

        # Convert sensory_input to input_data dict format
        if sensory_input is not None:
            if isinstance(sensory_input, dict):
                # Already in dict format (legacy)
                input_data = sensory_input
            else:
                # Convert tensor to dict with thalamus as input
                input_data = {"thalamus": sensory_input}
        else:
            # No input (maintenance mode)
            input_data = {}
        # Choose execution mode
        if self.use_parallel:
            return self._forward_parallel(input_data, n_timesteps)
        else:
            return self._forward_event_driven(input_data, n_timesteps)

    # =================================================================
    # EVENT ADAPTER SYSTEM
    # =================================================================

    def _wrap_components_with_adapters(self) -> Dict[str, "EventDrivenRegionBase"]:
        """Wrap components with event adapters for event-driven execution.

        This method creates EventDrivenRegionBase adapters for each component:
        1. Check ComponentRegistry for custom adapter (e.g., EventDrivenCortex)
        2. Fall back to GenericEventAdapter for user-defined components
        3. Cache adapters for reuse

        Returns:
            Dict mapping component names to event adapters

        Raises:
            ValueError: If registry not set or required imports missing
        """
        from thalia.events.adapters import GenericEventAdapter
        from thalia.events.adapters import EventDrivenRegionBase  # type: ignore[attr-defined]

        if self._registry is None:
            raise ValueError(
                "ComponentRegistry not set. BrainBuilder should set brain._registry "
                "after build()."
            )

        adapters: Dict[str, EventDrivenRegionBase] = {}

        for name, component in self.components.items():
            # Get component spec (if available)
            spec = self._component_specs.get(name)

            if spec is None:
                # No spec available (old-style creation), use generic adapter
                adapter_class = GenericEventAdapter
            else:
                # Check registry for custom adapter
                adapter_class = self._registry.get_adapter(
                    component_type=spec.component_type,
                    name=spec.registry_name,
                )

                if adapter_class is None:
                    # No custom adapter, use generic
                    adapter_class = GenericEventAdapter

            # Instantiate adapter
            try:
                # Check if it's the generic adapter or a specialized one
                if adapter_class is GenericEventAdapter:
                    # GenericEventAdapter expects (region, config, global_config)
                    adapter = adapter_class(
                        region=component,
                        config=None,  # Let GenericEventAdapter create EventRegionConfig
                        global_config=self.global_config,
                    )
                else:
                    # Specialized adapters expect (config, component)
                    # Get device from component (it might be torch.device or string)
                    component_device = getattr(component, 'device', self.global_config.device)
                    if isinstance(component_device, torch.device):
                        device_str = str(component_device)
                    else:
                        device_str = component_device

                    # Create EventRegionConfig for adapter
                    event_config = EventRegionConfig(
                        name=name,
                        output_targets=[],  # Will be filled later
                        membrane_tau_ms=20.0,  # Default
                        device=device_str,
                    )
                    adapter = adapter_class(event_config, component)

                adapters[name] = adapter
            except Exception as e:
                raise ValueError(
                    f"Failed to create event adapter for component '{name}': {e}"
                ) from e

        return adapters

    def _ensure_adapters(self) -> Dict[str, "EventDrivenRegionBase"]:
        """Ensure event adapters are initialized (lazy initialization).

        Returns:
            Dict of event adapters
        """
        if self._event_adapters is None:
            self._event_adapters = self._wrap_components_with_adapters()
        return self._event_adapters

    def _get_downstream_targets(self, source_name: str) -> List[Tuple[str, float]]:
        """Get downstream targets for a component with axonal delays.

        Args:
            source_name: Name of source component

        Returns:
            List of (target_name, delay_ms) tuples

        Example:
            targets = self._get_downstream_targets("cortex")
            # [("hippocampus", 5.0), ("striatum", 3.0)]
        """
        targets = []

        for target_name in self._topology.get(source_name, []):
            # Get pathway from connections
            pathway_key = f"{source_name}_to_{target_name}"
            pathway = self.connections.get(pathway_key)

            if pathway is not None:
                # Extract axonal delay from pathway config
                delay_ms = getattr(pathway, "delay_ms", 1.0)
                targets.append((target_name, delay_ms))

        return targets

    def _forward_event_driven(
        self,
        input_data: Dict[str, torch.Tensor],
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Event-driven execution with realistic axonal delays.

        Uses EventScheduler to process spike events with proper timing.
        Components are wrapped with EventDrivenRegionBase adapters that:
        - Apply membrane decay between events
        - Translate events to component forward calls
        - Route output spikes to downstream targets
        """
        # Ensure event adapters are initialized
        adapters = self._ensure_adapters()

        # Clear scheduler state
        self._scheduler.clear()

        # Schedule initial sensory inputs
        for component_name, sensory_spikes in input_data.items():
            if component_name in self.components:
                event = Event(
                    time=self._current_time,
                    event_type=EventType.SENSORY,
                    source="external",
                    target=component_name,
                    payload=SpikePayload(spikes=sensory_spikes),
                )
                self._scheduler.schedule(event)

        # Storage for component outputs
        outputs = {name: None for name in self.components.keys()}

        # Process events until end time
        end_time = self._current_time + (n_timesteps * self.global_config.dt_ms)

        while not self._scheduler.is_empty:
            # Check if next event is within our time window
            next_time = self._scheduler.peek_time()
            if next_time is None or next_time > end_time:
                break

            # Get next batch of simultaneous events
            events = self._scheduler.pop_simultaneous(tolerance_ms=0.01)

            for event in events:
                # Get event adapter (not raw component)
                adapter = adapters.get(event.target)
                if adapter is None:
                    continue

                # Process event through adapter to get output spikes
                if event.event_type == EventType.SPIKE or event.event_type == EventType.SENSORY:
                    if not isinstance(event.payload, SpikePayload):
                        continue

                    # Apply decay and process spikes through adapter
                    dt = event.time - adapter._last_update_time
                    if dt > 0:
                        adapter._apply_decay(dt)
                    adapter._last_update_time = event.time

                    # Get output spikes (use impl for regions, forward for pathways)
                    if hasattr(adapter, 'impl'):
                        output_spikes = adapter.impl.forward(event.payload.spikes)
                    else:
                        output_spikes = adapter.forward(event.payload.spikes)

                    outputs[event.target] = output_spikes

                    # Track spike counts
                    if output_spikes is not None:
                        spike_count = int(output_spikes.sum().item())
                        self._spike_counts[event.target] += spike_count

                    # Schedule output events to downstream components
                    if output_spikes is not None and output_spikes.sum() > 0:
                        self._schedule_downstream_events(
                            source=event.target,
                            output_spikes=output_spikes,
                            current_time=event.time,
                        )

            # Advance scheduler time
            if events:
                self._current_time = events[0].time

            # Broadcast oscillator phases every timestep
            self._broadcast_oscillator_phases()

        # Ensure time advances to end_time even if no more events
        self._current_time = end_time

        # Count total events processed for compatibility with EventDrivenBrain
        events_processed = len([v for v in outputs.values() if v is not None])

        return {
            "outputs": outputs,
            "time": self._current_time,
            "spike_counts": self._spike_counts.copy(),
            "events_processed": events_processed,
            "final_time": self._current_time,
        }

    def _init_parallel_executor(self) -> None:
        """Initialize parallel executor with region creators.

        This creates pickle-able region creators for each component and spawns
        worker processes. Only standard brain regions are supported in parallel
        mode (thalamus, cortex, hippocampus, pfc, striatum, cerebellum).

        Note: Parallel mode requires CPU device for multiprocessing serialization.
        GPU tensors are automatically moved to CPU for inter-process communication.

        Raises:
            ValueError: If device is not CPU or components cannot be parallelized
        """
        from thalia.events.parallel import create_region_creator, ParallelExecutor

        # Validate device
        device_str = str(self.device)
        if "cuda" in device_str.lower() or "gpu" in device_str.lower():
            raise ValueError(
                "Parallel execution requires device='cpu' due to multiprocessing limitations. "
                "GPU tensors cannot be serialized between processes. "
                f"Current device: {device_str}"
            )

        # Build region creators for each component
        region_creators = {}

        for name, component in self.components.items():
            # Infer region type from component class
            region_type = self._infer_region_type(component)

            if region_type is None:
                raise ValueError(
                    f"Component '{name}' of type {type(component).__name__} cannot be used in "
                    f"parallel mode. Only standard brain regions are supported: "
                    f"ThalamicRelay, LayeredCortex, PredictiveCortex, Hippocampus, "
                    f"TrisynapticHippocampus, Prefrontal, Striatum, Cerebellum. "
                    f"Set use_parallel=False to use custom components."
                )

            # Build config dict for this component
            config_dict = self._build_config_dict_for_parallel(name, component)

            # Create region creator (pickle-able function reference)
            region_creators[name] = create_region_creator(
                region_type=region_type,
                config_dict=config_dict,
                device="cpu",
            )

        # Create parallel executor
        self._parallel_executor = ParallelExecutor(
            region_creators=region_creators,
            batch_tolerance_ms=0.1,
            device="cpu",
        )

        # Start worker processes
        self._parallel_executor.start()

    def _infer_region_type(self, component: NeuralComponent) -> Optional[str]:
        """Infer region type string from component class name.

        Args:
            component: Component instance

        Returns:
            Region type string for create_region_creator(), or None if unknown

        Supported types:
            - ThalamicRelay → "thalamus"
            - LayeredCortex, PredictiveCortex → "cortex"
            - Hippocampus, TrisynapticHippocampus → "hippocampus"
            - Prefrontal → "pfc"
            - Striatum → "striatum"
            - Cerebellum → "cerebellum"
        """
        class_name = type(component).__name__

        type_map = {
            "ThalamicRelay": "thalamus",
            "LayeredCortex": "cortex",
            "PredictiveCortex": "cortex",
            "Hippocampus": "hippocampus",
            "TrisynapticHippocampus": "hippocampus",
            "Prefrontal": "pfc",
            "Striatum": "striatum",
            "Cerebellum": "cerebellum",
        }

        return type_map.get(class_name)

    def _build_config_dict_for_parallel(
        self,
        name: str,
        component: NeuralComponent,
    ) -> Dict[str, Any]:
        """Build configuration dictionary for parallel region creator.

        Extracts configuration from component and topology to create a
        pickle-able dict that can be sent to worker processes.

        Args:
            name: Component name
            component: Component instance

        Returns:
            Config dict with all parameters needed to recreate the component
        """
        config_dict = {
            "name": name,
            "output_targets": self._topology.get(name, []),
        }

        # Extract config attributes
        if hasattr(component, "config"):
            cfg = component.config

            # Common attributes
            if hasattr(cfg, "n_input"):
                config_dict["n_input"] = cfg.n_input
            if hasattr(cfg, "n_output"):
                config_dict["n_output"] = cfg.n_output

            # Region-specific attributes
            class_name = type(component).__name__

            if class_name == "Hippocampus":
                # Hippocampus needs DG, CA3, CA1 sizes
                if hasattr(component, "dg_size"):
                    config_dict["dg_size"] = component.dg_size
                if hasattr(component, "ca3_size"):
                    config_dict["ca3_size"] = component.ca3_size
                if hasattr(component, "ca1_size"):
                    config_dict["ca1_size"] = component.ca1_size

            elif class_name == "Striatum":
                # Striatum needs n_actions and neurons_per_action
                if hasattr(cfg, "n_actions"):
                    config_dict["n_actions"] = cfg.n_actions
                if hasattr(cfg, "neurons_per_action"):
                    config_dict["neurons_per_action"] = cfg.neurons_per_action

            elif class_name == "Prefrontal":
                # PFC config
                if hasattr(cfg, "n_neurons"):
                    config_dict["n_neurons"] = cfg.n_neurons

            elif class_name == "Cerebellum":
                # Cerebellum config
                if hasattr(cfg, "n_purkinje"):
                    config_dict["n_purkinje"] = cfg.n_purkinje

        return config_dict

    def _forward_parallel(
        self,
        input_data: Dict[str, torch.Tensor],
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Parallel execution across multiple CPU cores.

        Uses ParallelExecutor to distribute components across worker processes.
        Events are batched and processed in parallel, with outputs collected
        and re-scheduled by the main process.

        Args:
            input_data: Dict mapping component names to input tensors
            n_timesteps: Number of timesteps to execute

        Returns:
            Dict with spike counts and execution stats (no per-component outputs
            in parallel mode, matching EventDrivenBrain behavior)
        """
        if self._parallel_executor is None:
            raise ValueError("Parallel executor not initialized (use_parallel=False)")

        # Inject sensory inputs as events
        for target, spikes in input_data.items():
            if target in self.components:
                self._parallel_executor.inject_sensory_input(
                    pattern=spikes,
                    target=target,
                    time=self._current_time,
                )

        # Calculate end time
        end_time = self._current_time + (n_timesteps * self.global_config.dt_ms)

        # Run parallel execution until end time
        result = self._parallel_executor.run_until(end_time)

        # Update time and spike counts
        self._current_time = end_time
        self._spike_counts = result["spike_counts"]

        return {
            "outputs": {},  # Parallel mode doesn't return per-component outputs
            "time": self._current_time,
            "spike_counts": self._spike_counts.copy(),
            "events_processed": result.get("events_processed", 0),
            "final_time": self._current_time,
        }

    def __del__(self):
        """Clean up parallel executor if active."""
        if hasattr(self, "_parallel_executor") and self._parallel_executor is not None:
            try:
                self._parallel_executor.stop()
            except Exception:
                pass  # Ignore errors during cleanup

    def _broadcast_oscillator_phases(self) -> None:
        """Broadcast oscillator phases to all components.

        Advances oscillators by dt_ms and updates all components with current
        phases for all frequency bands (delta, theta, alpha, beta, gamma, ripple).

        This enables:
        - Delta slow-wave sleep consolidation
        - Theta-driven encoding/retrieval in hippocampus
        - Alpha attention gating
        - Beta motor control and working memory
        - Gamma feature binding in cortex
        - Ripple sharp-wave replay
        """
        # Advance oscillators
        self.oscillators.advance(dt_ms=self.global_config.dt_ms)

        # Get all phases and signals
        phases = self.oscillators.get_phases()
        signals = self.oscillators.get_signals()

        # Broadcast to components via event adapters (if in event-driven mode)
        if self._event_adapters is not None:
            for adapter in self._event_adapters.values():
                if hasattr(adapter, 'set_oscillator_phases'):
                    adapter.set_oscillator_phases(phases, signals)
                # Backward compatibility: individual setters
                elif hasattr(adapter, 'set_theta_phase'):
                    adapter.set_theta_phase(phases['theta'])

        # Also broadcast directly to components that support it
        for component in self.components.values():
            if hasattr(component, 'set_oscillator_phases'):
                component.set_oscillator_phases(phases, signals)
            # Backward compatibility: individual setters
            elif hasattr(component, 'set_theta_phase'):
                component.set_theta_phase(phases['theta'])

    # =========================================================================
    # REINFORCEMENT LEARNING INTERFACE
    # =========================================================================

    def select_action(self, explore: bool = True, use_planning: bool = True) -> tuple[int, float]:
        """Select action based on current striatum state.

        Compatible with EventDrivenBrain RL interface. Uses striatum to
        select actions based on accumulated evidence.

        If use_planning=True and model-based planning is enabled:
        - Uses MentalSimulationCoordinator for tree search
        - Returns best action from simulated rollouts
        - Falls back to striatum if planning disabled

        Args:
            explore: Whether to allow exploration (epsilon-greedy)
            use_planning: Whether to use model-based planning

        Returns:
            (action, confidence): Selected action index and confidence [0, 1]

        Raises:
            ValueError: If striatum component not found in brain

        Example:
            # After forward pass
            brain.forward({"thalamus": sensory_input}, n_timesteps=20)

            # Select action
            action, confidence = brain.select_action(explore=True)

            # Execute action in environment
            next_state, reward, done = env.step(action)

            # Deliver reward for learning
            brain.deliver_reward(external_reward=reward)
        """
        if "striatum" not in self.components:
            raise ValueError(
                "Striatum component not found. Cannot select action. "
                "Brain must include 'striatum' component for RL tasks."
            )

        striatum = self.components["striatum"]

        # Use mental simulation if requested and available
        if use_planning and self.mental_simulation is not None:
            # Get current state from PFC
            current_state = None
            if 'pfc' in self.components:
                pfc = self.components['pfc']
                if hasattr(pfc, 'state') and pfc.state is not None:
                    current_state = pfc.state.spikes

            if current_state is not None:
                # Use tree search to find best action
                action = self.mental_simulation.search(
                    state=current_state,
                    n_simulations=10,
                )
                return action, 1.0  # High confidence from planning

        # Striatum has finalize_action method for action selection
        if hasattr(striatum, "finalize_action"):
            # Call finalize_action with correct signature (only explore parameter)
            result = striatum.finalize_action(explore=explore)

            # Extract action from result dict
            action = result["selected_action"]

            # Compute confidence from probabilities or net votes
            probs = result.get("probs")
            if probs is not None:
                # Softmax case: use max probability as confidence
                confidence = float(probs.max().item())
            else:
                # Argmax case: use normalized net votes as confidence
                net_votes = result["net_votes"]
                if net_votes.sum() > 0:
                    confidence = float(net_votes[action].item() / net_votes.sum().item())
                else:
                    confidence = 1.0 / len(net_votes)  # Uniform if no votes

            # Store for deliver_reward
            self._last_action = action
            self._last_confidence = confidence

            return action, confidence
        else:
            raise AttributeError(
                f"Striatum component ({type(striatum).__name__}) does not have "
                f"finalize_action method. Ensure striatum implements RL interface."
            )

    def deliver_reward(self, external_reward: Optional[float] = None) -> None:
        """Deliver reward signal for learning.

        Compatible with EventDrivenBrain RL interface. Updates striatum
        with reward prediction error for TD learning.

        The brain acts as VTA (ventral tegmental area):
        1. Combines external reward with intrinsic reward (if computed)
        2. Gets expected value from striatum for the action taken
        3. Computes reward prediction error (RPE = reward - expected)
        4. Broadcasts dopamine signal (normalized RPE) to striatum

        Args:
            external_reward: Task-based reward value (typically -1 to +1),
                           or None for pure intrinsic reward

        Raises:
            ValueError: If striatum not found or no action was selected

        Example:
            # After action selection
            action, _ = brain.select_action()

            # Execute in environment
            reward = env.step(action)

            # Deliver reward for learning
            brain.deliver_reward(external_reward=reward)

        Note:
            This is a simplified RL interface. For full EventDrivenBrain
            compatibility including intrinsic rewards, neuromodulation,
            and consolidation, use EventDrivenBrain or implement those
            features separately.
        """
        if "striatum" not in self.components:
            raise ValueError(
                "Striatum component not found. Cannot deliver reward. "
                "Brain must include 'striatum' component for RL tasks."
            )

        striatum = self.components["striatum"]

        # Compute total reward (external + intrinsic if available)
        intrinsic_reward = self._compute_intrinsic_reward()
        if external_reward is None:
            total_reward = intrinsic_reward
        else:
            total_reward = external_reward + intrinsic_reward
            total_reward = max(-2.0, min(2.0, total_reward))

        # Deliver reward to striatum for learning (only if action was selected)
        # Note: Match EventDrivenBrain's permissive behavior - allow deliver_reward()
        # without prior select_action() for streaming/continuous learning scenarios
        if hasattr(self, "_last_action") and self._last_action is not None:
            # Note: striatum.deliver_reward(reward) uses dopamine from VTA via set_dopamine()
            # The actual RPE computation and dopamine-gated learning happens inside striatum
            if hasattr(striatum, "deliver_reward"):
                striatum.deliver_reward(reward=total_reward)
            else:
                raise AttributeError(
                    f"Striatum component ({type(striatum).__name__}) does not have "
                    f"deliver_reward method. Ensure striatum implements RL interface."
                )

        # Store experience automatically (for replay) via consolidation manager
        if self.consolidation_manager is not None and hasattr(self, "_last_action") and self._last_action is not None:
            # Sync last_action to container for consolidation manager
            self._last_action_container[0] = self._last_action

            self.consolidation_manager.store_experience(
                action=self._last_action,
                reward=total_reward,
                last_action_holder=self._last_action_container,
            )

        # Trigger background planning (Dyna) after real experience
        if self.dyna_planner is not None:
            current_state = None
            next_state = None

            # Get current and next states from PFC
            if 'pfc' in self.components:
                pfc = self.components['pfc']
                if hasattr(pfc, 'state') and pfc.state is not None:
                    next_state = pfc.state.spikes
                    # Current state would need to be saved from before action
                    # For now, use same state (limitation: need state history)
                    current_state = next_state

            if current_state is not None and next_state is not None and self._last_action is not None:
                goal_context = current_state

                self.dyna_planner.process_real_experience(
                    state=current_state,
                    action=self._last_action,
                    reward=total_reward,
                    next_state=next_state,
                    done=False,
                    goal_context=goal_context
                )

        # Sync last_action to container for consolidation manager
        self._last_action_container[0] = self._last_action

    def deliver_reward_with_counterfactual(
        self,
        reward: float,
        is_match: bool,
        selected_action: int,
        counterfactual_scale: float = 0.5,
    ) -> Dict[str, Any]:
        """Deliver reward with counterfactual learning for non-selected action.

        Implements model-based RL: after experiencing a real outcome, we also
        simulate "what would have happened if I had chosen differently?"

        This solves asymmetric learning where only the selected action updates.
        Now BOTH actions learn on every trial:
        - Selected action: learns from actual outcome
        - Non-selected action: learns from counterfactual (imagined) outcome

        Args:
            reward: Actual reward received
            is_match: Whether this was a match trial
            selected_action: Action that was actually taken (0=MATCH, 1=NOMATCH)
            counterfactual_scale: How much to scale counterfactual learning (0-1)

        Returns:
            Dict with both real and counterfactual learning metrics
        """
        # First deliver the real reward
        self.deliver_reward(external_reward=reward)

        # Compute counterfactual reward for the other action
        other_action = 1 - selected_action
        correct_action = 0 if is_match else 1
        counterfactual_reward = 1.0 if (other_action == correct_action) else -1.0

        # Get novelty boost for modulation
        novelty_boost = self._get_novelty_boost()
        modulated_reward = counterfactual_reward * novelty_boost * counterfactual_scale

        # Apply counterfactual learning to striatum if available
        counterfactual_metrics = {}
        if 'striatum' in self.components:
            striatum = self.components['striatum']
            # Simulate counterfactual action selection
            if hasattr(striatum, 'deliver_reward'):
                # Create temporary action state for counterfactual
                saved_action = self._last_action
                self._last_action = other_action

                # Deliver scaled counterfactual reward
                striatum.deliver_reward(reward=modulated_reward)
                counterfactual_metrics['counterfactual_applied'] = True

                # Restore real action
                self._last_action = saved_action

        return {
            "real": {"reward": reward},
            "counterfactual": {
                "reward": counterfactual_reward,
                "scaled_reward": modulated_reward,
                "metrics": counterfactual_metrics,
            },
            "selected_action": selected_action,
            "other_action": other_action,
            "counterfactual_reward": counterfactual_reward,
            "novelty_boost": novelty_boost,
        }

    def _get_novelty_boost(self) -> float:
        """Get novelty-based learning rate multiplier.

        Returns:
            Multiplier >= 1.0 based on detected novelty
        """
        if not hasattr(self, '_novelty_signal'):
            self._novelty_signal = 1.0
        return max(1.0, self._novelty_signal)

    # =========================================================================
    # NEUROMODULATION (Phase 1.6.3)
    # =========================================================================

    def _update_neuromodulators(self) -> None:
        """Update centralized neuromodulator systems and broadcast to components.

        Updates VTA (dopamine), locus coeruleus (norepinephrine), and nucleus
        basalis (acetylcholine) based on intrinsic reward, uncertainty, and
        prediction error. Then broadcasts signals to all components.

        This is called automatically during forward() to maintain neuromodulator
        dynamics every timestep, matching EventDrivenBrain behavior.

        Neuromodulator Sources:
            - VTA: Tonic from intrinsic reward, phasic from RPE
            - LC: Arousal from uncertainty
            - NB: Encoding strength from prediction error

        Note:
            For now, intrinsic reward and uncertainty are simplified.
            Full implementation with curiosity and metacognition will be
            added in later phases.
        """
        # Compute signals for neuromodulator updates
        intrinsic_reward = self._compute_intrinsic_reward()
        uncertainty = self._compute_uncertainty()
        prediction_error = self._compute_prediction_error()
        cognitive_load = self._compute_cognitive_load()

        # Update neuromodulator systems
        self.neuromodulator_manager.vta.update(dt_ms=self.global_config.dt_ms, intrinsic_reward=intrinsic_reward)
        self.neuromodulator_manager.locus_coeruleus.update(dt_ms=self.global_config.dt_ms, uncertainty=uncertainty)
        self.neuromodulator_manager.nucleus_basalis.update(dt_ms=self.global_config.dt_ms, prediction_error=prediction_error)

        # Update PFC cognitive load for temporal discounting (Phase 3)
        if 'pfc' in self.components:
            pfc = self.components['pfc']
            if hasattr(pfc, 'discounter') and pfc.discounter is not None:
                pfc.update_cognitive_load(cognitive_load)
            elif hasattr(pfc, 'update_cognitive_load'):
                # Direct method on PFC component
                pfc.update_cognitive_load(cognitive_load)

        # Get current neuromodulator levels
        dopamine = self.neuromodulator_manager.vta.get_global_dopamine()
        norepinephrine = self.neuromodulator_manager.locus_coeruleus.get_norepinephrine()

        # Apply DA-NE coordination
        dopamine, norepinephrine = self.neuromodulator_manager.coordination.coordinate_da_ne(
            dopamine, norepinephrine, prediction_error
        )

        # Broadcast to all components
        regions = {name: comp for name, comp in self.components.items()}
        self.neuromodulator_manager.broadcast_to_regions(regions)

        # Broadcast to pathways
        for pathway in self.connections.values():
            if hasattr(pathway, "set_dopamine"):
                pathway.set_dopamine(dopamine)
            if hasattr(pathway, "set_norepinephrine"):
                pathway.set_norepinephrine(norepinephrine)

    def _compute_intrinsic_reward(self) -> float:
        """Compute intrinsic reward from the brain's internal objectives.

        This implements the free energy principle: the brain rewards itself
        for minimizing prediction error (surprise). Intrinsic reward is
        ALWAYS computed - it's the brain's continuous self-evaluation.

        Sources:
        1. **Cortex predictive coding**: Low free energy → good predictions → reward
        2. **Hippocampus pattern completion**: High similarity → successful recall → reward

        This is biologically plausible:
        - VTA dopamine neurons respond to internal prediction errors
        - Curiosity and "eureka" moments are intrinsically rewarding
        - The brain learns even without external feedback

        Returns:
            Intrinsic reward in range [-1, 1]
        """
        reward = 0.0
        n_sources = 0

        # =====================================================================
        # 1. CORTEX PREDICTIVE CODING (free energy minimization)
        # =====================================================================
        # Low prediction error = good model of the world = reward
        if 'cortex' in self.components:
            cortex = self.components['cortex']

            # Try PredictiveCortex first (has explicit free_energy)
            if hasattr(cortex, 'state') and hasattr(cortex.state, 'free_energy'):
                free_energy = cortex.state.free_energy

                # Free energy is typically 0-10, lower is better
                # Map: 0 → +1 (perfect prediction), 5 → 0, 10+ → -1 (bad prediction)
                cortex_reward = 1.0 - 0.2 * min(free_energy, 10.0)
                cortex_reward = max(-1.0, min(1.0, cortex_reward))
                reward += cortex_reward
                n_sources += 1

            # Fallback: check for accumulated free energy in PredictiveCortex
            elif hasattr(cortex, '_total_free_energy'):
                total_fe = cortex._total_free_energy
                cortex_reward = 1.0 - 0.1 * min(total_fe, 20.0)
                cortex_reward = max(-1.0, min(1.0, cortex_reward))
                reward += cortex_reward
                n_sources += 1

            # LayeredCortex: Use L2/3 firing rate as proxy for processing quality
            # High activity = engaged processing, low = unresponsive
            elif (hasattr(cortex, 'state') and
                  hasattr(cortex.state, 'l23_spikes') and
                  cortex.state.l23_spikes is not None):
                from thalia.components.coding.spike_utils import compute_firing_rate
                l23_activity = compute_firing_rate(cortex.state.l23_spikes)
                # Map [0, 1] to [-0.5, 0.5] - less weight than free energy
                cortex_reward = (l23_activity - 0.5)
                reward += cortex_reward
                n_sources += 1

        # =====================================================================
        # 2. HIPPOCAMPUS PATTERN COMPLETION (memory recall quality)
        # =====================================================================
        # High pattern similarity = successful memory retrieval = reward
        # Biology: VTA observes CA1 output activity. Strong coherent firing = successful recall.
        # We infer similarity from CA1 spike rate (observable signal).
        if 'hippocampus' in self.components:
            hippocampus = self.components['hippocampus']
            if (hasattr(hippocampus, 'state') and
                hasattr(hippocampus.state, 'ca1_spikes') and
                hippocampus.state.ca1_spikes is not None):

                # CA1 firing rate as proxy for retrieval quality
                # High rate = strong recall, low rate = weak/no recall
                from thalia.components.coding.spike_utils import compute_firing_rate
                ca1_activity = compute_firing_rate(hippocampus.state.ca1_spikes)

                # Map CA1 activity [0, 1] to reward [-1, 1]
                # 0.5 activity = neutral (0 reward), >0.5 = positive, <0.5 = negative
                hippo_reward = 2.0 * ca1_activity - 1.0
                # Weight slightly less than cortex (memory is secondary to prediction)
                reward += 0.5 * hippo_reward
                n_sources += 1

        # =====================================================================
        # Average across sources
        # =====================================================================
        if n_sources > 0:
            reward = reward / n_sources
        else:
            # No signals → assume moderate intrinsic reward
            reward = 0.0

        return max(-1.0, min(1.0, reward))

    def _compute_uncertainty(self) -> float:
        """Compute current task uncertainty for arousal modulation.

        Uncertainty drives norepinephrine release from locus coeruleus.
        High uncertainty → high arousal → increased neural gain.

        Sources:
        1. Prediction error magnitude (cortex)
        2. Value estimate variance (striatum)
        3. Novelty detection

        Returns:
            Uncertainty estimate in [0, 1]
        """
        uncertainty = 0.0
        n_sources = 0

        # Cortex prediction error as uncertainty proxy
        if 'cortex' in self.components:
            cortex = self.components['cortex']
            if (hasattr(cortex, 'state') and
                hasattr(cortex.state, 'free_energy') and
                cortex.state.free_energy is not None):
                free_energy = cortex.state.free_energy
                # High FE → high uncertainty
                cortex_uncertainty = min(1.0, free_energy / 10.0)
                uncertainty += cortex_uncertainty
                n_sources += 1

        # Average across sources
        if n_sources > 0:
            uncertainty = uncertainty / n_sources
        else:
            # No signals → assume moderate uncertainty
            uncertainty = 0.3

        return max(0.0, min(1.0, uncertainty))

    def _compute_cognitive_load(self) -> float:
        """Compute current cognitive load from PFC working memory usage.

        Phase 3 functionality: Cognitive load drives temporal discounting.
        High working memory usage → high load → more impulsive choices.

        Returns:
            Cognitive load (0-1), where 1 = maximum capacity
        """
        if 'pfc' not in self.components:
            return 0.0

        pfc = self.components['pfc']

        # Measure PFC activity from spike output, not internal WM state
        if not hasattr(pfc, 'state') or pfc.state is None or not hasattr(pfc.state, 'spikes'):
            return 0.0

        if pfc.state.spikes is None:
            return 0.0

        # Measure working memory load from sustained spike activity
        from thalia.components.coding.spike_utils import compute_firing_rate
        wm_activity = compute_firing_rate(pfc.state.spikes)

        # Also consider number of active goals (if hierarchical goals enabled)
        goal_load = 0.0
        if hasattr(pfc, 'goal_manager') and pfc.goal_manager is not None:
            n_active = len(pfc.goal_manager.active_goals)
            max_goals = pfc.goal_manager.config.max_active_goals
            goal_load = n_active / max(max_goals, 1)

        # Combine WM activity and goal count (weighted average)
        cognitive_load = 0.7 * wm_activity + 0.3 * goal_load

        # Clamp to [0, 1]
        return max(0.0, min(1.0, cognitive_load))

    def _compute_prediction_error(self) -> float:
        """Compute current prediction error for ACh modulation.

        Prediction error drives ACh release from nucleus basalis.
        High PE → novelty → ACh burst → encoding mode.

        Sources:
        1. Cortex free energy (prediction error magnitude)
        2. Hippocampus retrieval mismatch

        Returns:
            Prediction error estimate in [0, 1]
        """
        prediction_error = 0.0
        n_sources = 0

        # Cortex predictive coding error
        if 'cortex' in self.components:
            cortex = self.components['cortex']
            if hasattr(cortex, 'state') and hasattr(cortex.state, 'free_energy'):
                free_energy = cortex.state.free_energy
                # Map FE to [0, 1]: 0 → 0, 5 → 0.5, 10+ → 1.0
                cortex_pe = min(1.0, free_energy / 10.0)
                prediction_error += cortex_pe
                n_sources += 1

        # Average across sources
        if n_sources > 0:
            prediction_error = prediction_error / n_sources
        else:
            # No signals → assume low PE (familiar context)
            prediction_error = 0.2

        return max(0.0, min(1.0, prediction_error))

    def consolidate(
        self,
        n_cycles: int = 5,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Perform memory consolidation (replay) automatically.

        This simulates sleep/offline replay where hippocampus replays stored
        episodes to strengthen cortical representations. Each replayed experience
        triggers actual learning via dopamine delivery.

        Biologically accurate consolidation:
        1. Sample experiences from hippocampal memory
        2. Replay state through brain (reactivate patterns)
        3. Deliver stored reward → dopamine → striatum learning
        4. HER automatically augments if enabled

        This is why consolidation works: replayed experiences trigger the SAME
        learning signals as real experiences, strengthening action values offline.

        Args:
            n_cycles: Number of replay cycles to run
            batch_size: Number of experiences per cycle
            verbose: Whether to print progress

        Returns:
            Dict with consolidation statistics

        Raises:
            ValueError: If hippocampus not present in brain
        """
        # Sync last_action to container before consolidation
        self._last_action_container[0] = self._last_action

        # Delegate to consolidation manager if available
        if self.consolidation_manager is not None:
            return self.consolidation_manager.consolidate(
                n_cycles=n_cycles,
                batch_size=batch_size,
                verbose=verbose,
                last_action_holder=self._last_action_container,
            )

        # Fallback: simplified consolidation without manager
        # This path is used when brain doesn't have all required components
        if "hippocampus" not in self.components:
            raise ValueError(
                "Hippocampus component required for consolidation. "
                "Brain must include 'hippocampus' component."
            )

        hippocampus = self.components["hippocampus"]

        # Check for consolidation support (either HER or standard replay)
        has_her_replay = hasattr(hippocampus, "sample_her_replay_batch")
        has_standard_replay = hasattr(hippocampus, "sample_replay_batch")

        if not has_her_replay and not has_standard_replay:
            raise AttributeError(
                f"Hippocampus ({type(hippocampus).__name__}) does not support consolidation. "
                f"Ensure hippocampus implements sample_her_replay_batch() or sample_replay_batch()."
            )

        stats = {
            "cycles_completed": 0,
            "total_replayed": 0,
            "experiences_learned": 0,
            "her_enabled": has_her_replay,
        }

        # Enter consolidation mode if HER enabled
        if has_her_replay and hasattr(hippocampus, "enter_consolidation_mode"):
            hippocampus.enter_consolidation_mode()
            if verbose:
                her_diag = hippocampus.get_her_diagnostics()
                print(f"  HER: {her_diag['n_episodes']} episodes, {her_diag['n_transitions']} transitions")

        # Run replay cycles
        for cycle in range(n_cycles):
            # Sample replay batch from hippocampal memory
            if has_her_replay:
                # HER-enabled hippocampus (preferred)
                batch = hippocampus.sample_her_replay_batch(batch_size=batch_size)
            else:
                # Standard replay
                batch = hippocampus.sample_replay_batch(batch_size=batch_size)

            if not batch:
                if verbose:
                    print(f"  Cycle {cycle + 1}/{n_cycles}: No experiences to replay")
            else:
                stats["total_replayed"] += len(batch)

                # Replay each experience
                for experience in batch:
                    # Replay state through brain
                    state = experience.get("state")
                    if state is not None:
                        self.forward({"thalamus": state}, n_timesteps=5)

                    # Deliver stored reward for learning
                    reward = experience.get("reward", 0.0)
                    action = experience.get("action")

                    if action is not None and "striatum" in self.components:
                        self._last_action_container[0] = action
                        self.deliver_reward(external_reward=reward)
                        stats["experiences_learned"] += 1

                if verbose:
                    print(f"  Cycle {cycle + 1}/{n_cycles}: Replayed {len(batch)} experiences, {stats['experiences_learned']} learned")

            # Increment cycle count regardless of batch size
            stats["cycles_completed"] += 1

        # Exit consolidation mode if HER enabled
        if has_her_replay and hasattr(hippocampus, "exit_consolidation_mode"):
            hippocampus.exit_consolidation_mode()

        return stats

    # =========================================================================
    # HEALTH & CRITICALITY MONITORING (Phase 1.7.6)
    # =========================================================================

    def check_health(self) -> "HealthReport":
        """Check network health and detect pathological states.

        Returns health report with detected issues, severity, and recommendations.
        Compatible with EventDrivenBrain's health monitoring interface.

        Returns:
            HealthReport object with:
                - is_healthy: bool
                - issues: List[IssueReport]
                - summary: str
                - overall_severity: float

        Example:
            health = brain.check_health()
            if not health.is_healthy:
                for issue in health.issues:
                    print(f"{issue.issue_type}: {issue.description}")
        """
        from thalia.diagnostics.health_monitor import HealthReport

        # Get comprehensive diagnostics for health check
        diagnostics = self.get_diagnostics()

        # Run health check through HealthMonitor and return HealthReport directly
        return self.health_monitor.check_health(diagnostics)

    # =========================================================================
    # DIAGNOSTICS & GROWTH (Phase 1.6.4)
    # =========================================================================

    def check_growth_needs(self) -> Dict[str, Any]:
        """Check if any components need growth based on capacity metrics.

        Analyzes utilization, saturation, and activity patterns to determine
        if components need more capacity. Compatible with EventDrivenBrain
        growth detection.

        Returns:
            Dict mapping component names to growth recommendations

        Example:
            report = brain.check_growth_needs()
            for name, metrics in report.items():
                if metrics['growth_recommended']:
                    print(f"{name}: {metrics['growth_reason']}")

        Note:
            Requires components to implement capacity metrics. If a component
            doesn't support metrics, it will be skipped.
        """
        from thalia.coordination.growth import GrowthManager

        growth_report = {}

        # Check each component
        for name, component in self.components.items():
            # Skip if component doesn't support growth analysis
            if not hasattr(component, "config"):
                continue

            # Use GrowthManager to analyze capacity
            manager = GrowthManager(region_name=name)

            try:
                metrics = manager.get_capacity_metrics(component)

                growth_report[name] = {
                    "firing_rate": metrics.firing_rate,
                    "weight_saturation": metrics.weight_saturation,
                    "synapse_usage": metrics.synapse_usage,
                    "neuron_count": metrics.neuron_count,
                    "growth_recommended": metrics.growth_recommended,
                    "growth_reason": metrics.growth_reason,
                }
            except (AttributeError, TypeError):
                # Component doesn't support capacity metrics
                growth_report[name] = {
                    "growth_recommended": False,
                    "growth_reason": "Component does not support growth metrics",
                }

        return growth_report

    def auto_grow(self, threshold: float = 0.8) -> Dict[str, int]:
        """Automatically grow components that need more capacity.

        When a component grows, also updates all connected pathways to maintain
        proper connectivity. This ensures pathway dimensions stay synchronized
        with component sizes.

        Args:
            threshold: Capacity threshold for triggering growth (0.0-1.0)
                      Currently not used - growth based on recommendations

        Returns:
            Dict mapping component names to number of neurons added

        Example:
            # Check and grow if needed
            growth = brain.auto_grow(threshold=0.8)
            if growth:
                print(f"Grew: {growth}")

        Note:
            Growth history is not tracked in DynamicBrain (unlike EventDrivenBrain).
            Add tracking if needed for your use case.
        """
        growth_actions = {}
        report = self.check_growth_needs()

        for component_name, metrics in report.items():
            if not metrics["growth_recommended"]:
                continue

            component = self.components[component_name]

            # Skip if component doesn't support growth
            if not hasattr(component, "grow_output"):
                continue

            # Calculate growth amount (10% or minimum 8 neurons)
            current_size = getattr(component.config, "n_output", None)
            if current_size is None:
                continue

            growth_amount = max(int(current_size * 0.1), 8)

            # Record metrics before growth
            metrics_before = metrics.copy()

            # Grow the component
            component.grow_output(n_new=growth_amount)
            growth_actions[component_name] = growth_amount

            # Grow connected pathways via PathwayManager
            self.pathway_manager.grow_connected_pathways(
                component_name=component_name,
                growth_amount=growth_amount,
            )

            # Record growth event in history
            from thalia.coordination.growth import GrowthEvent
            from datetime import datetime, timezone

            # Get metrics after growth (simplified - component should re-compute)
            new_size = getattr(component.config, "n_output", None)
            metrics_after = {"n_output": new_size} if new_size else {}

            event = GrowthEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                component_name=component_name,
                component_type="region",
                event_type="grow_output",
                n_neurons_added=growth_amount,
                n_synapses_added=0,  # Pathways track their own growth
                reason=f"Capacity threshold exceeded: utilization={metrics_before.get('utilization', 0):.2f}",
                metrics_before=metrics_before,
                metrics_after=metrics_after,
            )
            self._growth_history.append(event)

        return growth_actions

    def _grow_connected_pathways(self, component_name: str, growth_amount: int) -> None:
        """Grow all pathways connected to a component that has grown.

        Deprecated: Use pathway_manager.grow_connected_pathways() instead.
        Kept for backward compatibility.

        Args:
            component_name: Name of component that grew
            growth_amount: Number of neurons added to component
        """
        # Delegate to PathwayManager
        self.pathway_manager.grow_connected_pathways(
            component_name=component_name,
            growth_amount=growth_amount,
        )

    # =========================================================================
    # COMPONENT GRAPH UTILITIES
    # =========================================================================

    def _gather_component_inputs(
        self,
        component_name: str,
        external_input: Optional[torch.Tensor],
        prior_outputs: Dict[str, Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Collect inputs for a component from connections and external sources.

        Args:
            component_name: Name of component receiving inputs
            external_input: External input (e.g., sensory data), if any
            prior_outputs: Outputs from upstream components

        Returns:
            Combined input tensor, or None if no inputs available
        """
        inputs = []

        # Check for external input (e.g., sensory input to thalamus)
        if external_input is not None:
            inputs.append(external_input)

        # Gather inputs from upstream connections
        for conn_key, pathway in self.connections.items():
            # Connection keys are tuples: (source, target)
            src, tgt = conn_key

            # Is this connection targeting our component?
            if tgt == component_name:
                upstream_output = prior_outputs.get(src)
                if upstream_output is not None:
                    # Route through pathway
                    pathway_output = pathway.forward(upstream_output)
                    inputs.append(pathway_output)

        # Combine inputs
        if not inputs:
            # No inputs - component is source node or receives no input this timestep
            return None
        elif len(inputs) == 1:
            return inputs[0]
        else:
            # Multiple inputs - concatenate along feature dimension
            # TODO: Make this configurable per component (concatenate vs sum)
            return torch.cat(inputs, dim=-1)

    def get_component(self, name: str) -> NeuralComponent:
        """Get component by name.

        Args:
            name: Component name

        Returns:
            Component instance

        Raises:
            KeyError: If component name not found
        """
        if name not in self.components:
            available = list(self.components.keys())
            raise KeyError(
                f"Component '{name}' not found. Available: {available}"
            )
        # Cast from Module to NeuralComponent (all our components are NeuralComponent)
        return self.components[name]  # type: ignore[return-value]

    def add_component(
        self,
        name: str,
        component: NeuralComponent,
    ) -> None:
        """Dynamically add a component (e.g., during growth).

        Args:
            name: Component instance name
            component: Component to add

        Raises:
            ValueError: If component name already exists
        """
        if name in self.components:
            raise ValueError(f"Component '{name}' already exists")

        self.components[name] = component
        self._topology[name] = []

        # Initialize spike count tracking for new component
        if hasattr(self, "_spike_counts"):
            self._spike_counts[name] = 0

        # Invalidate cached execution order
        self._execution_order = None

    def add_connection(
        self,
        source: str,
        target: str,
        pathway: NeuralComponent,
    ) -> None:
        """Dynamically add a connection.

        Args:
            source: Source component name
            target: Target component name
            pathway: Pathway component

        Raises:
            ValueError: If source or target component doesn't exist
        """
        if source not in self.components:
            raise ValueError(f"Source component '{source}' not found")
        if target not in self.components:
            raise ValueError(f"Target component '{target}' not found")

        # Store with tuple key
        self.connections[(source, target)] = pathway

        # Also register in ModuleDict for parameter tracking
        conn_key = f"{source}_to_{target}"
        self._connection_modules[conn_key] = pathway

        # Update topology
        if source in self._topology:
            if target not in self._topology[source]:
                self._topology[source].append(target)

        # Invalidate cached execution order
        self._execution_order = None

    def _schedule_downstream_events(
        self,
        source: str,
        output_spikes: torch.Tensor,
        current_time: float,
    ) -> None:
        """Schedule spike events to downstream components with axonal delays.

        For multi-source pathways, forwards immediately when ANY source fires.
        Missing sources will be filled with zeros by the pathway (biologically
        plausible - no spikes from that source yet).

        Args:
            source: Source component name
            output_spikes: Output spikes from source
            current_time: Current simulation time
        """
        from thalia.pathways.multi_source_pathway import MultiSourcePathway

        # Find all connections from this source
        for (src, tgt), pathway in self.connections.items():
            if src == source:
                if isinstance(pathway, MultiSourcePathway):
                    # Multi-source pathway - update this source's output in buffer
                    if tgt not in self._multi_source_buffers:
                        self._multi_source_buffers[tgt] = {}

                    # Extract port-specific output if specified
                    conn_spec = self._connection_specs.get((src, tgt))
                    if conn_spec and hasattr(conn_spec, 'source_port') and conn_spec.source_port:
                        # Get source component to extract port data
                        source_comp = self.components[src]
                        port_output = self._extract_port_output(source_comp, output_spikes, conn_spec.source_port)
                    else:
                        port_output = output_spikes

                    # Store this source's output (port-extracted)
                    self._multi_source_buffers[tgt][source] = port_output

                    # Forward immediately with ALL currently available sources
                    # (pathway will use zeros for any missing sources)
                    delay_ms = getattr(pathway.config, 'axonal_delay_ms', 1.0)

                    # Forward with dict of currently available source outputs
                    pathway_output = pathway.forward(self._multi_source_buffers[tgt])

                    # Schedule event to target
                    available_sources = list(self._multi_source_buffers[tgt].keys())
                    event = Event(
                        time=current_time + delay_ms,
                        event_type=EventType.SPIKE,
                        source=f"multi:{','.join(available_sources)}",
                        target=tgt,
                        payload=SpikePayload(spikes=pathway_output),
                    )
                    self._scheduler.schedule(event)

                else:
                    # Single-source pathway - process immediately
                    delay_ms = getattr(pathway.config, 'axonal_delay_ms', 1.0)

                    # Extract port-specific output if specified
                    conn_spec = self._connection_specs.get((src, tgt))
                    if conn_spec and hasattr(conn_spec, 'source_port') and conn_spec.source_port:
                        # Get source component to extract port data
                        source_comp = self.components[src]
                        port_output = self._extract_port_output(source_comp, output_spikes, conn_spec.source_port)
                    else:
                        port_output = output_spikes

                    # Route spikes through pathway
                    pathway_output = pathway.forward(port_output)

                    # Schedule event to target with delay
                    event = Event(
                        time=current_time + delay_ms,
                        event_type=EventType.SPIKE,
                        source=source,
                        target=tgt,
                        payload=SpikePayload(spikes=pathway_output),
                    )
                    self._scheduler.schedule(event)

    def _extract_port_output(self, component: NeuralComponent, output: torch.Tensor, port: str) -> torch.Tensor:
        """Extract port-specific output from component output.

        Args:
            component: Source component
            output: Full component output
            port: Port name ('l23', 'l5', 'l4')

        Returns:
            Sliced output for the specified port
        """
        # For LayeredCortex: output is [L2/3, L5] concatenated
        if hasattr(component, 'l23_size') and hasattr(component, 'l5_size'):
            if port == "l23":
                return output[:component.l23_size]
            elif port == "l5":
                return output[component.l23_size:]
            elif port == "l4":
                # L4 is internal, not in output
                raise ValueError("L4 is not part of cortex output")

        raise ValueError(f"Component {type(component).__name__} does not support port '{port}'")

    @property
    def regions(self) -> Dict[str, Any]:
        """Get all brain regions (EventDrivenBrain compatibility).

        Returns:
            Dict mapping region names to component instances

        Note:
            This is an alias for `components` for API compatibility.
        """
        return dict(self.components)

    def reset_state(self) -> None:
        """Reset all component and system states."""
        for component in self.components.values():
            component.reset_state()

        for pathway in self.connections.values():
            pathway.reset_state()

        # Reset oscillators
        self.oscillators.reset()

        # Reset spike counts
        self._spike_counts = {name: 0 for name in self.components.keys()}

        self._current_time = 0.0

    def _collect_striatum_diagnostics(self) -> StriatumDiagnostics:
        """Collect structured diagnostics from striatum.

        Returns:
            StriatumDiagnostics dataclass with per-action weights and metrics
        """
        if 'striatum' not in self.components:
            # Return empty diagnostics if striatum not present
            return StriatumDiagnostics(
                d1_per_action=[],
                d2_per_action=[],
                net_per_action=[],
                d1_elig_per_action=[],
                d2_elig_per_action=[],
                last_action=None,
                exploring=False,
                exploration_prob=0.0,
                action_counts=[],
                total_trials=0,
            )

        striatum = self.components['striatum']

        # Get n_actions from striatum
        n_actions = getattr(striatum, 'n_actions', 2)
        neurons_per = getattr(striatum, 'neurons_per_action', 10)

        # Per-action weight means
        d1_per_action = []
        d2_per_action = []
        net_per_action = []

        for a in range(n_actions):
            start = a * neurons_per
            end = start + neurons_per

            # Safe access to weights (might be None)
            if hasattr(striatum, 'd1_weights') and striatum.d1_weights is not None:
                d1_mean = striatum.d1_weights[start:end].mean().item()
            else:
                d1_mean = 0.0

            if hasattr(striatum, 'd2_weights') and striatum.d2_weights is not None:
                d2_mean = striatum.d2_weights[start:end].mean().item()
            else:
                d2_mean = 0.0

            d1_per_action.append(d1_mean)
            d2_per_action.append(d2_mean)
            net_per_action.append(d1_mean - d2_mean)

        # Eligibility traces
        d1_elig_per_action = []
        d2_elig_per_action = []
        for a in range(n_actions):
            start = a * neurons_per
            end = start + neurons_per
            # Safe access to eligibility traces (might be None)
            if hasattr(striatum, 'd1_eligibility') and striatum.d1_eligibility is not None:
                d1_elig_per_action.append(striatum.d1_eligibility[start:end].abs().mean().item())
            else:
                d1_elig_per_action.append(0.0)
            if hasattr(striatum, 'd2_eligibility') and striatum.d2_eligibility is not None:
                d2_elig_per_action.append(striatum.d2_eligibility[start:end].abs().mean().item())
            else:
                d2_elig_per_action.append(0.0)

        # UCB and exploration - safe access to potentially missing attributes
        action_counts = []
        total_trials = 0
        if hasattr(striatum, '_action_counts') and striatum._action_counts is not None:
            action_counts = [int(c) for c in striatum._action_counts.tolist()]
        if hasattr(striatum, '_total_trials'):
            total_trials = int(getattr(striatum, '_total_trials', 0))
        exploration_prob = getattr(striatum, '_last_exploration_prob', 0.0)

        return StriatumDiagnostics(
            d1_per_action=d1_per_action,
            d2_per_action=d2_per_action,
            net_per_action=net_per_action,
            d1_elig_per_action=d1_elig_per_action,
            d2_elig_per_action=d2_elig_per_action,
            last_action=self._last_action,
            exploring=getattr(striatum, '_last_exploring', False),
            exploration_prob=exploration_prob,
            action_counts=action_counts,
            total_trials=total_trials,
        )

    def _collect_hippocampus_diagnostics(self) -> HippocampusDiagnostics:
        """Collect structured diagnostics from hippocampus.

        Returns:
            HippocampusDiagnostics dataclass with layer activity and memory metrics
        """
        if 'hippocampus' not in self.components:
            # Return empty diagnostics if hippocampus not present
            return HippocampusDiagnostics(
                ca1_total_spikes=0.0,
                ca1_normalized=0.0,
                dg_spikes=0.0,
                ca3_spikes=0.0,
                ca1_spikes=0.0,
                n_stored_episodes=0,
            )

        hippo = self.components['hippocampus']

        # CA1 activity (key for match/mismatch)
        ca1_spikes = 0.0
        if hasattr(hippo, 'state') and hippo.state is not None:
            if hasattr(hippo.state, 'ca1_spikes') and hippo.state.ca1_spikes is not None:
                ca1_spikes = hippo.state.ca1_spikes.sum().item()

        # Normalize by hippocampus size
        hippo_size = getattr(hippo.config, 'n_output', 128) if hasattr(hippo, 'config') else 128
        ca1_normalized = ca1_spikes / max(1, hippo_size)

        # Layer activity
        dg_spikes = 0.0
        ca3_spikes = 0.0
        if hasattr(hippo, 'state') and hippo.state is not None:
            if hasattr(hippo.state, 'dg_spikes') and hippo.state.dg_spikes is not None:
                dg_spikes = hippo.state.dg_spikes.sum().item()
            if hasattr(hippo.state, 'ca3_spikes') and hippo.state.ca3_spikes is not None:
                ca3_spikes = hippo.state.ca3_spikes.sum().item()

        # Memory metrics
        n_stored = 0
        if hasattr(hippo, 'episode_buffer'):
            n_stored = len(hippo.episode_buffer)

        return HippocampusDiagnostics(
            ca1_total_spikes=ca1_spikes,
            ca1_normalized=ca1_normalized,
            dg_spikes=dg_spikes,
            ca3_spikes=ca3_spikes,
            ca1_spikes=ca1_spikes,
            n_stored_episodes=n_stored,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information from all subsystems.

        Returns diagnostic metrics from:
        - Components (regions)
        - Pathways
        - Oscillators
        - Neuromodulators
        - Planning systems (if enabled)
        - Criticality (if enabled)

        Returns:
            Dict containing diagnostic metrics from all brain subsystems

        Example:
            diag = brain.get_diagnostics()
            print(diag['components']['cortex'])  # Component diagnostics
            print(diag['pathways'])  # Pathway diagnostics
            print(diag['oscillators'])  # Oscillator phases/signals
        """
        diagnostics = {
            "components": {},
            "spike_counts": {},
            "pathways": {},
            "oscillators": {},
            "neuromodulators": {},
        }

        # Component diagnostics
        for name, component in self.components.items():
            if hasattr(component, "get_diagnostics"):
                diagnostics["components"][name] = component.get_diagnostics()

            # Collect spike counts for health monitoring
            if hasattr(component, "state") and component.state is not None:
                if hasattr(component.state, "spikes") and component.state.spikes is not None:
                    diagnostics["spike_counts"][name] = component.state.spikes.sum().item()

        # Pathway diagnostics (from PathwayManager)
        diagnostics["pathways"] = self.pathway_manager.get_diagnostics()

        # Oscillator diagnostics
        diagnostics["oscillators"] = {
            'delta': {
                'phase': self.oscillators.delta.phase,
                'frequency_hz': self.oscillators.delta.frequency_hz,
                'signal': self.oscillators.delta.signal,
            },
            'theta': {
                'phase': self.oscillators.theta.phase,
                'frequency_hz': self.oscillators.theta.frequency_hz,
                'signal': self.oscillators.theta.signal,
            },
            'alpha': {
                'phase': self.oscillators.alpha.phase,
                'frequency_hz': self.oscillators.alpha.frequency_hz,
                'signal': self.oscillators.alpha.signal,
            },
            'beta': {
                'phase': self.oscillators.beta.phase,
                'frequency_hz': self.oscillators.beta.frequency_hz,
                'signal': self.oscillators.beta.signal,
            },
            'gamma': {
                'phase': self.oscillators.gamma.phase,
                'frequency_hz': self.oscillators.gamma.frequency_hz,
                'signal': self.oscillators.gamma.signal,
            },
            'ripple': {
                'phase': self.oscillators.ripple.phase,
                'frequency_hz': self.oscillators.ripple.frequency_hz,
                'signal': self.oscillators.ripple.signal,
            },
        }

        # Neuromodulator diagnostics
        diagnostics["neuromodulators"] = self.neuromodulator_manager.get_diagnostics()

        # Planning diagnostics (if enabled)
        if self.mental_simulation is not None:
            diagnostics["planning"] = {
                "mental_simulation_enabled": True,
                "dyna_enabled": self.dyna_planner is not None,
            }

        # Criticality diagnostics (if enabled)
        if self.criticality_monitor is not None:
            diagnostics["criticality"] = self.criticality_monitor.get_diagnostics()

        return diagnostics

    def get_structured_diagnostics(self) -> BrainSystemDiagnostics:
        """Get fully structured diagnostics as a dataclass.

        Returns:
            BrainSystemDiagnostics with all component data

        Example:
            diag = brain.get_structured_diagnostics()
            print(f"Striatum D1 for action 0: {diag.striatum.d1_per_action[0]}")
            print(f"Hippocampus CA1 spikes: {diag.hippocampus.ca1_total_spikes}")
        """
        # Get trial metadata if available
        trial_num = 0
        is_match = getattr(self, '_last_is_match', False)
        correct = getattr(self, '_last_correct', False)

        return BrainSystemDiagnostics(
            trial_num=trial_num,
            is_match=is_match,
            selected_action=self._last_action or 0,
            correct=correct,
            striatum=self._collect_striatum_diagnostics(),
            hippocampus=self._collect_hippocampus_diagnostics(),
        )

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns:
            Dict containing all component states, neuromodulator states, and metadata

        Note:
            Uses "regions" key (not "components") for CheckpointManager compatibility.
            DynamicBrain's components are stored as regions in the checkpoint format.
        """
        state = {
            "global_config": self.global_config,
            "current_time": self._current_time,
            "topology": self._topology,
            "regions": {},  # Use "regions" key for CheckpointManager compatibility
            "pathways": {},
            "oscillators": {},
            "neuromodulators": {},
            "config": {},  # Add config for CheckpointManager validation
            "growth_history": [event.to_dict() for event in self._growth_history],
        }

        # Add config sizes for CheckpointManager validation
        for attr in ["input_size", "cortex_size", "hippocampus_size", "pfc_size", "n_actions"]:
            if hasattr(self.config, attr):
                state["config"][attr] = getattr(self.config, attr)

        # Get component states (stored as "regions" for compatibility)
        for name, component in self.components.items():
            state["regions"][name] = component.get_full_state()

        # Get pathway states via PathwayManager
        state["pathways"] = self.pathway_manager.get_state()

        # Get oscillator states
        state["oscillators"] = {
            'delta': self.oscillators.delta.get_state(),
            'theta': self.oscillators.theta.get_state(),
            'alpha': self.oscillators.alpha.get_state(),
            'beta': self.oscillators.beta.get_state(),
            'gamma': self.oscillators.gamma.get_state(),
            'ripple': self.oscillators.ripple.get_state(),
        }

        # Get neuromodulator states using proper get_state() methods
        if self.neuromodulator_manager.vta is not None:
            if hasattr(self.neuromodulator_manager.vta, 'get_state'):
                state["neuromodulators"]["vta"] = self.neuromodulator_manager.vta.get_state()
            else:
                # Fallback for VTA without get_state()
                state["neuromodulators"]["vta"] = {
                    "global_dopamine": self.neuromodulator_manager.vta._global_dopamine,
                    "tonic_dopamine": self.neuromodulator_manager.vta._tonic_dopamine,
                    "phasic_dopamine": self.neuromodulator_manager.vta._phasic_dopamine,
                }
        if self.neuromodulator_manager.locus_coeruleus is not None:
            if hasattr(self.neuromodulator_manager.locus_coeruleus, 'get_state'):
                state["neuromodulators"]["locus_coeruleus"] = self.neuromodulator_manager.locus_coeruleus.get_state()
            else:
                # Fallback for LC without get_state()
                state["neuromodulators"]["locus_coeruleus"] = {
                    "norepinephrine": self.neuromodulator_manager.locus_coeruleus.get_norepinephrine(apply_homeostasis=False),
                }
        if self.neuromodulator_manager.nucleus_basalis is not None:
            if hasattr(self.neuromodulator_manager.nucleus_basalis, 'get_state'):
                state["neuromodulators"]["nucleus_basalis"] = self.neuromodulator_manager.nucleus_basalis.get_state()
            else:
                # Fallback for NB without get_state()
                state["neuromodulators"]["nucleus_basalis"] = {
                    "acetylcholine": self.neuromodulator_manager.nucleus_basalis.get_acetylcholine(apply_homeostasis=False),
                }

        return state

    def save_checkpoint(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Save checkpoint (wrapper for CheckpointManager.save).

        This is a convenience wrapper that provides EventDrivenBrain-compatible
        API for checkpoint saving.

        Args:
            path: Path to save checkpoint
            metadata: Optional metadata dict (epoch, loss, etc.)
            **kwargs: Additional arguments passed to CheckpointManager.save()
                     (compression, compression_level, precision_policy)

        Returns:
            Dict containing save info (size, time, components saved)

        Example:
            >>> brain.save_checkpoint(
            ...     "checkpoints/epoch_100.ckpt",
            ...     metadata={"epoch": 100, "loss": 0.42}
            ... )
        """
        return self.checkpoint_manager.save(path, metadata, **kwargs)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint (wrapper for CheckpointManager.load).

        This is a convenience wrapper that provides EventDrivenBrain-compatible
        API for checkpoint loading.

        Args:
            path: Path to checkpoint file
            device: Device to load to (None = use brain's device)
            strict: Whether to enforce strict config matching

        Returns:
            Dict containing load info (metadata, load time, etc.)

        Example:
            >>> brain.load_checkpoint("checkpoints/epoch_100.ckpt")
        """
        return self.checkpoint_manager.load(path, device, strict)

    def get_growth_history(self) -> List[Dict[str, Any]]:
        """Get complete growth history.

        Returns:
            List of growth events as dicts (serializable)

        Example:
            >>> history = brain.get_growth_history()
            >>> for event in history:
            ...     print(f"{event['timestamp']}: {event['component_name']} "
            ...           f"grew by {event['n_neurons_added']} neurons")
        """
        return [event.to_dict() for event in self._growth_history]

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dict from get_full_state()

        Note:
            Supports both "regions" (new format) and "components" (old format)
            for backward compatibility.

            Also supports both DynamicBrain format (current_time at top level)
            and EventDrivenBrain format (current_time inside scheduler dict).
        """
        # Handle current_time in both formats
        if "current_time" in state:
            self._current_time = state["current_time"]
        elif "scheduler" in state and "current_time" in state["scheduler"]:
            self._current_time = state["scheduler"]["current_time"]
        else:
            self._current_time = 0.0  # Default if not present

        # Handle topology (may not be present in EventDrivenBrain format)
        self._topology = state.get("topology", self._topology)

        # Load component states (support both "regions" and "components" keys)
        component_states = state.get("regions", state.get("components", {}))
        for name, component_state in component_states.items():
            if name in self.components:
                self.components[name].load_full_state(component_state)

        # Load pathway states via PathwayManager
        if "pathways" in state:
            self.pathway_manager.load_state(state["pathways"])
        elif "connections" in state:
            # Backward compatibility: old format used "connections" directly
            for conn_key, pathway_state in state["connections"].items():
                if conn_key in self.connections:
                    self.connections[conn_key].load_full_state(pathway_state)

        # Load oscillator states
        if "oscillators" in state:
            osc_state = state["oscillators"]
            if "delta" in osc_state:
                self.oscillators.delta.set_state(osc_state["delta"])
            if "theta" in osc_state:
                self.oscillators.theta.set_state(osc_state["theta"])
            if "alpha" in osc_state:
                self.oscillators.alpha.set_state(osc_state["alpha"])
            if "beta" in osc_state:
                self.oscillators.beta.set_state(osc_state["beta"])
            if "gamma" in osc_state:
                self.oscillators.gamma.set_state(osc_state["gamma"])
            if "ripple" in osc_state:
                self.oscillators.ripple.set_state(osc_state["ripple"])

        # Load neuromodulator states (if present)
        if "neuromodulators" in state:
            neuromod_state = state["neuromodulators"]
            if "vta" in neuromod_state and self.neuromodulator_manager.vta is not None:
                if hasattr(self.neuromodulator_manager.vta, 'set_state'):
                    self.neuromodulator_manager.vta.set_state(neuromod_state["vta"])
                else:
                    # Fallback for VTA without set_state()
                    vta_state = neuromod_state["vta"]
                    self.neuromodulator_manager.vta._tonic_dopamine = vta_state.get("tonic_dopamine", 0.0)
                    self.neuromodulator_manager.vta._phasic_dopamine = vta_state.get("phasic_dopamine", 0.0)
                    self.neuromodulator_manager.vta._global_dopamine = vta_state.get("global_dopamine", 0.0)
            if "locus_coeruleus" in neuromod_state and self.neuromodulator_manager.locus_coeruleus is not None:
                if hasattr(self.neuromodulator_manager.locus_coeruleus, 'set_state'):
                    self.neuromodulator_manager.locus_coeruleus.set_state(neuromod_state["locus_coeruleus"])
            if "nucleus_basalis" in neuromod_state and self.neuromodulator_manager.nucleus_basalis is not None:
                if hasattr(self.neuromodulator_manager.nucleus_basalis, 'set_state'):
                    self.neuromodulator_manager.nucleus_basalis.set_state(neuromod_state["nucleus_basalis"])

        # Load growth history (if present)
        if "growth_history" in state:
            from thalia.coordination.growth import GrowthEvent
            self._growth_history = [
                GrowthEvent.from_dict(event_dict)
                for event_dict in state["growth_history"]
            ]
        else:
            self._growth_history = []  # Initialize empty if not in checkpoint

        # Invalidate cached execution order
        self._execution_order = None


__all__ = [
    "ComponentSpec",
    "ConnectionSpec",
    "DynamicBrain",
]
