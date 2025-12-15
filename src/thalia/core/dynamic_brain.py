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
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from collections import deque

import torch
import torch.nn as nn

from thalia.regions.base import NeuralComponent
from thalia.events.system import EventScheduler, Event, EventType, SpikePayload
from thalia.events.adapters.base import EventRegionConfig

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
    """
    source: str
    """Source component name"""

    target: str
    """Target component name"""

    pathway_type: str
    """Pathway registry name (e.g., 'spiking_stdp', 'attention')"""

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
    ):
        """Initialize DynamicBrain from component graph.

        Args:
            components: Dict mapping component names to instances
            connections: Dict mapping (source, target) tuples to pathways
            global_config: Global configuration (device, dt_ms, etc.)
            use_parallel: Enable parallel execution (multi-core CPU)
            n_workers: Number of worker processes (default: number of regions)
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

        # =================================================================
        # PARALLEL EXECUTION (Optional - TODO)
        # =================================================================
        # Note: Parallel execution requires region_creators (callables)
        # instead of region instances. This will be implemented when
        # BrainBuilder supports lazy component instantiation.
        self.use_parallel = use_parallel
        self._parallel_executor = None

        if use_parallel:
            raise NotImplementedError(
                "Parallel execution not yet supported for DynamicBrain. "
                "Use EventDrivenBrain for parallel execution, or set use_parallel=False."
            )

        # =================================================================
        # PATHWAY MANAGER (Phase 1.7.1)
        # =================================================================
        # Centralized pathway management for diagnostics and growth coordination
        # Note: PathwayManager expects specific EventDrivenBrain structure
        # For DynamicBrain, we create a simpler adapter that provides same API
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
        # Provides centralized neuromodulation like EventDrivenBrain
        from thalia.neuromodulation.manager import NeuromodulatorManager

        self.neuromodulator_manager = NeuromodulatorManager()

        # Shortcuts for backward compatibility
        self.vta = self.neuromodulator_manager.vta
        self.locus_coeruleus = self.neuromodulator_manager.locus_coeruleus
        self.nucleus_basalis = self.neuromodulator_manager.nucleus_basalis

        # =================================================================
        # REINFORCEMENT LEARNING STATE (Phase 1.6.2)
        # =================================================================
        self._last_action: Optional[int] = None
        self._last_confidence: Optional[float] = None

        # Mutable container for shared last_action state (needed by ConsolidationManager)
        self._last_action_container = [None]

        # =================================================================
        # CONSOLIDATION MANAGER (Phase 1.7.4)
        # =================================================================
        # Note: ConsolidationManager requires EventDrivenBrain's unified config
        # DynamicBrain uses simpler consolidation without manager
        # For full ConsolidationManager support, use EventDrivenBrain
        self.consolidation_manager = None

        # =================================================================
        # CHECKPOINT MANAGER (Phase 1.7.4)
        # =================================================================
        # Centralized checkpoint save/load with compression and validation
        from thalia.io.checkpoint_manager import CheckpointManager

        self.checkpoint_manager = CheckpointManager(
            brain=self,
            default_compression='zstd',  # Match EventDrivenBrain
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
        # EVENT ADAPTER SYSTEM
        # =================================================================
        # For event-driven execution, we wrap components with EventDrivenRegionBase adapters
        # These adapters translate events, apply membrane decay, and route outputs
        self._component_specs: Dict[str, ComponentSpec] = {}
        """Component specifications (set by BrainBuilder after build)"""

        self._event_adapters: Optional[Dict[str, "EventDrivenRegionBase"]] = None
        """Event adapters (lazy-initialized on first event-driven forward)"""

        self._registry: Optional["ComponentRegistry"] = None
        """Component registry reference (for adapter lookup)"""

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

    @property
    def current_time(self) -> float:
        """Get current simulation time in milliseconds.

        Returns:
            Current simulation time in ms
        """
        return self._current_time

    @property
    def adapters(self) -> Dict[str, Any]:
        """Get components dict (alias for CheckpointManager compatibility).

        EventDrivenBrain uses 'adapters', DynamicBrain uses 'components'.
        This property provides compatibility with CheckpointManager.

        Returns:
            Dict mapping component names to components
        """
        return dict(self.components.items())

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
        """
        from thalia.core.brain_builder import BrainBuilder

        sizes = config.brain.sizes
        builder = BrainBuilder(config.global_)

        # Add regions with sizes from config
        # Thalamus is input interface
        builder.add_component(
            "thalamus", "thalamus",
            n_input=sizes.input_size,
            n_output=sizes.input_size
        )

        # Other regions - n_input inferred from connections
        builder.add_component("cortex", "cortex", n_output=sizes.cortex_size)
        builder.add_component("hippocampus", "hippocampus", n_output=sizes.hippocampus_size)
        builder.add_component("pfc", "prefrontal", n_output=sizes.pfc_size)
        builder.add_component("striatum", "striatum", n_output=sizes.n_actions)
        builder.add_component("cerebellum", "cerebellum", n_output=sizes.cortex_size // 2)

        # Add connections (sensorimotor topology)
        builder.connect("thalamus", "cortex", "spiking")
        builder.connect("cortex", "hippocampus", "spiking")
        builder.connect("hippocampus", "cortex", "spiking")  # Bidirectional
        builder.connect("cortex", "pfc", "spiking")
        builder.connect("pfc", "striatum", "spiking")
        builder.connect("striatum", "pfc", "spiking")  # Bidirectional
        builder.connect("pfc", "cerebellum", "spiking")

        return builder.build()

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

    def _topological_sort(self) -> List[str]:
        """Compute topological ordering of components.

        Uses Kahn's algorithm for topological sorting.

        Returns:
            List of component names in execution order

        Raises:
            ValueError: If graph contains cycles
        """
        # Calculate in-degrees
        in_degree = {name: 0 for name in self.components.keys()}
        for neighbors in self._topology.values():
            for neighbor in neighbors:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1

        # Start with nodes that have no dependencies
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)

            # Reduce in-degree of neighbors
            for neighbor in self._topology.get(current, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Check for cycles
        if len(order) != len(self.components):
            raise ValueError(
                f"Component graph contains cycles! "
                f"Processed {len(order)} of {len(self.components)} components. "
                f"Remaining: {set(in_degree.keys()) - set(order)}"
            )

        return order

    def _get_execution_order(self) -> List[str]:
        """Get cached topological execution order.

        Computes order on first call and caches result.
        """
        if self._execution_order is None:
            self._execution_order = self._topological_sort()
        return self._execution_order

    def forward(
        self,
        input_data: Dict[str, torch.Tensor],
        n_timesteps: int,
        use_event_driven: bool = True,
    ) -> Dict[str, Any]:
        """Execute brain for n_timesteps.

        Args:
            input_data: Dict mapping component names to input tensors
                        (e.g., {"thalamus": sensory_input})
            n_timesteps: Number of simulation timesteps
            use_event_driven: Use event-driven execution (default: True, recommended)
                            If False, uses simple synchronous execution

        Returns:
            Dict containing:
                - "outputs": Dict[component_name -> final output]
                - "history": Optional list of timestep outputs
                - "time": Final simulation time

        Example:
            brain = DynamicBrain(components, connections, config)

            sensory_input = torch.randn(128, device=device)
            result = brain.forward(
                input_data={"thalamus": sensory_input},
                n_timesteps=10
            )

            cortex_output = result["outputs"]["cortex"]
        """
        # Choose execution mode
        if self.use_parallel:
            return self._forward_parallel(input_data, n_timesteps)
        elif use_event_driven:
            return self._forward_event_driven(input_data, n_timesteps)
        else:
            return self._forward_synchronous(input_data, n_timesteps)

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

        return {
            "outputs": outputs,
            "time": self._current_time,
        }

    def _forward_parallel(
        self,
        input_data: Dict[str, torch.Tensor],
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Parallel execution across multiple CPU cores.

        Uses ParallelExecutor to distribute components across workers.
        """
        if self._parallel_executor is None:
            raise ValueError("Parallel executor not initialized (use_parallel=False)")

        # Execute in parallel
        outputs = self._parallel_executor.run(
            input_data=input_data,
            n_timesteps=n_timesteps,
        )

        self._current_time += n_timesteps * self.global_config.dt_ms

        return {
            "outputs": outputs,
            "time": self._current_time,
        }

    def _forward_synchronous(
        self,
        input_data: Dict[str, torch.Tensor],
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Simple synchronous execution (no events, no delays).

        This is the simplest execution mode for testing/debugging.
        """
        # Get execution order
        execution_order = self._get_execution_order()

        # Storage for component outputs at each timestep
        outputs = {name: None for name in self.components.keys()}

        # Execute timesteps
        for _ in range(n_timesteps):
            # Process components in topological order
            for component_name in execution_order:
                component = self.components[component_name]

                # Gather inputs for this component
                component_input = self._gather_component_inputs(
                    component_name,
                    input_data.get(component_name),
                    outputs,
                )

                # Execute component forward pass
                if component_input is not None:
                    component_output = component.forward(component_input)
                else:
                    # Component has no inputs (e.g., oscillator, source node)
                    component_output = component.forward(None)

                # Store output
                outputs[component_name] = component_output

            # Advance oscillators and broadcast phases
            self._broadcast_oscillator_phases()

            # Update criticality monitor if enabled (Phase 1.7.6)
            if self.criticality_monitor is not None:
                # Collect all spikes into a single tensor for branching ratio analysis
                all_spikes = []
                for component in self.components.values():
                    if hasattr(component, 'state') and component.state is not None:
                        if hasattr(component.state, 'spikes') and component.state.spikes is not None:
                            all_spikes.append(component.state.spikes)

                if all_spikes:
                    # Concatenate all spikes and update criticality monitor
                    combined_spikes = torch.cat([s.flatten() for s in all_spikes])
                    self.criticality_monitor.update(combined_spikes)

            # Update time
            self._current_time += self.global_config.dt_ms

        return {
            "outputs": outputs,
            "time": self._current_time,
        }

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

        if not hasattr(self, "_last_action") or self._last_action is None:
            raise ValueError(
                "No action has been selected. Call select_action() before deliver_reward()."
            )

        striatum = self.components["striatum"]

        # Compute total reward (external + intrinsic if available)
        # For now, use external reward only (intrinsic rewards in Phase 1.6.3)
        total_reward = external_reward if external_reward is not None else 0.0

        # Deliver reward to striatum for learning
        # Note: striatum.deliver_reward(reward) uses dopamine from VTA via set_dopamine()
        # The actual RPE computation and dopamine-gated learning happens inside striatum
        if hasattr(striatum, "deliver_reward"):
            striatum.deliver_reward(reward=total_reward)
        else:
            raise AttributeError(
                f"Striatum component ({type(striatum).__name__}) does not have "
                f"deliver_reward method. Ensure striatum implements RL interface."
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

        # Update neuromodulator systems
        self.vta.update(dt_ms=self.global_config.dt_ms, intrinsic_reward=intrinsic_reward)
        self.locus_coeruleus.update(dt_ms=self.global_config.dt_ms, uncertainty=uncertainty)
        self.nucleus_basalis.update(dt_ms=self.global_config.dt_ms, prediction_error=prediction_error)

        # Get current neuromodulator levels
        dopamine = self.vta.get_global_dopamine()
        norepinephrine = self.locus_coeruleus.get_norepinephrine()

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
        """Compute intrinsic reward from curiosity/novelty.

        Simplified version: returns 0.0 for now.
        Full implementation with curiosity in later phase.
        """
        return 0.0

    def _compute_uncertainty(self) -> float:
        """Compute uncertainty from prediction errors.

        Simplified version: returns 0.0 for now.
        Full implementation with prediction uncertainty in later phase.
        """
        return 0.0

    def _compute_prediction_error(self) -> float:
        """Compute prediction error magnitude.

        Simplified version: returns 0.0 for now.
        Full implementation with predictive coding in later phase.
        """
        return 0.0

    def consolidate(
        self,
        n_cycles: int = 5,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Perform memory consolidation through hippocampal replay.

        Simulates sleep/offline replay where hippocampus replays stored
        experiences to strengthen cortical and striatal representations.
        This is biologically plausible consolidation similar to EventDrivenBrain.

        Consolidation Process:
            1. Sample experiences from hippocampal memory
            2. Replay states through brain (reactivate patterns)
            3. Deliver stored reward → dopamine → learning
            4. HER augmentation if enabled

        Args:
            n_cycles: Number of replay cycles
            batch_size: Experiences per cycle
            verbose: Print progress

        Returns:
            Dict with consolidation statistics

        Raises:
            ValueError: If hippocampus not present in brain

        Note:
            If ConsolidationManager is available (brain has hippocampus + striatum),
            delegates to manager for full EventDrivenBrain-compatible consolidation.
            Otherwise, falls back to simplified consolidation.
        """
        # Prefer ConsolidationManager if available
        if self.consolidation_manager is not None:
            return self.consolidation_manager.consolidate(
                n_cycles=n_cycles,
                batch_size=batch_size,
                verbose=verbose,
                last_action_holder=self._last_action_container,
            )

        # Fallback: simplified consolidation without manager
        # Check for hippocampus
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
                        self._last_action = action
                        self.deliver_reward(external_reward=reward)
                        stats["experiences_learned"] += 1

                if verbose:
                    print(f"  Cycle {cycle + 1}/{n_cycles}: Replayed {len(batch)} experiences")

            # Increment cycle count regardless of batch size
            stats["cycles_completed"] += 1

        # Exit consolidation mode if HER enabled
        if has_her_replay and hasattr(hippocampus, "exit_consolidation_mode"):
            hippocampus.exit_consolidation_mode()

        return stats

    # =========================================================================
    # HEALTH & CRITICALITY MONITORING (Phase 1.7.6)
    # =========================================================================

    def check_health(self) -> Dict[str, Any]:
        """Check network health and detect pathological states.

        Returns health report with detected issues, severity, and recommendations.
        Compatible with EventDrivenBrain's health monitoring interface.

        Returns:
            HealthReport dict with:
                - is_healthy: bool
                - issues: List[IssueReport]
                - summary: str
                - severity_max: float

        Example:
            health = brain.check_health()
            if not health['is_healthy']:
                for issue in health['issues']:
                    print(f"{issue['issue_type']}: {issue['description']}")
        """
        # Get comprehensive diagnostics for health check
        diagnostics = self.get_diagnostics()

        # Run health check through HealthMonitor
        health_report = self.health_monitor.check_health(diagnostics)

        # Convert HealthReport to dict format
        return {
            'is_healthy': health_report.is_healthy,
            'issues': [
                {
                    'issue_type': issue.issue_type.name,
                    'severity': issue.severity,
                    'description': issue.description,
                    'recommendation': issue.recommendation,
                    'metrics': issue.metrics,
                }
                for issue in health_report.issues
            ],
            'summary': health_report.summary,
            'severity_max': health_report.overall_severity,
        }

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

            # Grow the component
            component.grow_output(n_new=growth_amount)
            growth_actions[component_name] = growth_amount

            # Grow connected pathways via PathwayManager
            self.pathway_manager.grow_connected_pathways(
                component_name=component_name,
                growth_amount=growth_amount,
            )

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

        Args:
            source: Source component name
            output_spikes: Output spikes from source
            current_time: Current simulation time
        """
        # Find all connections from this source
        for (src, tgt), pathway in self.connections.items():
            if src == source:
                # Get axonal delay from pathway config
                delay_ms = getattr(pathway.config, 'axonal_delay_ms', 1.0)

                # Route spikes through pathway
                pathway_output = pathway.forward(output_spikes)

                # Schedule event to target with delay
                event = Event(
                    time=current_time + delay_ms,
                    event_type=EventType.SPIKE,
                    source=source,
                    target=tgt,
                    payload=SpikePayload(spikes=pathway_output),
                )
                self._scheduler.schedule(event)

    def reset_state(self) -> None:
        """Reset all component and system states."""
        for component in self.components.values():
            component.reset_state()

        for pathway in self.connections.values():
            pathway.reset_state()

        # Reset oscillators
        self.oscillators.reset()

        self._current_time = 0.0

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

        # Get neuromodulator states (save only key attributes)
        if self.vta is not None:
            state["neuromodulators"]["vta"] = {
                "global_dopamine": self.vta._global_dopamine,
                "tonic_dopamine": self.vta._tonic_dopamine,
                "phasic_dopamine": self.vta._phasic_dopamine,
            }
        if self.locus_coeruleus is not None:
            state["neuromodulators"]["locus_coeruleus"] = {
                "norepinephrine": self.locus_coeruleus.get_norepinephrine(apply_homeostasis=False),
            }
        if self.nucleus_basalis is not None:
            state["neuromodulators"]["nucleus_basalis"] = {
                "acetylcholine": self.nucleus_basalis.get_acetylcholine(apply_homeostasis=False),
            }

        return state

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
            if "vta" in neuromod_state and self.vta is not None:
                vta_state = neuromod_state["vta"]
                self.vta._tonic_dopamine = vta_state.get("tonic_dopamine", 0.0)
                self.vta._phasic_dopamine = vta_state.get("phasic_dopamine", 0.0)
                self.vta._global_dopamine = vta_state.get("global_dopamine", 0.0)
            # Note: LC and NB currently don't have settable internal state
            # Future: Add state restoration for LC and NB when needed

        # Invalidate cached execution order
        self._execution_order = None
        if "neuromodulators" in state:
            neuromod_state = state["neuromodulators"]
            if "vta" in neuromod_state and self.vta is not None:
                vta_state = neuromod_state["vta"]
                self.vta._tonic_dopamine = vta_state.get("tonic_dopamine", 0.0)
                self.vta._phasic_dopamine = vta_state.get("phasic_dopamine", 0.0)
                self.vta._global_dopamine = vta_state.get("global_dopamine", 0.0)
            # Note: LC and NB currently don't have settable internal state
            # Future: Add state restoration for LC and NB when needed

        # Invalidate cached execution order
        self._execution_order = None


__all__ = [
    "ComponentSpec",
    "ConnectionSpec",
    "DynamicBrain",
]
