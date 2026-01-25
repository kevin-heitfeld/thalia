"""
Dynamic Brain - Component Graph Executor

This module implements a flexible brain architecture where the brain is
treated as a directed graph of neural components (regions, pathways, modules).

DynamicBrain supports:
- Arbitrary number of components
- Flexible topologies (not limited to fixed connectivity)
- User-defined custom components via ComponentRegistry
- Dynamic component addition/removal
- Plugin architecture for external extensions
- Clock-driven execution with axonal delays

Architecture:
    DynamicBrain = Graph of Components
    - nodes: regions (NeuralRegion), pathways (AxonalProjection), custom modules
    - edges: data flow between components
    - execution: clock-driven sequential

Author: Thalia Project
Date: December 15, 2025
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union, cast

import torch
import torch.nn as nn

from thalia.components.coding import compute_firing_rate
from thalia.coordination.growth import GrowthEvent, GrowthManager
from thalia.coordination.oscillator import OscillatorManager
from thalia.core.component_spec import ComponentSpec, ConnectionSpec
from thalia.core.diagnostics import (
    BrainSystemDiagnostics,
    HippocampusDiagnostics,
    StriatumDiagnostics,
)
from thalia.core.neural_region import NeuralRegion
from thalia.core.protocols.component import LearnableComponent
from thalia.diagnostics import CriticalityMonitor, HealthMonitor
from thalia.io.checkpoint_manager import CheckpointManager
from thalia.neuromodulation.manager import NeuromodulatorManager
from thalia.pathways.dynamic_pathway_manager import DynamicPathwayManager
from thalia.stimuli.base import StimulusPattern
from thalia.typing import (
    CheckpointMetadata,
    ComponentGraph,
    SourceOutputs,
    StateDict,
    TopologyGraph,
)

if TYPE_CHECKING:
    from thalia.config.brain_config import BrainConfig
    from thalia.diagnostics.health_monitor import HealthReport
    from thalia.managers.component_registry import ComponentRegistry


class DynamicBrain(nn.Module):
    """Dynamic brain constructed from component graph.

    DynamicBrain builds arbitrary topologies from registered components:
    - Flexible component graph vs. hardcoded regions
    - User-extensible via ComponentRegistry
    - Arbitrary connectivity patterns
    - Plugin support for external components

    Architecture:
        - components: Dict[name -> LearnableComponent] (nodes)
        - connections: Dict[(source, target) -> Pathway] (edges)
        - topology: Directed graph adjacency list
        - execution: Topological ordering

    Example:
        components = {
            "thalamus": ThalamicRelay(config),
            "cortex": LayeredCortex(config),
            "hippocampus": TrisynapticHippocampus(config),
        }

        connections = {
            ("thalamus", "cortex"): AxonalProjection(...),
            ("cortex", "hippocampus"): AxonalProjection(...),
        }

        brain = DynamicBrain(components, connections, brain_config)

    Supports:
        - Custom user regions/pathways (via ComponentRegistry)
        - Arbitrary connectivity patterns
        - Dynamic component addition/removal
        - Checkpoint save/load of arbitrary graphs
    """

    def __init__(
        self,
        components: Dict[str, LearnableComponent],
        connections: Dict[Tuple[str, str], LearnableComponent],
        brain_config: "BrainConfig",
        connection_specs: Optional[Dict[Tuple[str, str], Any]] = None,
    ):
        """Initialize DynamicBrain from component graph.

        Args:
            components: Dict mapping component names to instances
            connections: Dict mapping (source, target) tuples to pathways
            brain_config: Brain configuration (device, dt_ms, oscillators, etc.)
            connection_specs: Optional dict with source_port/target_port info per connection
        """
        super().__init__()

        self.brain_config = brain_config
        self.device = torch.device(brain_config.device)
        self.dt_ms = brain_config.dt_ms  # Cache for fast access

        # =================================================================
        # DISABLE GRADIENTS (biologically-plausible local learning)
        # =================================================================
        # Thalia uses local learning rules (STDP, BCM, Hebbian, three-factor)
        # that do NOT require backpropagation. Disabling gradients provides:
        # - 50% memory savings (no backward graph storage)
        # - 20-40% faster forward passes
        # - Explicit biological constraint
        #
        # Exception: Metacognition module re-enables with torch.enable_grad()
        if not brain_config.enable_gradients:
            torch.set_grad_enabled(False)

        # Minimal config for checkpoint compatibility
        # Note: Sizes will be added after components are known
        self.config = SimpleNamespace(device=brain_config.device)

        # Store components as nn.ModuleDict for proper parameter tracking
        self.components = nn.ModuleDict(components)

        # Update config with component sizes (for CheckpointManager compatibility)
        self._update_config_sizes()

        # Store connections with tuple keys for easy lookup
        # Also register in ModuleDict for parameter tracking
        self.connections: Dict[Tuple[str, str], LearnableComponent] = connections
        self._connection_modules = nn.ModuleDict(
            {f"{src}_to_{tgt}": pathway for (src, tgt), pathway in connections.items()}
        )

        # Store connection specs (for port-based routing)
        self._connection_specs = connection_specs or {}

        # Build topology graph for execution order
        self._topology = self._build_topology_graph()
        self._execution_order: Optional[List[str]] = None

        # Current simulation time
        self._current_time: float = 0.0

        # =================================================================
        # PATHWAY MANAGER (Phase 1.7.1)
        # =================================================================
        # Centralized pathway management for diagnostics and growth coordination

        self.pathway_manager = DynamicPathwayManager(
            connections=self.connections,
            topology=self._topology,
            device=self.device,
            dt_ms=self.dt_ms,
        )

        # =================================================================
        # OSCILLATOR MANAGER (Phase 1.7.2)
        # =================================================================
        # Rhythmic coordination via all brain oscillations (delta, theta, alpha, beta, gamma, ripple)
        # Provides theta-driven encoding/retrieval, gamma feature binding, cross-frequency coupling

        self.oscillators = OscillatorManager(
            dt_ms=self.dt_ms,
            device=self.device,
            delta_freq=brain_config.delta_frequency_hz,
            theta_freq=brain_config.theta_frequency_hz,
            alpha_freq=brain_config.alpha_frequency_hz,
            beta_freq=brain_config.beta_frequency_hz,
            gamma_freq=brain_config.gamma_frequency_hz,
            ripple_freq=brain_config.ripple_frequency_hz,
            couplings=None,  # Use default couplings (theta-gamma, etc.)
        )

        # =================================================================
        # NEUROMODULATOR SYSTEMS (Phase 1.6.3)
        # =================================================================
        # VTA (dopamine), LC (norepinephrine), NB (acetylcholine)
        # Provides centralized neuromodulation
        self.neuromodulator_manager = NeuromodulatorManager()

        # =================================================================
        # REINFORCEMENT LEARNING STATE (Phase 1.6.2)
        # =================================================================
        self._last_action: Optional[int] = None
        self._last_confidence: Optional[float] = None

        # Mutable container for shared last_action state (needed by UnifiedReplayCoordinator)
        self._last_action_container: List[Optional[int]] = [None]

        # =================================================================
        # SPIKE COUNTING
        # =================================================================
        # Track total spikes per component for diagnostics
        self._spike_counts: Dict[str, int] = {name: 0 for name in self.components.keys()}

        # GPU-friendly spike tracking: accumulate on GPU, sync only when needed
        self._spike_tensors: SourceOutputs = {
            name: torch.tensor(0, dtype=torch.int64, device=self.device)
            for name in self.components.keys()
        }

        # =================================================================
        # CLOCK-DRIVEN OPTIMIZATIONS (Phase 1)
        # =================================================================
        # Pre-compute connection topology for fast lookup (O(1) instead of O(N))
        self._component_connections: Dict[str, List[Tuple[str, Any]]] = {}
        for (src, tgt), pathway in self.connections.items():
            if tgt not in self._component_connections:
                self._component_connections[tgt] = []
            self._component_connections[tgt].append((src, pathway))

        # Pre-allocate reusable dict for component inputs (avoid allocation in hot loop)
        self._reusable_component_inputs: SourceOutputs = {}

        # Pre-allocate output cache (update in-place instead of creating new entries)
        self._output_cache: Dict[str, Optional[torch.Tensor]] = {
            name: None for name in self.components.keys()
        }

        # =================================================================
        # GROWTH HISTORY TRACKING (Phase 1.7.7)
        # =================================================================
        # Track all growth events for analysis and debugging
        self._growth_history: List[Any] = []  # List[GrowthEvent] from coordination.growth

        # =================================================================
        # UNIFIED REPLAY COORDINATOR (DEPRECATED - Phase 2)
        # =================================================================
        # **DEPRECATED**: UnifiedReplayCoordinator is deprecated in favor of
        # spontaneous replay (Phase 2 Emergent RL). Kept for backward compatibility.
        #
        # Use brain.consolidate() instead, which triggers spontaneous replay via
        # acetylcholine modulation (no explicit coordinator needed).
        #
        # This will be removed in a future release after all tests are migrated.
        #
        # OLD: Manages all replay types: sleep consolidation, immediate replay,
        #      forward planning, and background planning
        # NEW: Spontaneous replay via CA3 attractor dynamics + synaptic tagging
        #
        # Initialize if brain has required components (hippocampus, striatum, cortex, pfc)

        if all(comp in self.components for comp in ["hippocampus", "striatum", "cortex", "pfc"]):
            # Set cortex output size (L23+L5) for state reconstruction
            cortex = self._get_component("cortex")
            output_size = None

            # Try getting from config first
            if hasattr(cortex, "config"):
                config = cortex.config
                if hasattr(config, "l23_size") and hasattr(config, "l5_size"):
                    output_size = int(config.l23_size) + int(config.l5_size)  # type: ignore[arg-type]

            # Try getting from instance attributes
            if output_size is None and hasattr(cortex, "l23_size") and hasattr(cortex, "l5_size"):
                output_size = int(cortex.l23_size) + int(cortex.l5_size)  # type: ignore[arg-type]

            # Fallback to n_output
            if output_size is None and hasattr(cortex, "n_output"):
                output_size = cortex.n_output  # type: ignore[assignment]

        # =================================================================
        # CHECKPOINT MANAGER (Phase 1.7.4)
        # =================================================================
        # Centralized checkpoint save/load with compression and validation
        self.checkpoint_manager = CheckpointManager(
            brain=self,
            default_compression="zstd",  # Default compression format
        )

        # =================================================================
        # HEALTH & CRITICALITY MONITORING (Phase 1.7.6)
        # =================================================================
        # Initialize health monitor (always enabled)
        self.health_monitor = HealthMonitor(
            enable_oscillator_monitoring=True  # We have oscillators
        )

        # Initialize criticality monitor (optional, enabled by config)
        self.criticality_monitor: Optional[CriticalityMonitor] = None
        criticality_enabled = getattr(brain_config, "monitor_criticality", False)

        if criticality_enabled:
            # CriticalityMonitor doesn't need component sizes - it tracks spike counts directly
            self.criticality_monitor = CriticalityMonitor()

        # =================================================================
        # COMPONENT METADATA
        # =================================================================
        self._component_specs: Dict[str, ComponentSpec] = {}
        """Component specifications (set by BrainBuilder after build)"""

        self._registry: Optional[ComponentRegistry] = None
        """Component registry reference (for component type lookup)"""

        # =================================================================
        # INITIALIZE TEMPORAL PARAMETERS (Phase 1)
        # =================================================================
        # Broadcast dt_ms to all components for initial setup
        # Components compute decay factors, phase increments, etc.
        self._broadcast_temporal_update()

    def _get_component(self, name: str) -> NeuralRegion:
        """Type-safe accessor for components from ModuleDict.

        Args:
            name: Component name

        Returns:
            Component cast to NeuralRegion (avoids Union[Tensor, Module] from ModuleDict)
        """
        return cast(NeuralRegion, self.components[name])

    def _update_config_sizes(self) -> None:
        """Update config with component sizes for CheckpointManager compatibility."""
        # Try to extract common size attributes for compatibility
        if "thalamus" in self.components:
            thalamus = self._get_component("thalamus")
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
            striatum = self._get_component("striatum")
            if hasattr(striatum.config, "n_actions"):
                self.config.n_actions = striatum.config.n_actions

        # Set defaults for any missing attributes (for CheckpointManager compatibility)
        if not hasattr(self.config, "input_size"):
            self.config.input_size = None
        if not hasattr(self.config, "n_actions"):
            self.config.n_actions = None
        if not hasattr(self.config, "hippocampus_size"):
            self.config.hippocampus_size = (
                getattr(self._get_component("hippocampus"), "n_output", 128)
                if "hippocampus" in self.components
                else 128
            )
        if not hasattr(self.config, "pfc_size"):
            self.config.pfc_size = (
                getattr(self._get_component("pfc"), "n_output", 64)
                if "pfc" in self.components
                else 64
            )
        if not hasattr(self.config, "device"):
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

    def set_timestep(self, new_dt_ms: float) -> None:
        """Change simulation timestep adaptively during training.

        Updates dt_ms and propagates temporal parameter updates to:
        - All components (neurons, STP, learning strategies)
        - Oscillator manager (phase increments)
        - Pathway manager (delay buffers)

        Use cases:
        - Memory replay: 10x speedup (dt=10ms) during consolidation
        - Critical learning: Slow down to 0.1ms for precise timing
        - Energy efficiency: Larger dt when dynamics are stable

        Args:
            new_dt_ms: New timestep in milliseconds (must be positive)

        Raises:
            ValueError: If new_dt_ms <= 0

        Example:
            # Speed up replay 10x
            brain.set_timestep(10.0)
            brain.forward(replay_input)

            # Restore normal speed
            brain.set_timestep(1.0)
        """
        if new_dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {new_dt_ms}")

        # Update brain dt
        self.dt_ms = new_dt_ms
        self.brain_config.dt_ms = new_dt_ms

        # Broadcast to all components and subsystems
        self._broadcast_temporal_update()

    def _broadcast_temporal_update(self) -> None:
        """Notify all components and subsystems of dt change.

        Called by:
        - __init__() to initialize temporal parameters
        - set_timestep() when dt changes during simulation

        Propagates update to:
        - Components (neurons, STP, learning strategies via update_temporal_parameters)
        - Oscillator manager (phase increments)
        - Pathway manager (delay buffers)
        """
        # Update all components
        for component in self.components.values():
            if hasattr(component, "update_temporal_parameters"):
                component.update_temporal_parameters(self.dt_ms)

        # Update all connections/pathways
        for pathway in self.connections.values():
            if hasattr(pathway, "update_temporal_parameters"):
                pathway.update_temporal_parameters(self.dt_ms)

        # Update oscillator manager
        if hasattr(self, "oscillators") and hasattr(self.oscillators, "update_temporal_parameters"):
            self.oscillators.update_temporal_parameters(self.dt_ms)

        # Update pathway manager
        if hasattr(self, "pathway_manager"):
            self.pathway_manager.dt_ms = self.dt_ms

    def measure_phase_locking(self) -> float:
        """Measure gamma-theta phase locking.

        Computes the phase-amplitude coupling between gamma amplitude
        and theta phase, a key metric of oscillator coordination.

        Returns:
            Phase locking value [0, 1], where higher is better coupling
        """
        # Get coupling amplitude if available
        coupled_amps = self.oscillators.get_coupled_amplitudes()
        if "gamma" in coupled_amps:
            # Use coupling amplitude as proxy for phase locking
            # (higher coupling = better phase locking)
            return coupled_amps["gamma"]

        # Fallback: compute simple phase coherence
        # Gamma should be at ~40 Hz, theta at ~8 Hz
        # Ideal ratio is 5:1 (5 gamma cycles per theta cycle)
        expected_ratio = 5.0
        actual_ratio = self.oscillators.gamma.frequency_hz / self.oscillators.theta.frequency_hz

        # Phase locking = how close to expected ratio
        ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio
        phase_locking = max(0.0, 1.0 - ratio_error)

        return phase_locking

    def _build_topology_graph(self) -> TopologyGraph:
        """Build adjacency list of component dependencies.

        Returns:
            Dict mapping component names to list of downstream dependencies

        Example:
            {"thalamus": ["cortex"], "cortex": ["hippocampus", "striatum"]}
        """
        graph: Dict[str, List[str]] = {name: [] for name in self.components.keys()}

        # Extract connections from tuple keys
        for src, tgt in self.connections.keys():
            if src in graph:
                graph[src].append(tgt)

        return graph

    def _get_execution_order(self) -> List[str]:
        """Get execution order for components using topological sort.

        Uses modified Kahn's algorithm to order components by dependencies,
        with alphabetical tiebreaking for determinism. This ensures that
        components execute after their inputs are available in the cache,
        minimizing latency through the network.

        For circular dependencies (recurrent connections), breaks cycles by
        choosing the alphabetically first component in the cycle.

        Returns:
            List of component names in dependency-respecting order

        Note:
            Cached for performance. Invalidated when components added/removed.
        """
        if self._execution_order is not None:
            return self._execution_order

        # Build dependency graph: component -> list of components it depends on
        dependencies: Dict[str, Set[str]] = {name: set() for name in self.components.keys()}

        for src, tgt in self.connections.keys():
            # tgt depends on src (needs src's output)
            if tgt in dependencies:  # tgt might not exist if connection is to port
                dependencies[tgt].add(src)

        # Kahn's algorithm with alphabetical tiebreaking
        # Start with components that have no dependencies (sensory inputs)
        in_degree = {name: len(deps) for name, deps in dependencies.items()}
        queue = sorted([name for name, degree in in_degree.items() if degree == 0])
        sorted_order = []

        while queue:
            # Pop alphabetically first (deterministic tiebreaking)
            current = queue.pop(0)
            sorted_order.append(current)

            # Reduce in-degree for dependents
            for name, deps in dependencies.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        # All dependencies satisfied, add to queue (keep sorted)
                        queue.append(name)
                        queue.sort()

        # Handle cycles: any remaining components have circular dependencies
        # Add them in alphabetical order (breaks cycles arbitrarily but deterministically)
        remaining = sorted([name for name, degree in in_degree.items() if degree > 0])
        sorted_order.extend(remaining)

        # Cache result
        self._execution_order = sorted_order
        return sorted_order

    def forward(
        self,
        sensory_input: Optional[
            Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor], StimulusPattern]]]
        ] = None,
        n_timesteps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute brain for n_timesteps.

        Supports multiple input modes:

        **1. Stimulus Pattern (recommended)** - Explicit temporal structure:
           ```python
           from thalia.stimuli import Sustained, Sequential, Transient

           # Sustained: constant input over duration
           brain.forward({"thalamus": Sustained(pattern, duration_ms=500)})

           # Sequential: explicit per-frame input
           brain.forward({"thalamus": Sequential(frames=[f1, f2, f3])})

           # Transient: brief pulse
           brain.forward({"thalamus": Transient(pattern, onset_ms=0)})
           ```

        **2. List of tensors** - Explicit per-timestep inputs:
           ```python
           brain.forward({"thalamus": [input_t1, input_t2, input_t3]})
           ```
           n_timesteps inferred from list length

        **3. Single tensor + n_timesteps** - Broadcast mode:
           ```python
           brain.forward({"thalamus": input}, n_timesteps=10)
           ```
           Same input sustained for all timesteps

        **4. Single tensor** - One timestep:
           ```python
           brain.forward({"thalamus": input})
           ```

        Args:
            sensory_input: Input pattern(s):
                          - StimulusPattern: Sustained/Transient/Sequential/Programmatic
                          - List[Tensor]: Explicit per-timestep inputs
                          - Tensor: Single timestep or broadcast with n_timesteps
                          - Dict: Multi-component inputs (any of above types)
                          - None: Maintenance mode
            n_timesteps: Number of timesteps (optional):
                        - If StimulusPattern: inferred from pattern duration
                        - If List: inferred from list length
                        - If Tensor + n_timesteps: broadcasts input
                        - If None and Tensor: defaults to 1 timestep

        Returns:
            Dict containing:
                - "outputs": Dict[component_name -> final output]
                - "time": Final simulation time
                - "spike_counts": Total spike counts per component

        Example:
            >>> from thalia.stimuli import Sustained, Sequential
            >>>
            >>> # Sustained visual input
            >>> pattern = torch.randn(128)
            >>> stim = Sustained(pattern, duration_ms=500)
            >>> result = brain.forward({"thalamus": stim})
            >>>
            >>> # Movie frames
            >>> frames = [torch.randn(128) for _ in range(100)]
            >>> stim = Sequential(frames)
            >>> result = brain.forward({"thalamus": stim})
        """
        # Handle default n_timesteps
        if n_timesteps is None:
            n_timesteps = 1  # Single timestep default (sensory input provided once)

        # Convert sensory_input to input_data dict format and detect mode
        sequence_mode = False
        input_sequences: Dict[str, List[torch.Tensor]] = {}
        dt_ms = self.dt_ms  # Use cached value for fast access

        if sensory_input is not None:
            if isinstance(sensory_input, dict):
                # Check if any values are lists or StimulusPattern (sequence mode)
                for key, value in sensory_input.items():
                    if isinstance(value, StimulusPattern):
                        # Convert StimulusPattern to sequence of tensors
                        sequence_mode = True
                        n_steps = value.duration_timesteps(dt_ms)
                        if n_timesteps is not None and n_timesteps != 1 and n_timesteps != n_steps:
                            raise ValueError(
                                f"StimulusPattern for '{key}' has duration {n_steps} timesteps, "
                                f"but n_timesteps={n_timesteps} specified. Remove n_timesteps argument."
                            )
                        n_timesteps = n_steps

                        # Generate sequence from stimulus pattern
                        frames = []
                        for t in range(n_steps):
                            frame = value.get_input(t, dt_ms)
                            if frame is None:
                                # Pattern returns None (silence) - use zeros
                                frame = torch.zeros(
                                    value.shape, dtype=torch.bool, device=value.device
                                )
                            frames.append(frame)
                        input_sequences[key] = frames
                    elif isinstance(value, list):
                        sequence_mode = True
                        input_sequences[key] = value
                        # Infer n_timesteps from sequence length
                        n_timesteps = len(value)
                    else:
                        # Single tensor (will be broadcast in broadcast mode)
                        input_sequences[key] = [value]  # Wrap for uniform handling
            else:
                # Convert tensor to dict with thalamus as input
                input_sequences = {"thalamus": [sensory_input]}  # Wrap for broadcast
        else:
            # No input (maintenance mode)
            input_sequences = {}

        # Validate sequence mode consistency
        if sequence_mode:
            seq_lengths = [len(seq) for seq in input_sequences.values() if isinstance(seq, list)]
            if len(set(seq_lengths)) > 1:
                raise ValueError(
                    f"Sequence mode: All input sequences must have same length. "
                    f"Got lengths: {seq_lengths}"
                )

        # === CLOCK-DRIVEN EXECUTION (ADR-003) ===
        # All regions execute every timestep in execution order.
        # Axonal delays are handled internally by pathway CircularDelayBuffers.
        # This ensures continuous dynamics: membrane decay, recurrent connections,
        # oscillators, and short-term plasticity all evolve every timestep.
        outputs = {name: None for name in self.components.keys()}

        for timestep in range(n_timesteps):
            # 1. Route sensory inputs to entry components (DO NOT store in output_cache)
            # Direct sensory inputs should NOT be treated as component outputs
            sensory_inputs_this_timestep: SourceOutputs = {}
            for comp_name, sequence in input_sequences.items():
                if comp_name in self.components:
                    # Get input for this timestep
                    if sequence_mode:
                        input_t = sequence[timestep]
                    else:
                        # Broadcast mode: repeat first (only) element
                        input_t = sequence[0]

                    # Store sensory input separately (not in output_cache)
                    sensory_inputs_this_timestep[comp_name] = input_t

            # 2. Execute all components in execution order
            new_outputs: SourceOutputs = {}

            for comp_name in self._get_execution_order():
                component = self.components[comp_name]

                # Collect inputs from all incoming pathways using PREVIOUS timestep's cache
                # OPTIMIZATION: Reuse dict instead of creating new one each iteration
                self._reusable_component_inputs.clear()

                # OPTIMIZATION: Direct lookup instead of iterating all connections
                for src, pathway in self._component_connections.get(comp_name, []):
                    if src in self._output_cache and self._output_cache[src] is not None:
                        # Pathway applies axonal delay internally via CircularDelayBuffer
                        # Using previous timestep's output ensures all components see same state
                        delayed_outputs = pathway.forward({src: self._output_cache[src]})
                        self._reusable_component_inputs.update(delayed_outputs)

                # Also check if this component received direct sensory input
                if comp_name in sensory_inputs_this_timestep:
                    # Direct sensory input (no pathway), use "input" key
                    self._reusable_component_inputs["input"] = sensory_inputs_this_timestep[
                        comp_name
                    ]

                # Execute component
                # Empty dict is valid (zero-input execution for recurrent/spontaneous activity)
                component_output = component.forward(self._reusable_component_inputs)

                # Store output temporarily (don't update cache until all components execute)
                outputs[comp_name] = component_output
                new_outputs[comp_name] = component_output

            # 3. Update cache with this timestep's outputs (for next timestep)
            # This ensures true parallel semantics - no component sees outputs from
            # other components within the same timestep
            # OPTIMIZATION: In-place update instead of dict.update()
            for comp_name, output in new_outputs.items():
                self._output_cache[comp_name] = output

            # Track spike counts
            # OPTIMIZATION: Accumulate on GPU, avoid .item() sync in hot loop
            for comp_name, component_output in new_outputs.items():
                if component_output is not None and isinstance(component_output, torch.Tensor):
                    # Keep on GPU - no sync!
                    self._spike_tensors[comp_name] += component_output.sum()

            # 4. Broadcast oscillator phases every timestep
            self._broadcast_oscillator_phases()

        # Update final time
        self._current_time += n_timesteps * self.dt_ms

        # OPTIMIZATION: Sync spike counts from GPU only once at end
        for comp_name, spike_tensor in self._spike_tensors.items():
            self._spike_counts[comp_name] = int(spike_tensor.item())

        return {
            "outputs": outputs,
            "time": self._current_time,
            "spike_counts": self._spike_counts.copy(),
            "final_time": self._current_time,
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

        **Note on Gamma Oscillations**:
        Explicit gamma waves are disabled because two gamma frequencies naturally
        emerge from the L6a-TRN-relay and L6b-relay feedback loops (~40Hz and ~60Hz).
        This emergence is biologically accurate: cortical gamma arises from
        corticothalamic interactions, not a central oscillator.
        """
        # Advance oscillators
        self.oscillators.advance(dt_ms=self.dt_ms)

        # Get all phases and signals
        phases = self.oscillators.get_phases()
        signals = self.oscillators.get_signals()

        # Broadcast directly to components that support it
        for component in cast(Dict[str, NeuralRegion], self.components).values():
            if hasattr(component, "set_oscillator_phases"):
                component.set_oscillator_phases(phases, signals)

    # =========================================================================
    # REINFORCEMENT LEARNING INTERFACE
    # =========================================================================

    def select_action(self, explore: bool = True, use_planning: bool = True) -> tuple[int, float]:
        """Select action based on current striatum state.

        Uses striatum to select actions based on accumulated evidence.

        If use_planning=True and UnifiedReplayCoordinator is available:
        - Uses forward planning to simulate action outcomes
        - Returns best action from simulated rollouts
        - Falls back to striatum if planning unavailable

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

        striatum = self._get_component("striatum")

        # Striatum has finalize_action method for action selection
        if hasattr(striatum, "finalize_action"):
            # Call finalize_action with correct signature (only explore parameter)
            result = striatum.finalize_action(explore=explore)  # type: ignore[operator]

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

    def reset_trial(self) -> None:
        """Reset state for new trial in action learning experiments.

        Delegates to striatum.reset_trial() if present. This clears trial-specific
        state (votes, FSI membrane, action selection) while preserving learned
        weights and exploration statistics.

        Biology: Between trials, FSI neurons return to resting potential and
        action votes reset, but synaptic plasticity persists.

        Use Case:
            for trial in range(n_trials):
                brain.reset_trial()  # Start fresh
                for t in range(trial_length):
                    spikes = brain(inputs)
                action = brain.finalize_action()
                reward = environment.get_reward(action)
                brain.deliver_reward(reward)

        Raises:
            AttributeError: If striatum doesn't have reset_trial method
        """
        if "striatum" in self.components:
            striatum = self.components["striatum"]
            if hasattr(striatum, "reset_trial"):
                striatum.reset_trial()  # type: ignore[attr-defined]
            else:
                raise AttributeError(
                    f"Striatum component ({type(striatum).__name__}) does not have "
                    f"reset_trial method. Update striatum implementation."
                )
        else:
            # No striatum - no-op (some architectures may not need this)
            pass

    def deliver_reward(self, external_reward: Optional[float] = None) -> Dict[str, Any]:
        """Deliver reward signal and update exploration statistics.

        **Continuous Learning Architecture**: Learning happens automatically in striatum's
        forward() pass using broadcast dopamine. This method broadcasts dopamine globally
        and updates exploration statistics.

        Uses striatum to select actions based on accumulated evidence.
        The brain broadcasts reward as dopamine signal to all regions,
        which use three-factor learning (eligibility × dopamine × lr)
        for continuous synaptic updates.
s
        Action values emerge from D1-D2 synaptic weight competition - no Q-values!

        Args:
            external_reward: Task-based reward value (typically -1 to +1),
                           or None for pure intrinsic reward

        Returns:
            Metrics dict containing:
                - d1_ltp: 0.0 (learning tracked internally)
                - d1_ltd: 0.0 (learning tracked internally)
                - d2_ltp: 0.0 (learning tracked internally)
                - d2_ltd: 0.0 (learning tracked internally)
                - net_change: 0.0 (learning tracked internally)
                - dopamine: Dopamine level broadcast globally

        Raises:
            ValueError: If striatum not found

        Example:
            # After action selection
            action, _ = brain.select_action()

            # Execute in environment
            reward = env.step(action)

            # Deliver reward (updates dopamine and exploration)
            metrics = brain.deliver_reward(external_reward=reward)
            print(f"Dopamine level: {metrics['dopamine']:.2f}")

        Note:
            Actual learning happens continuously in striatum.forward() using
            the broadcast dopamine level. This method only broadcasts dopamine
            and updates exploration - no discrete learning trigger.
        """
        if "striatum" not in self.components:
            raise ValueError(
                "Striatum component not found. Cannot deliver reward. "
                "Brain must include 'striatum' component for RL tasks."
            )

        striatum = self._get_component("striatum")

        # Compute total reward (external + intrinsic if available)
        intrinsic_reward = self._compute_intrinsic_reward()
        if external_reward is None:
            total_reward = intrinsic_reward
        else:
            total_reward = external_reward + intrinsic_reward
            total_reward = max(-2.0, min(2.0, total_reward))

        # Deliver reward to VTA and broadcast dopamine globally
        # This ensures all regions (hippocampus, prefrontal, striatum, etc.) receive
        # dopamine signal for synaptic tagging and learning
        expected_value = 0.0  # Simplified: no value prediction yet
        self.neuromodulator_manager.vta.deliver_reward(
            external_reward=total_reward,
            expected_value=expected_value,
        )
        # Broadcast updated dopamine to all regions
        self.neuromodulator_manager.broadcast_to_regions(self.components)

        # Update exploration statistics based on reward
        # Striatum applies continuous learning automatically in forward() using broadcast dopamine
        correct = external_reward > 0 if external_reward is not None else intrinsic_reward > 0
        if hasattr(striatum, "exploration"):
            striatum.exploration.update_performance(total_reward, correct)  # type: ignore[attr-defined]

        # Return learning metrics (continuous learning happens in forward())
        learning_metrics = {
            "d1_ltp": 0.0,  # Learning metrics tracked internally in forward()
            "d1_ltd": 0.0,
            "d2_ltp": 0.0,
            "d2_ltd": 0.0,
            "net_change": 0.0,
            "dopamine": self.neuromodulator_manager.vta.get_global_dopamine(),
        }

        # Sync last_action to container for consolidation manager
        if self._last_action is not None:
            self._last_action_container[0] = self._last_action

        return learning_metrics

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
        if "striatum" in self.components:
            striatum = self._get_component("striatum")
            # Apply counterfactual learning for alternate action
            if hasattr(striatum, "deliver_counterfactual_reward"):
                # Deliver counterfactual reward with scale factor
                striatum.deliver_counterfactual_reward(  # type: ignore[attr-defined]
                    reward=modulated_reward,
                    action=other_action,
                    counterfactual_scale=counterfactual_scale,
                )
                counterfactual_metrics["counterfactual_applied"] = True

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
        if not hasattr(self, "_novelty_signal"):
            self._novelty_signal = 1.0
        return max(1.0, self._novelty_signal)

    # =========================================================================
    # NEUROMODULATION
    # =========================================================================

    def _update_neuromodulators(self) -> None:
        """Update centralized neuromodulator systems and broadcast to components.

        Updates VTA (dopamine), locus coeruleus (norepinephrine), and nucleus
        basalis (acetylcholine) based on intrinsic reward, uncertainty, and
        prediction error. Then broadcasts signals to all components.

        This is called automatically during forward() to maintain neuromodulator
        dynamics every timestep.

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
        self.neuromodulator_manager.vta.update(dt_ms=self.dt_ms, intrinsic_reward=intrinsic_reward)
        self.neuromodulator_manager.locus_coeruleus.update(
            dt_ms=self.dt_ms, uncertainty=uncertainty
        )
        self.neuromodulator_manager.nucleus_basalis.update(
            dt_ms=self.dt_ms, prediction_error=prediction_error
        )

        # Update PFC cognitive load for temporal discounting (Phase 3)
        if "pfc" in self.components:
            pfc = self._get_component("pfc")
            if hasattr(pfc, "discounter") and pfc.discounter is not None:
                pfc.update_cognitive_load(cognitive_load)  # type: ignore[operator]
            elif hasattr(pfc, "update_cognitive_load"):
                # Direct method on PFC component
                pfc.update_cognitive_load(cognitive_load)  # type: ignore[operator]

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
            if hasattr(pathway, "set_neuromodulators"):
                pathway.set_neuromodulators(
                    dopamine=dopamine,
                    norepinephrine=norepinephrine,
                )

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
        if "cortex" in self.components:
            cortex = self._get_component("cortex")

            # Try PredictiveCortex first (has explicit free_energy)
            if hasattr(cortex, "state") and hasattr(cortex.state, "free_energy"):
                free_energy = cortex.state.free_energy  # type: ignore[union-attr]

                # Free energy is typically 0-10, lower is better
                # Map: 0 → +1 (perfect prediction), 5 → 0, 10+ → -1 (bad prediction)
                cortex_reward = 1.0 - 0.2 * float(min(free_energy, 10.0))  # type: ignore[arg-type]
                cortex_reward = max(-1.0, min(1.0, cortex_reward))
                reward += cortex_reward
                n_sources += 1

            # Fallback: check for accumulated free energy in PredictiveCortex
            elif hasattr(cortex, "_total_free_energy"):
                total_fe = float(cortex._total_free_energy)  # type: ignore[arg-type]
                cortex_reward = 1.0 - 0.1 * min(total_fe, 20.0)
                cortex_reward = max(-1.0, min(1.0, cortex_reward))
                reward += cortex_reward
                n_sources += 1

            # LayeredCortex: Use L2/3 firing rate as proxy for processing quality
            # High activity = engaged processing, low = unresponsive
            elif (
                hasattr(cortex, "state")
                and hasattr(cortex.state, "l23_spikes")
                and cortex.state.l23_spikes is not None
            ):
                l23_activity = compute_firing_rate(cortex.state.l23_spikes)  # type: ignore[arg-type]
                # Map [0, 1] to [-0.5, 0.5] - less weight than free energy
                cortex_reward = l23_activity - 0.5
                reward += cortex_reward
                n_sources += 1

        # =====================================================================
        # 2. HIPPOCAMPUS PATTERN COMPLETION (memory recall quality)
        # =====================================================================
        # High pattern similarity = successful memory retrieval = reward
        # Biology: VTA observes CA1 output activity. Strong coherent firing = successful recall.
        # We infer similarity from CA1 spike rate (observable signal).
        if "hippocampus" in self.components:
            hippocampus = self._get_component("hippocampus")
            if (
                hasattr(hippocampus, "state")
                and hasattr(hippocampus.state, "ca1_spikes")
                and hippocampus.state.ca1_spikes is not None
            ):

                # CA1 firing rate as proxy for retrieval quality
                # High rate = strong recall, low rate = weak/no recall
                ca1_activity = compute_firing_rate(hippocampus.state.ca1_spikes)  # type: ignore[arg-type]

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
        if "cortex" in self.components:
            cortex = self._get_component("cortex")
            if (
                hasattr(cortex, "state")
                and hasattr(cortex.state, "free_energy")
                and cortex.state.free_energy is not None
            ):
                free_energy = cortex.state.free_energy  # type: ignore[union-attr]
                # High FE → high uncertainty
                cortex_uncertainty = float(min(1.0, free_energy / 10.0))  # type: ignore[arg-type]
                uncertainty += cortex_uncertainty
                n_sources += 1

        # Average across sources
        if n_sources > 0:
            uncertainty = float(uncertainty) / n_sources
        else:
            # No signals → assume moderate uncertainty
            uncertainty = 0.3

        return float(max(0.0, min(1.0, uncertainty)))

    def _compute_cognitive_load(self) -> float:
        """Compute current cognitive load from PFC working memory usage.

        Phase 3 functionality: Cognitive load drives temporal discounting.
        High working memory usage → high load → more impulsive choices.

        Returns:
            Cognitive load (0-1), where 1 = maximum capacity
        """
        if "pfc" not in self.components:
            return 0.0

        pfc = self._get_component("pfc")

        # Measure PFC activity from spike output, not internal WM state
        if not hasattr(pfc, "state") or pfc.state is None or not hasattr(pfc.state, "spikes"):
            return 0.0

        if pfc.state.spikes is None:
            return 0.0

        # Measure working memory load from sustained spike activity
        wm_activity = compute_firing_rate(pfc.state.spikes)  # type: ignore[arg-type]

        # Also consider number of active goals (if hierarchical goals enabled)
        goal_load = 0.0
        if hasattr(pfc, "goal_manager") and pfc.goal_manager is not None:
            n_active = len(pfc.goal_manager.active_goals)  # type: ignore[union-attr,arg-type]
            max_goals = int(pfc.goal_manager.config.max_active_goals)  # type: ignore[union-attr,arg-type]
            goal_load = float(n_active) / max(max_goals, 1)

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
        if "cortex" in self.components:
            cortex = self._get_component("cortex")
            if hasattr(cortex, "state") and hasattr(cortex.state, "free_energy"):
                free_energy = cortex.state.free_energy  # type: ignore[union-attr]
                # Map FE to [0, 1]: 0 → 0, 5 → 0.5, 10+ → 1.0
                cortex_pe = float(min(1.0, float(free_energy) / 10.0))  # type: ignore[arg-type, operator]
                prediction_error += cortex_pe  # type: ignore[operator]
                n_sources += 1

        # Average across sources
        if n_sources > 0:
            prediction_error = float(prediction_error) / n_sources  # type: ignore[operator]
        else:
            # No signals → assume low PE (familiar context)
            prediction_error = 0.2

        return float(max(0.0, min(1.0, prediction_error)))  # type: ignore[arg-type, return-value]

    def consolidate(
        self,
        duration_ms: float = 1000.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Trigger memory consolidation via spontaneous replay.

        No explicit coordination - just set acetylcholine low and run brain forward.
        Hippocampus spontaneously replays high-priority patterns via sharp-wave ripples.

        Biological mechanism:
        1. Lower acetylcholine (0.1) → sleep/rest mode
        2. Hippocampus CA3 spontaneously reactivates stored patterns
        3. Replay probability ∝ synaptic tag strength (Frey-Morris tagging)
        4. Ripples occur at ~1-3 Hz during low ACh
        5. Restore acetylcholine (0.7) → awake/encoding mode

        Args:
            duration_ms: Consolidation duration in milliseconds (default 1000ms = 1 second)
            verbose: Whether to print ripple statistics

        Returns:
            Dict with consolidation statistics:
                - ripples: Number of sharp-wave ripples detected
                - duration_ms: Total consolidation duration
                - ripple_rate_hz: Ripples per second

        Raises:
            ValueError: If hippocampus not present in brain

        Example:
            # Consolidate for 10 seconds
            stats = brain.consolidate(duration_ms=10000, verbose=True)
            # Output: "Consolidation: 23 ripples in 10000ms (2.3 Hz)"
        """
        # Check for hippocampus
        if "hippocampus" not in self.components:
            raise ValueError(
                "Hippocampus component required for consolidation. "
                "Brain must include 'hippocampus' component."
            )

        hippocampus = self._get_component("hippocampus")

        # Check that hippocampus supports spontaneous replay
        if not hasattr(hippocampus, "set_acetylcholine"):
            raise AttributeError(
                f"Hippocampus ({type(hippocampus).__name__}) does not support spontaneous replay. "
                f"Ensure hippocampus implements set_acetylcholine() method."
            )

        # Sync last_action to container before consolidation
        if self._last_action is not None:
            self._last_action_container[0] = self._last_action

        # Enter consolidation mode (low acetylcholine)
        hippocampus.set_acetylcholine(0.1)

        # Run brain forward - replay happens automatically
        n_timesteps = int(duration_ms / self.dt_ms)
        ripple_count = 0

        for _ in range(n_timesteps):
            self.forward(None)  # No sensory input during sleep

            # Count ripples (if hippocampus state has ripple_detected)
            if hasattr(hippocampus, "state") and hippocampus.state is not None:
                if (
                    hasattr(hippocampus.state, "ripple_detected")
                    and hippocampus.state.ripple_detected
                ):
                    ripple_count += 1

        # Return to encoding mode (high acetylcholine)
        hippocampus.set_acetylcholine(0.7)

        # Compute ripple rate
        ripple_rate_hz = ripple_count / (duration_ms / 1000.0) if duration_ms > 0 else 0.0

        if verbose:
            print(
                f"Consolidation: {ripple_count} ripples in {duration_ms}ms ({ripple_rate_hz:.2f} Hz)"
            )

        return {
            "ripples": ripple_count,
            "duration_ms": duration_ms,
            "ripple_rate_hz": ripple_rate_hz,
        }

    # =========================================================================
    # HEALTH & CRITICALITY MONITORING
    # =========================================================================

    def check_health(self) -> HealthReport:
        """Check network health and detect pathological states.

        Returns health report with detected issues, severity, and recommendations.

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
        # Get comprehensive diagnostics for health check
        diagnostics = self.get_diagnostics()

        # Run health check through HealthMonitor and return HealthReport directly
        return self.health_monitor.check_health(diagnostics)

    # =========================================================================
    # DIAGNOSTICS & GROWTH
    # =========================================================================

    def check_growth_needs(self) -> Dict[str, Any]:
        """Check if any components need growth based on capacity metrics.

        Analyzes utilization, saturation, and activity patterns to determine
        if components need more capacity. Uses GrowthManager for standardized
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
                    "weight_saturation": (
                        metrics.saturation_fraction
                        if metrics.saturation_fraction is not None
                        else 0.0
                    ),
                    "synapse_usage": metrics.synapse_usage,
                    "neuron_count": metrics.total_neurons,
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
            Growth history is not tracked in DynamicBrain.
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
            component.grow_output(n_new=growth_amount)  # type: ignore[operator]
            growth_actions[component_name] = growth_amount

            # Grow connected pathways via PathwayManager
            self.pathway_manager.grow_connected_pathways(
                component_name=component_name,
                growth_amount=growth_amount,
            )

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
            # Multiple inputs - concatenate along feature dimension (generic fallback)
            # Note: Components implement biologically-appropriate integration via InputRouter:
            # - Thalamus: Spatial summation (sum EPSPs from multiple sources)
            # - Cortex: Dendritic compartmentalization (semi-independent processing)
            # - Striatum: Weighted integration with dopamine modulation
            # - Superior Colliculus: Winner-take-all (max across modalities)
            # This concatenation is only used in the generic graph execution path
            # when components don't override with their own forward() logic.
            return torch.cat(inputs, dim=-1)

    def get_component(self, name: str) -> LearnableComponent:
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
            raise KeyError(f"Component '{name}' not found. Available: {available}")
        # Cast from Module to LearnableComponent (all our components are LearnableComponent)
        component: LearnableComponent = self.components[name]  # type: ignore[assignment]
        return component

    def add_component(
        self,
        name: str,
        component: LearnableComponent,
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

        # Initialize GPU-friendly spike tracking tensor
        if hasattr(self, "_spike_tensors"):
            self._spike_tensors[name] = torch.tensor(0, dtype=torch.int64, device=self.device)

        # Invalidate cached execution order
        self._execution_order = None

    def add_connection(
        self,
        source: str,
        target: str,
        pathway: LearnableComponent,
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
        # Extract base target name (may have port suffix like "thalamus:l6a_feedback")
        target_base = target.split(":")[0]
        if target_base not in self.components:
            raise ValueError(f"Target component '{target_base}' not found")

        # Store with tuple key (target may be compound like "thalamus:l6a_feedback")
        self.connections[(source, target)] = pathway

        # Also register in ModuleDict for parameter tracking
        # Strip port suffix for module key (PyTorch doesn't like colons in keys)
        conn_key = f"{source}_to_{target_base}"
        # Make key unique if port specified
        if ":" in target:
            port = target.split(":")[1]
            conn_key = f"{conn_key}_{port}"
        self._connection_modules[conn_key] = pathway

        # Update topology
        if source in self._topology:
            if target not in self._topology[source]:
                self._topology[source].append(target)

        # Invalidate cached execution order
        self._execution_order = None

    @property
    def regions(self) -> ComponentGraph:
        """Get all brain regions.

        Returns:
            Dict mapping region names to component instances

        Note:
            This is an alias for `components` for API compatibility.
        """
        return dict(self.components)  # type: ignore[arg-type, return-value]

    def reset_state(self) -> None:
        """Reset all component and system states."""
        for component in cast(Dict[str, NeuralRegion], self.components).values():
            component.reset_state()

        for pathway in self.connections.values():
            pathway.reset_state()  # type: ignore[operator]

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
        if "striatum" not in self.components:
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

        striatum = self._get_component("striatum")

        # Get n_actions from striatum
        n_actions = getattr(striatum, "n_actions", 2)
        neurons_per = getattr(striatum, "neurons_per_action", 10)

        # Per-action weight means
        d1_per_action = []
        d2_per_action = []
        net_per_action = []

        for a in range(n_actions):
            start = a * neurons_per
            end = start + neurons_per

            # Safe access to weights (might not be linked yet)
            d1_mean = 0.0
            d2_mean = 0.0

            if hasattr(striatum, "d1_pathway"):
                try:
                    weights = striatum.d1_pathway.weights
                    if weights is not None:
                        d1_mean = weights[start:end].mean().item()  # type: ignore[index]
                except RuntimeError:
                    # Pathway not linked yet - skip
                    pass

            if hasattr(striatum, "d2_pathway"):
                try:
                    weights = striatum.d2_pathway.weights
                    if weights is not None:
                        d2_mean = weights[start:end].mean().item()  # type: ignore[index]
                except RuntimeError:
                    # Pathway not linked yet - skip
                    pass

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
            if (
                hasattr(striatum, "d1_pathway")
                and hasattr(striatum.d1_pathway, "eligibility")
                and striatum.d1_pathway.eligibility is not None
            ):
                d1_elig_per_action.append(
                    striatum.d1_pathway.eligibility[start:end].abs().mean().item()  # type: ignore[index]
                )
            else:
                d1_elig_per_action.append(0.0)
            if (
                hasattr(striatum, "d2_pathway")
                and hasattr(striatum.d2_pathway, "eligibility")
                and striatum.d2_pathway.eligibility is not None
            ):
                d2_elig_per_action.append(
                    striatum.d2_pathway.eligibility[start:end].abs().mean().item()  # type: ignore[index]
                )
            else:
                d2_elig_per_action.append(0.0)

        # UCB and exploration - safe access to potentially missing attributes
        action_counts = []
        total_trials = 0
        if hasattr(striatum, "_action_counts") and striatum._action_counts is not None:
            action_counts = [int(c) for c in striatum._action_counts.tolist()]  # type: ignore[operator]
        if hasattr(striatum, "_total_trials"):
            total_trials = int(getattr(striatum, "_total_trials", 0))
        exploration_prob = getattr(striatum, "_last_exploration_prob", 0.0)

        return StriatumDiagnostics(
            d1_per_action=d1_per_action,
            d2_per_action=d2_per_action,
            net_per_action=net_per_action,
            d1_elig_per_action=d1_elig_per_action,
            d2_elig_per_action=d2_elig_per_action,
            last_action=self._last_action,
            exploring=getattr(striatum, "_last_exploring", False),
            exploration_prob=exploration_prob,
            action_counts=action_counts,
            total_trials=total_trials,
        )

    def _collect_hippocampus_diagnostics(self) -> HippocampusDiagnostics:
        """Collect structured diagnostics from hippocampus.

        Returns:
            HippocampusDiagnostics dataclass with layer activity and memory metrics
        """
        if "hippocampus" not in self.components:
            # Return empty diagnostics if hippocampus not present
            return HippocampusDiagnostics(
                ca1_total_spikes=0.0,
                ca1_normalized=0.0,
                dg_spikes=0.0,
                ca3_spikes=0.0,
                ca1_spikes=0.0,
            )

        hippo = self._get_component("hippocampus")

        # CA1 activity (key for match/mismatch)
        ca1_spikes = 0.0
        if hasattr(hippo, "state") and hippo.state is not None:
            if hasattr(hippo.state, "ca1_spikes") and hippo.state.ca1_spikes is not None:
                ca1_spikes = hippo.state.ca1_spikes.sum().item()

        # Normalize by hippocampus size
        hippo_size = getattr(hippo.config, "n_output", 128) if hasattr(hippo, "config") else 128
        ca1_normalized = ca1_spikes / max(1, hippo_size)

        # Layer activity
        dg_spikes = 0.0
        ca3_spikes = 0.0
        if hasattr(hippo, "state") and hippo.state is not None:
            if hasattr(hippo.state, "dg_spikes") and hippo.state.dg_spikes is not None:
                dg_spikes = hippo.state.dg_spikes.sum().item()
            if hasattr(hippo.state, "ca3_spikes") and hippo.state.ca3_spikes is not None:
                ca3_spikes = hippo.state.ca3_spikes.sum().item()

        return HippocampusDiagnostics(
            ca1_total_spikes=ca1_spikes,
            ca1_normalized=ca1_normalized,
            dg_spikes=dg_spikes,
            ca3_spikes=ca3_spikes,
            ca1_spikes=ca1_spikes,
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
        diagnostics: Dict[str, Any] = {
            "components": {},
            "spike_counts": {},
            "pathways": {},
            "oscillators": {},
            "neuromodulators": {},
        }

        # Component diagnostics
        for name, component in self.components.items():
            if hasattr(component, "get_diagnostics"):
                diagnostics["components"][name] = component.get_diagnostics()  # type: ignore[operator]

            # Collect spike counts for health monitoring
            if hasattr(component, "state") and component.state is not None:
                if hasattr(component.state, "spikes") and component.state.spikes is not None:
                    diagnostics["spike_counts"][name] = component.state.spikes.sum().item()  # type: ignore[operator]

        # Pathway diagnostics (from PathwayManager)
        diagnostics["pathways"] = self.pathway_manager.get_diagnostics()

        # Oscillator diagnostics
        diagnostics["oscillators"] = {
            "delta": {
                "phase": self.oscillators.delta.phase,
                "frequency_hz": self.oscillators.delta.frequency_hz,
                "signal": self.oscillators.delta.signal,
            },
            "theta": {
                "phase": self.oscillators.theta.phase,
                "frequency_hz": self.oscillators.theta.frequency_hz,
                "signal": self.oscillators.theta.signal,
            },
            "alpha": {
                "phase": self.oscillators.alpha.phase,
                "frequency_hz": self.oscillators.alpha.frequency_hz,
                "signal": self.oscillators.alpha.signal,
            },
            "beta": {
                "phase": self.oscillators.beta.phase,
                "frequency_hz": self.oscillators.beta.frequency_hz,
                "signal": self.oscillators.beta.signal,
            },
            "gamma": {
                "phase": self.oscillators.gamma.phase,
                "frequency_hz": self.oscillators.gamma.frequency_hz,
                "signal": self.oscillators.gamma.signal,
            },
            "ripple": {
                "phase": self.oscillators.ripple.phase,
                "frequency_hz": self.oscillators.ripple.frequency_hz,
                "signal": self.oscillators.ripple.signal,
            },
        }

        # Neuromodulator diagnostics
        diagnostics["neuromodulators"] = self.neuromodulator_manager.get_diagnostics()

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
        is_match = getattr(self, "_last_is_match", False)
        correct = getattr(self, "_last_correct", False)

        return BrainSystemDiagnostics(
            trial_num=trial_num,
            is_match=is_match,
            selected_action=self._last_action or 0,
            correct=correct,
            striatum=self._collect_striatum_diagnostics(),
            hippocampus=self._collect_hippocampus_diagnostics(),
        )

    def get_full_state(self) -> StateDict:
        """Get complete state for checkpointing.

        Returns:
            Dict containing all component states, neuromodulator states, and metadata

        Note:
            Uses "regions" key (not "components") for CheckpointManager compatibility.
            DynamicBrain's components are stored as regions in the checkpoint format.
        """
        state: Dict[str, Any] = {
            "brain_config": self.brain_config,
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
        regions_dict: Dict[str, Any] = state["regions"]
        for name, component in cast(Dict[str, NeuralRegion], self.components).items():
            regions_dict[name] = component.get_full_state()

        # Get pathway states via PathwayManager
        state["pathways"] = self.pathway_manager.get_state()  # type: ignore[attr-defined]

        # Get oscillator states
        state["oscillators"] = {  # type: ignore[index]
            "delta": self.oscillators.delta.get_state(),  # type: ignore[attr-defined]
            "theta": self.oscillators.theta.get_state(),  # type: ignore[attr-defined]
            "alpha": self.oscillators.alpha.get_state(),  # type: ignore[attr-defined]
            "beta": self.oscillators.beta.get_state(),  # type: ignore[attr-defined]
            "gamma": self.oscillators.gamma.get_state(),  # type: ignore[attr-defined]
            "ripple": self.oscillators.ripple.get_state(),  # type: ignore[attr-defined]
        }

        # Get neuromodulator states using proper get_state() methods
        if self.neuromodulator_manager.vta is not None:  # type: ignore[attr-defined]
            if hasattr(self.neuromodulator_manager.vta, "get_state"):  # type: ignore[attr-defined]
                state["neuromodulators"]["vta"] = self.neuromodulator_manager.vta.get_state()  # type: ignore[attr-defined, index]
            else:
                # Fallback for VTA without get_state()
                state["neuromodulators"]["vta"] = {  # type: ignore[index]
                    "global_dopamine": self.neuromodulator_manager.vta._global_dopamine,  # type: ignore[attr-defined]
                    "tonic_dopamine": self.neuromodulator_manager.vta._tonic_dopamine,  # type: ignore[attr-defined]
                    "phasic_dopamine": self.neuromodulator_manager.vta._phasic_dopamine,  # type: ignore[attr-defined]
                }
        if self.neuromodulator_manager.locus_coeruleus is not None:  # type: ignore[attr-defined]
            if hasattr(self.neuromodulator_manager.locus_coeruleus, "get_state"):  # type: ignore[attr-defined]
                state["neuromodulators"][  # type: ignore[index]
                    "locus_coeruleus"
                ] = self.neuromodulator_manager.locus_coeruleus.get_state()  # type: ignore[attr-defined]
            else:
                # Fallback for LC without get_state()
                state["neuromodulators"]["locus_coeruleus"] = {  # type: ignore[index]
                    "norepinephrine": self.neuromodulator_manager.locus_coeruleus.get_norepinephrine(  # type: ignore[attr-defined]
                        apply_homeostasis=False
                    ),
                }
        if self.neuromodulator_manager.nucleus_basalis is not None:  # type: ignore[attr-defined]
            if hasattr(self.neuromodulator_manager.nucleus_basalis, "get_state"):  # type: ignore[attr-defined]
                state["neuromodulators"][  # type: ignore[index]
                    "nucleus_basalis"
                ] = self.neuromodulator_manager.nucleus_basalis.get_state()  # type: ignore[attr-defined]
            else:
                # Fallback for NB without get_state()
                state["neuromodulators"]["nucleus_basalis"] = {  # type: ignore[index]
                    "acetylcholine": self.neuromodulator_manager.nucleus_basalis.get_acetylcholine(  # type: ignore[attr-defined]
                        apply_homeostasis=False
                    ),
                }

        return state  # type: ignore[return-value]

    def save_checkpoint(
        self,
        path: Union[str, Path],
        metadata: Optional[CheckpointMetadata] = None,
        **kwargs,
    ) -> CheckpointMetadata:
        """Save checkpoint (wrapper for CheckpointManager.save).

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
    ) -> CheckpointMetadata:
        """Load checkpoint (wrapper for CheckpointManager.load).

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
        """
        # Load time
        self._current_time = float(state.get("current_time", 0.0))

        # Load topology
        topology_value = state.get("topology", self._topology)
        if isinstance(topology_value, dict):
            self._topology = topology_value

        # Load component states
        component_states: Dict[str, Any] = state.get("regions", {})
        for name, component_state in component_states.items():
            if name in self.components:
                self.components[name].load_full_state(component_state)  # type: ignore[operator]

        # Load pathway states via PathwayManager
        if "pathways" in state:
            self.pathway_manager.load_state(state["pathways"])

        # **CRITICAL**: Move entire brain to correct device after loading state
        # This handles CPU→CUDA transfer by moving all parameters, buffers,
        # and registered submodules to the target device
        self.to(self.device)

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
                if hasattr(self.neuromodulator_manager.vta, "set_state"):
                    self.neuromodulator_manager.vta.set_state(neuromod_state["vta"])
                else:
                    # Fallback for VTA without set_state()
                    vta_state = neuromod_state["vta"]
                    self.neuromodulator_manager.vta._tonic_dopamine = vta_state.get(
                        "tonic_dopamine", 0.0
                    )
                    self.neuromodulator_manager.vta._phasic_dopamine = vta_state.get(
                        "phasic_dopamine", 0.0
                    )
                    self.neuromodulator_manager.vta._global_dopamine = vta_state.get(
                        "global_dopamine", 0.0
                    )
            if (
                "locus_coeruleus" in neuromod_state
                and self.neuromodulator_manager.locus_coeruleus is not None
            ):
                if hasattr(self.neuromodulator_manager.locus_coeruleus, "set_state"):
                    self.neuromodulator_manager.locus_coeruleus.set_state(
                        neuromod_state["locus_coeruleus"]
                    )
            if (
                "nucleus_basalis" in neuromod_state
                and self.neuromodulator_manager.nucleus_basalis is not None
            ):
                if hasattr(self.neuromodulator_manager.nucleus_basalis, "set_state"):
                    self.neuromodulator_manager.nucleus_basalis.set_state(
                        neuromod_state["nucleus_basalis"]
                    )

        # Load growth history (if present)
        if "growth_history" in state:
            self._growth_history = [
                GrowthEvent.from_dict(event_dict) for event_dict in state["growth_history"]
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
