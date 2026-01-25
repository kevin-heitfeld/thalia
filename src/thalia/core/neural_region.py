"""NeuralRegion: Biologically accurate region with synaptic inputs at dendrites.

This is the base class for brain regions where:
- Weights live at TARGET dendrites (not in pathways/axons)
- Learning rules are region-specific (per-source customization)
- Multi-source integration is natural (Dict[str, Tensor] input)

Biological accuracy:
- Axons = pure routing (delay only, no weights)
- Dendrites = synaptic weights (at target)
- Soma = integration + spiking
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import torch
import torch.nn as nn

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import WeightInitializer
from thalia.core.protocols.component import BrainComponentMixin
from thalia.learning import create_strategy
from thalia.learning.strategy_mixin import LearningStrategyMixin
from thalia.mixins.diagnostics_mixin import DiagnosticsMixin
from thalia.mixins.growth_mixin import GrowthMixin
from thalia.mixins.resettable_mixin import ResettableMixin
from thalia.mixins.state_loading_mixin import StateLoadingMixin
from thalia.neuromodulation.mixin import NeuromodulatorMixin
from thalia.typing import InputSizes, LearningStrategies, SourceOutputs, StateDict


# Custom warning for performance issues
class PerformanceWarning(UserWarning):
    """Warning for performance-degrading configurations."""


# Type hint for learning strategies (duck typing)
class LearningStrategy(Protocol):
    """Protocol for learning strategies that can update synaptic weights."""

    def compute_update(
        self, weights: torch.Tensor, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, Dict[str, Any]]: ...


class NeuralRegion(
    nn.Module,
    BrainComponentMixin,
    NeuromodulatorMixin,
    GrowthMixin,
    ResettableMixin,
    DiagnosticsMixin,
    StateLoadingMixin,
    LearningStrategyMixin,
):
    """Base class for brain regions with biologically accurate synaptic inputs.

    Mixins provide (MRO order):
    - nn.Module: PyTorch module functionality (FIRST for proper __init__)
    - BrainComponentMixin: Oscillator phase properties and defaults
    - NeuromodulatorMixin: Dopamine, acetylcholine, norepinephrine control
    - GrowthMixin: Dynamic expansion (grow_input/grow_output)
    - ResettableMixin: State reset helpers
    - DiagnosticsMixin: Health monitoring and metrics
    - StateLoadingMixin: Common state restoration (load_state with helpers)
    - LearningStrategyMixin: Pluggable learning rules (STDP, BCM, three-factor, etc.)

    Regions in the new architecture:
    1. Own their synaptic weights (one weight matrix per input source)
    2. Define learning rules (per-source plasticity)
    3. Integrate multi-source inputs naturally

    Subclassing:
        Regions with internal structure (like LayeredCortex) should:
        1. Call super().__init__() to get synaptic_weights dict
        2. Define internal neurons/weights for within-region processing
        3. Override forward() to apply synaptic weights then internal processing
    """

    def __init__(
        self,
        n_neurons: int,
        neuron_config: Optional[ConductanceLIFConfig] = None,
        default_learning_strategy: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize neural region with neurons and empty synaptic weight dict.

        Args:
            n_neurons: Number of neurons in this region
            neuron_config: Configuration for ConductanceLIF neurons
            default_learning_strategy: Default plasticity strategy for new input sources
                                       Options: "stdp", "bcm", "hebbian", "three_factor"
            device: Device for computation ("cpu" or "cuda")
            **kwargs: Additional arguments for base class compatibility

        Note:
            dt_ms is NO LONGER stored in NeuralRegion. It's managed by BrainConfig
            and propagated via update_temporal_parameters() when needed.
        """
        # Initialize as nn.Module (new v3.0 hierarchy)
        super().__init__()

        self.n_neurons = n_neurons
        self.n_input = 0  # Updated as sources are added
        self.n_output = n_neurons
        self.device = torch.device(device)  # Regular attribute, not property
        self.default_learning_strategy = default_learning_strategy

        # Plasticity control (for surgery/experiments)
        self.plasticity_enabled = True

        # Create neuron population
        neuron_cfg = neuron_config or ConductanceLIFConfig()
        self.neurons = ConductanceLIF(n_neurons, neuron_cfg, device=self.device)

        # Synaptic weights: one weight matrix per input source
        # Structure: {"thalamus": [n_neurons, 128], "hippocampus": [n_neurons, 200], ...}
        # These are the TARGET dendrites receiving from each source
        self.synaptic_weights: nn.ParameterDict = nn.ParameterDict()

        # Plasticity rules: one learning strategy per input source
        # Structure: {"thalamus": STDPStrategy(...), "hippocampus": BCMStrategy(...), ...}
        self.plasticity_rules: LearningStrategies = {}

        # Track which sources have been added
        self.input_sources: InputSizes = {}  # {source_name: n_input}

        # State
        self.output_spikes: Optional[torch.Tensor] = None

        # Port-based routing infrastructure (ADR-015)
        self._port_outputs: Dict[str, torch.Tensor] = {}
        self._port_sizes: Dict[str, int] = {}
        self._registered_ports: set[str] = set()

    def add_input_source(
        self,
        source_name: str,
        n_input: int,
        learning_strategy=...,  # Sentinel for default
        sparsity: float = 0.2,
        weight_scale: float = 0.3,
    ) -> None:
        """Add synaptic weights for a new input source.

        This creates the dendritic synapses that receive from the specified source.
        Biologically: These are synapses ON this region's neurons, not in the axons.

        Args:
            source_name: Name of source region (e.g., "thalamus", "hippocampus")
            n_input: Number of neurons in source region
            learning_strategy: Plasticity rule for these synapses
                          ... (default) = use region's default_learning_strategy
                          None = explicitly disable learning
                          str = specific rule ("stdp", "bcm", "hebbian", "three_factor")
            sparsity: Connection sparsity (0.0 = dense, 1.0 = no connections)
            weight_scale: Initial weight scale (mean synaptic strength)

        Raises:
            ValueError: If source_name already exists
        """
        if source_name in self.input_sources:
            raise ValueError(f"Input source '{source_name}' already exists")

        # Create weight matrix [n_neurons, n_input]
        # Biologically: Each postsynaptic neuron receives from ~20% of presynaptic neurons
        weights = WeightInitializer.sparse_random(
            n_output=self.n_neurons,
            n_input=n_input,
            sparsity=sparsity,
            weight_scale=weight_scale,
            device=self.device,
        )

        # Register as parameter for gradient tracking and state_dict
        # Use ParameterDict to ensure proper device movement and state saving
        self.synaptic_weights[source_name] = nn.Parameter(weights)

        self.input_sources[source_name] = n_input

        # Update total n_input for tracking
        self.n_input = sum(self.input_sources.values())

        # Create plasticity rule based on learning_strategy parameter
        # ... (sentinel) → use region's default_learning_strategy
        # None → explicitly disable learning
        # str → use specific rule
        if learning_strategy is ...:
            # Use default if set
            rule = self.default_learning_strategy
        elif learning_strategy is None:
            # Explicitly disabled
            rule = None
        else:
            # Specific rule provided
            rule = learning_strategy

        if rule:
            self.plasticity_rules[source_name] = create_strategy(
                rule,
                learning_rate=0.001,  # Conservative default
            )

    def forward(self, inputs: SourceOutputs) -> torch.Tensor:
        """Process inputs through synapses and neurons.

        This is the core of biological integration:
        1. Each source applies its synaptic weights → currents
        2. Sum all synaptic currents
        3. Integrate in neuron soma
        4. Apply learning rules to update synaptic weights

        Args:
            inputs: Dict mapping source names to spike vectors
                   Example: {"thalamus": [128], "hippocampus": [200]}
                   Sources not in dict are treated as silent (no spikes)

        Returns:
            Output spikes [n_neurons]

        Raises:
            ValueError: If input source not registered with add_input_source()
        """
        # Infer device from actual parameter location (in case .to(device) was called)
        device = (
            next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else torch.device(self.device)
        )

        # Compute synaptic currents for each source
        g_exc_total = torch.zeros(self.n_neurons, device=device)

        for source_name, input_spikes in inputs.items():
            if source_name not in self.synaptic_weights:
                raise ValueError(
                    f"No synaptic weights for source '{source_name}'. "
                    f"Call add_input_source('{source_name}', n_input=...) first."
                )

            # Apply synaptic weights: dendrites receive input
            weights = self.synaptic_weights[source_name]
            g_exc = torch.matmul(weights, input_spikes.float())
            g_exc_total += g_exc

        # Integrate in neuron soma (could add inhibition here)
        # Simple version: excitation only
        g_inh = torch.zeros(self.n_neurons, device=device)
        output_spikes, _ = self.neurons(g_exc_total, g_inh)

        # Apply plasticity AFTER computing output spikes (local learning requires both pre and post)
        for source_name, input_spikes in inputs.items():
            if self.plasticity_enabled and source_name in self.plasticity_rules:
                plasticity = self.plasticity_rules[source_name]

                # Compute weight update (local learning rule)
                new_weights, _ = plasticity.compute_update(
                    weights=self.synaptic_weights[source_name],
                    pre=input_spikes,
                    post=output_spikes,
                )

                # Update synaptic weights in-place
                self.synaptic_weights[source_name].data = new_weights

        # Store state for next learning update
        self.output_spikes = output_spikes

        return output_spikes

    # =========================================================================
    # Port-Based Routing (ADR-015)
    # =========================================================================

    def register_output_port(self, port_name: str, size: int) -> None:
        """Register an output port for routing.

        Ports enable routing specific outputs to specific targets, matching biological
        reality where different cell types project to different targets (e.g., L6a→TRN, L6b→relay).

        Args:
            port_name: Name of the port (e.g., "l6a", "l6b")
            size: Number of neurons in this output

        Raises:
            ValueError: If port already registered
        """
        if port_name in self._registered_ports:
            raise ValueError(f"Port '{port_name}' already registered in {self.__class__.__name__}")

        self._port_sizes[port_name] = size
        self._registered_ports.add(port_name)

    def set_port_output(self, port_name: str, spikes: torch.Tensor) -> None:
        """Store output for a specific port.

        Called during forward() to set outputs for each registered port.

        Args:
            port_name: Name of the port
            spikes: Spike tensor for this port (shape: [size])

        Raises:
            ValueError: If port not registered or size mismatch
        """
        if port_name not in self._registered_ports:
            raise ValueError(
                f"Port '{port_name}' not registered in {self.__class__.__name__}. "
                f"Available ports: {sorted(self._registered_ports)}"
            )

        expected_size = self._port_sizes[port_name]
        if spikes.shape[0] != expected_size:
            raise ValueError(
                f"Port '{port_name}' in {self.__class__.__name__} expects {expected_size} "
                f"neurons, got {spikes.shape[0]}"
            )

        self._port_outputs[port_name] = spikes

    def get_port_output(self, port_name: Optional[str] = None) -> torch.Tensor:
        """Get output from a specific port.

        Used by AxonalProjection to route from specific outputs.

        Args:
            port_name: Name of the port. If None, returns "default" port.

        Returns:
            Spike tensor from the specified port

        Raises:
            ValueError: If port not found or no output set

        Example:
            >>> spikes = cortex.get_port_output("l6a")  # Get L6a output
            >>> spikes = cortex.get_port_output()  # Get default output
        """
        if port_name is None:
            port_name = "default"

        if port_name not in self._port_outputs:
            raise ValueError(
                f"No output set for port '{port_name}' in {self.__class__.__name__}. "
                f"Available outputs: {sorted(self._port_outputs.keys())}"
            )

        return self._port_outputs[port_name]

    def clear_port_outputs(self) -> None:
        """Clear all port outputs.

        Called at the start of forward() to reset outputs from previous timestep.
        Regions using ports should call this at the beginning of their forward() method.
        """
        self._port_outputs.clear()

    def get_registered_ports(self) -> list[str]:
        """Get list of all registered port names.

        Returns:
            Sorted list of port names

        Example:
            >>> ports = cortex.get_registered_ports()
            >>> # ['default', 'l23', 'l5', 'l6a', 'l6b']
        """
        return sorted(self._registered_ports)

    def has_port(self, port_name: str) -> bool:
        """Check if a port is registered.

        Args:
            port_name: Name of the port to check

        Returns:
            True if port is registered, False otherwise

        Example:
            >>> if region.has_port("l6a"):
            >>>     spikes = region.get_port_output("l6a")
        """
        return port_name in self._registered_ports

    # =========================================================================
    # State Reset
    # =========================================================================

    def reset_state(self) -> None:
        """Reset neuron state and learning traces."""
        self.neurons.reset_state()
        self.output_spikes = None
        self.clear_port_outputs()  # Clear port-based routing state

        # Reset learning rule states
        for strategy in self.plasticity_rules.values():
            if hasattr(strategy, "reset_state"):
                strategy.reset_state()

    def _reset_subsystems(self, *subsystem_names: str) -> None:
        """Reset multiple subsystems by calling their reset_state() methods.

        Convenience helper to avoid repetitive code in reset_state() implementations.

        Args:
            *subsystem_names: Names of attributes to reset (must have reset_state())

        Example:
            >>> def reset_state(self):
            >>>     super().reset_state()
            >>>     self._reset_subsystems('_trace_manager', 'climbing_fiber')
        """
        for name in subsystem_names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, "reset_state"):
                    subsystem.reset_state()

    # =========================================================================
    # BrainComponent Protocol Implementation (Required Methods)
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for neurons and learning strategies.

        Called by brain when dt_ms changes. Propagates update to:
        - Neurons (conductance decay factors)
        - Learning strategies (trace decay factors)

        Args:
            dt_ms: New timestep in milliseconds
        """
        # Update neuron decay factors
        if hasattr(self, "neurons"):
            self.neurons.update_temporal_parameters(dt_ms)

        # Update learning strategy trace decay
        for strategy in self.plasticity_rules.values():
            if hasattr(strategy, "update_temporal_parameters"):
                strategy.update_temporal_parameters(dt_ms)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get current activity and health metrics.

        Returns basic diagnostics. Subclasses should override to add region-specific metrics.

        Returns:
            Dictionary with firing rates, weight stats, and other metrics
        """
        diagnostics = {}

        # Firing rate
        if self.output_spikes is not None:
            firing_rate = self.output_spikes.float().mean().item()
            diagnostics["firing_rate"] = firing_rate

        # Weight statistics for each source
        for source_name, weights in self.synaptic_weights.items():
            w = weights.detach()
            diagnostics[f"{source_name}_weight_mean"] = w.mean().item()
            diagnostics[f"{source_name}_weight_std"] = w.std().item()

        # Neuron count
        diagnostics["n_neurons"] = self.n_neurons
        diagnostics["n_sources"] = len(self.input_sources)

        return diagnostics

    def check_health(self) -> Any:
        """Check for pathological states.

        Returns basic health report. Uses DiagnosticsMixin if available.

        Returns:
            HealthReport with detected issues
        """
        from thalia.diagnostics.health_monitor import HealthReport

        # Basic healthy report - subclasses can override for detailed checks
        return HealthReport(
            is_healthy=True,
            overall_severity=0.0,
            issues=[],
            summary=f"{self.__class__.__name__}: Healthy",
            metrics=self.get_diagnostics(),
        )

    def get_capacity_metrics(self) -> Any:
        """Get capacity utilization metrics for growth decisions.

        Returns:
            CapacityMetrics with growth recommendations
        """
        from thalia.coordination.growth import CapacityMetrics

        # Basic metrics - subclasses can override for sophisticated analysis
        firing_rate = 0.0
        active_neurons = 0
        if self.output_spikes is not None:
            firing_rate = self.output_spikes.float().mean().item()
            active_neurons = int(self.output_spikes.sum().item())

        # Utilization estimate (firing rate as proxy)
        utilization = firing_rate

        synapse_count = sum(self.input_sources.values()) * self.n_neurons

        return CapacityMetrics(
            utilization=utilization,
            total_neurons=self.n_neurons,
            active_neurons=active_neurons,
            growth_recommended=False,
            growth_amount=0,
            firing_rate=firing_rate,
            synapse_usage=0.0 if synapse_count == 0 else float(active_neurons) / synapse_count,
            synapse_count=synapse_count,
            growth_reason="",
        )

    def get_full_state(self) -> StateDict:
        """Serialize complete component state for checkpointing.

        Returns:
            Dictionary with weights, config, and metadata
        """
        state = {
            "type": self.__class__.__name__,
            "n_neurons": self.n_neurons,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "device": str(self.device),
            "dt_ms": self.dt_ms,
            "default_learning_strategy": self.default_learning_strategy,
            "input_sources": self.input_sources.copy(),
            "synaptic_weights": {
                name: weights.detach().cpu() for name, weights in self.synaptic_weights.items()
            },
        }
        return state

    def load_full_state(self, state: StateDict) -> None:
        """Restore component from checkpoint.

        Args:
            state: Dictionary from get_full_state()
        """
        # Validate compatibility
        if state["n_neurons"] != self.n_neurons:
            raise ValueError(
                f"State mismatch: saved {state['n_neurons']} neurons, "
                f"current has {self.n_neurons}"
            )

        # Restore synaptic weights
        for name, weights_cpu in state["synaptic_weights"].items():
            if name in self.synaptic_weights:
                self.synaptic_weights[name].data = weights_cpu.to(self.device)
            else:
                # Add missing source
                n_input = weights_cpu.shape[1]
                self.add_input_source(name, n_input=n_input, learning_strategy=None)
                self.synaptic_weights[name].data = weights_cpu.to(self.device)

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Dict[str, float] | None = None,
        theta_slot: int = 0,
        coupled_amplitudes: Dict[str, float] | None = None,
    ) -> None:
        """Set oscillator phases for neural oscillations (default: no-op).

        Subclasses with oscillators should override to update phase state.

        Args:
            phases: Dictionary of oscillator phases (e.g., {'theta': 0.5, 'gamma': 0.25})
            signals: Optional oscillator signals (amplitudes)
            theta_slot: Current theta slot for sequence learning
            coupled_amplitudes: Optional coupled oscillator amplitudes
        """
        # Default: no oscillators, subclasses override if needed

    def grow_source(
        self,
        source_name: str,
        new_size: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.2,
        weight_scale: float = 0.3,
    ) -> None:
        """Grow input size for a specific source (expands weight columns).

        **Multi-Source Architecture**: Each input source has its own weight matrix.
        This method expands the input dimension (columns) for a specific source
        while preserving existing learned weights.

        When upstream regions grow their output, call this method to expand the
        corresponding input weights.

        Args:
            source_name: Name of source to grow (e.g., "thalamus", "cortex:l5")
            new_size: New total size for this source's input dimension
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new weights (0.0-1.0)
            weight_scale: Initial weight scale for new connections

        Raises:
            KeyError: If source not found in synaptic_weights

        Example:
            >>> # Thalamus grows from 128 to 148 neurons
            >>> thalamus.grow_output(20)
            >>> # Update cortex weights for thalamic input
            >>> cortex.grow_source("thalamus", new_size=148)
        """
        if source_name not in self.synaptic_weights:
            raise KeyError(
                f"Source '{source_name}' not found in synaptic_weights. "
                f"Available sources: {list(self.synaptic_weights.keys())}"
            )

        old_weights = self.synaptic_weights[source_name]
        old_n_input = old_weights.shape[1]
        n_new = new_size - old_n_input

        if n_new <= 0:
            return  # No growth needed

        # Initialize new input columns
        if initialization == "xavier":
            new_cols = WeightInitializer.xavier(
                n_output=self.n_neurons,
                n_input=n_new,
                gain=weight_scale,
                device=self.device,
            )
        elif initialization == "uniform":
            new_cols = WeightInitializer.uniform(
                n_output=self.n_neurons,
                n_input=n_new,
                low=0.0,
                high=weight_scale,
                device=self.device,
            )
        else:  # sparse_random (default)
            new_cols = WeightInitializer.sparse_random(
                n_output=self.n_neurons,
                n_input=n_new,
                sparsity=sparsity,
                weight_scale=weight_scale,
                device=self.device,
            )

        # Concatenate new columns to existing weights
        expanded_weights = torch.cat([old_weights, new_cols], dim=1)
        self.synaptic_weights[source_name] = nn.Parameter(expanded_weights)

        # Update input_sources tracking
        self.input_sources[source_name] = new_size
        self.n_input = sum(self.input_sources.values())

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow component's output dimension by adding neurons.

        For NeuralRegion, this adds neurons and rows to synaptic weight matrices.

        Args:
            n_new: Number of output neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity for new neurons
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.grow_output() not yet implemented. "
            f"See docs/architecture/UNIFIED_GROWTH_API.md for implementation guide."
        )

    @property
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""
        return torch.float32


__all__ = ["NeuralRegion"]
