"""NeuralRegion: Biologically accurate region with synaptic inputs at dendrites.

This is the new base class for brain regions in v2.0 architecture where:
- Weights live at TARGET dendrites (not in pathways/axons)
- Learning rules are region-specific (per-source customization)
- Multi-source integration is natural (Dict[str, Tensor] input)

Key differences from legacy architecture:
- OLD: SpikingPathway has weights → Region.forward(tensor)
- NEW: Region has synaptic_weights dict → Region.forward(dict_of_tensors)

Biological accuracy:
- Axons = pure routing (delay only, no weights)
- Dendrites = synaptic weights (at target)
- Soma = integration + spiking
"""

from typing import Dict, Optional, Any, Protocol
import torch
import torch.nn as nn

from thalia.typing import SourceOutputs, DiagnosticsDict, StateDict
from thalia.learning import create_strategy
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import WeightInitializer
from thalia.mixins.diagnostics_mixin import DiagnosticsMixin
from thalia.mixins.growth_mixin import GrowthMixin
from thalia.mixins.resettable_mixin import ResettableMixin
from thalia.mixins.state_loading_mixin import StateLoadingMixin
from thalia.learning.strategy_mixin import LearningStrategyMixin
from thalia.neuromodulation.mixin import NeuromodulatorMixin
from thalia.core.protocols.component import BrainComponentMixin


# Custom warning for performance issues
class PerformanceWarning(UserWarning):
    """Warning for performance-degrading configurations."""
    pass

# Type hint for learning strategies (duck typing)
class LearningStrategy(Protocol):
    """Protocol for learning strategies that can update synaptic weights."""
    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        ...


class NeuralRegion(nn.Module, BrainComponentMixin, NeuromodulatorMixin, GrowthMixin, ResettableMixin, DiagnosticsMixin, StateLoadingMixin, LearningStrategyMixin):
    """Base class for brain regions with biologically accurate synaptic inputs.

    This is a NEW hierarchy for v3.0 architecture, independent of the legacy
    BrainComponent system. Regions are nn.Module with specialized mixins.

    **Note**: NeuralRegion does NOT inherit from BrainComponentBase. It's a simpler,
    cleaner architecture for v3.0. Regions informally implement the BrainComponent
    protocol through mixins, but aren't bound by the abstract base class.

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

    Example:
        >>> # Create region with default learning rule
        >>> region = NeuralRegion(
        ...     n_neurons=500,
        ...     neuron_config=ConductanceLIFConfig(),
        ...     default_learning_rule="stdp"
        ... )
        >>>
        >>> # Add input sources with their synaptic weights
        >>> region.add_input_source("thalamus", n_input=128)  # Uses default STDP
        >>> region.add_input_source("hippocampus", n_input=200, learning_rule="bcm")  # Override
        >>>
        >>> # Forward pass with multi-source input
        >>> outputs = region.forward({
        ...     "thalamus": thalamic_spikes,      # [128]
        ...     "hippocampus": hippocampal_spikes # [200]
        ... })  # Returns [500]

    Subclassing:
        Regions with internal structure (like LayeredCortex) should:
        1. Call super().__init__() to get synaptic_weights dict
        2. Define internal neurons/weights for within-region processing
        3. Override forward() to apply synaptic weights then internal processing

        Example:
            class LayeredCortex(NeuralRegion):
                def __init__(self, ...):
                    super().__init__(n_neurons=l23_size + l5_size, ...)

                    # Internal layers
                    self.l4_neurons = ConductanceLIF(l4_size, ...)
                    self.l23_neurons = ConductanceLIF(l23_size, ...)

                    # Internal weights (within cortex)
                    self.w_l4_l23 = nn.Parameter(...)

                    # External weights come from self.synaptic_weights dict!

                def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
                    # Apply synaptic weights for thalamic input
                    thalamic_current = self._apply_synapses("thalamus", inputs["thalamus"])

                    # Internal cortical processing
                    l4_spikes = self.l4_neurons(thalamic_current)
                    l23_spikes = self._internal_l4_to_l23(l4_spikes)

                    return l23_spikes
    """

    def __init__(
        self,
        n_neurons: int,
        neuron_config: Optional[ConductanceLIFConfig] = None,
        default_learning_rule: Optional[str] = None,
        device: str = "cpu",
        dt_ms: float = 1.0,
        **kwargs
    ):
        """Initialize neural region with neurons and empty synaptic weight dict.

        Args:
            n_neurons: Number of neurons in this region
            neuron_config: Configuration for ConductanceLIF neurons
            default_learning_rule: Default plasticity rule for new input sources
                                   Options: "stdp", "bcm", "hebbian", "three_factor"
            device: Device for computation ("cpu" or "cuda")
            dt_ms: Simulation timestep in milliseconds
            **kwargs: Additional arguments for base class compatibility
        """
        # Initialize as nn.Module (new v3.0 hierarchy)
        super().__init__()

        self.n_neurons = n_neurons
        self.n_input = 0  # Updated as sources are added
        self.n_output = n_neurons
        self.device = torch.device(device)  # Regular attribute, not property
        self.dt_ms = dt_ms
        self.default_learning_rule = default_learning_rule

        # Create neuron population
        neuron_cfg = neuron_config or ConductanceLIFConfig()
        self.neurons = ConductanceLIF(n_neurons, neuron_cfg)

        # Synaptic weights: one weight matrix per input source
        # Structure: {"thalamus": [n_neurons, 128], "hippocampus": [n_neurons, 200], ...}
        # These are the TARGET dendrites receiving from each source
        self.synaptic_weights: nn.ParameterDict = nn.ParameterDict()

        # Plasticity rules: one learning strategy per input source
        # Structure: {"thalamus": STDPStrategy(...), "hippocampus": BCMStrategy(...), ...}
        self.plasticity_rules: Dict[str, LearningStrategy] = {}

        # Track which sources have been added
        self.input_sources: Dict[str, int] = {}  # {source_name: n_input}

        # Learning control
        self.plasticity_enabled: bool = True

        # State
        self.output_spikes: Optional[torch.Tensor] = None

    def add_input_source(
        self,
        source_name: str,
        n_input: int,
        learning_rule: Optional[str] = ...,  # Sentinel: ... = use default, None = no learning, str = specific rule
        sparsity: float = 0.2,
        weight_scale: float = 0.3,
    ) -> None:
        """Add synaptic weights for a new input source.

        This creates the dendritic synapses that receive from the specified source.
        Biologically: These are synapses ON this region's neurons, not in the axons.

        Args:
            source_name: Name of source region (e.g., "thalamus", "hippocampus")
            n_input: Number of neurons in source region
            learning_rule: Plasticity rule for these synapses
                          ... (default) = use region's default_learning_rule
                          None = explicitly disable learning for this source
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

        # Create plasticity rule based on learning_rule parameter
        # ... (default sentinel) → use region's default_learning_rule
        # None → explicitly disable learning
        # str → use specific rule
        if learning_rule is ...:
            # Use default if set
            rule = self.default_learning_rule
        elif learning_rule is None:
            # Explicitly disabled
            rule = None
        else:
            # Specific rule provided
            rule = learning_rule

        if rule:
            self.plasticity_rules[source_name] = create_strategy(
                rule,
                learning_rate=0.001,  # Conservative default
            )

    def forward(self, inputs: SourceOutputs, **kwargs) -> torch.Tensor:
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
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device(self.device)

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

            # Apply plasticity if learning enabled
            if source_name in self.plasticity_rules:
                plasticity = self.plasticity_rules[source_name]

                # Compute weight update (local learning rule)
                new_weights, _ = plasticity.compute_update(
                    weights=weights,
                    pre=input_spikes,
                    post=self.output_spikes if self.output_spikes is not None else torch.zeros(self.n_neurons, device=device),
                )

                # Update synaptic weights in-place
                self.synaptic_weights[source_name].data = new_weights

        # Integrate in neuron soma (could add inhibition here)
        # Simple version: excitation only
        g_inh = torch.zeros(self.n_neurons, device=device)
        output_spikes, _ = self.neurons(g_exc_total, g_inh)

        # Store state for next learning update
        self.output_spikes = output_spikes

        return output_spikes

    def reset_state(self) -> None:
        """Reset neuron state and learning traces."""
        self.neurons.reset_state()
        self.output_spikes = None

        # Reset learning rule states
        for strategy in self.plasticity_rules.values():
            if hasattr(strategy, 'reset_state'):
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
                if subsystem is not None and hasattr(subsystem, 'reset_state'):
                    subsystem.reset_state()

    # =========================================================================
    # BrainComponent Protocol Implementation (Required Methods)
    # =========================================================================

    def get_diagnostics(self) -> DiagnosticsDict:
        """Get current activity and health metrics.

        Returns basic diagnostics. Subclasses should override to add region-specific metrics.

        Returns:
            Dictionary with firing rates, weight stats, and other metrics
        """
        diagnostics = {}

        # Firing rate
        if self.output_spikes is not None:
            firing_rate = self.output_spikes.float().mean().item()
            diagnostics['firing_rate'] = firing_rate

        # Weight statistics for each source
        for source_name, weights in self.synaptic_weights.items():
            w = weights.detach()
            diagnostics[f'{source_name}_weight_mean'] = w.mean().item()
            diagnostics[f'{source_name}_weight_std'] = w.std().item()

        # Neuron count
        diagnostics['n_neurons'] = self.n_neurons
        diagnostics['n_sources'] = len(self.input_sources)

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
            metrics=self.get_diagnostics()
        )

    def get_capacity_metrics(self) -> Any:
        """Get capacity utilization metrics for growth decisions.

        Returns:
            CapacityMetrics with growth recommendations
        """
        from thalia.coordination.growth import CapacityMetrics

        # Basic metrics - subclasses can override for sophisticated analysis
        firing_rate = 0.0
        if self.output_spikes is not None:
            firing_rate = self.output_spikes.float().mean().item()

        return CapacityMetrics(
            firing_rate=firing_rate,
            weight_saturation=0.0,
            synapse_usage=0.0,
            neuron_count=self.n_neurons,
            synapse_count=sum(self.input_sources.values()) * self.n_neurons,
            growth_recommended=False,
            growth_reason="",
        )

    def get_full_state(self) -> StateDict:
        """Serialize complete component state for checkpointing.

        Returns:
            Dictionary with weights, config, and metadata
        """
        state = {
            'type': self.__class__.__name__,
            'n_neurons': self.n_neurons,
            'n_input': self.n_input,
            'n_output': self.n_output,
            'device': str(self.device),
            'dt_ms': self.dt_ms,
            'default_learning_rule': self.default_learning_rule,
            'input_sources': self.input_sources.copy(),
            'synaptic_weights': {
                name: weights.detach().cpu()
                for name, weights in self.synaptic_weights.items()
            },
            'plasticity_enabled': self.plasticity_enabled,
        }
        return state

    def load_full_state(self, state: StateDict) -> None:
        """Restore component from checkpoint.

        Args:
            state: Dictionary from get_full_state()
        """
        # Validate compatibility
        if state['n_neurons'] != self.n_neurons:
            raise ValueError(
                f"State mismatch: saved {state['n_neurons']} neurons, "
                f"current has {self.n_neurons}"
            )

        # Restore synaptic weights
        for name, weights_cpu in state['synaptic_weights'].items():
            if name in self.synaptic_weights:
                self.synaptic_weights[name].data = weights_cpu.to(self.device)
            else:
                # Add missing source
                n_input = weights_cpu.shape[1]
                self.add_input_source(name, n_input=n_input, learning_rule=None)
                self.synaptic_weights[name].data = weights_cpu.to(self.device)

        # Restore flags
        self.plasticity_enabled = state.get('plasticity_enabled', True)

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
        pass  # Default: no oscillators, subclasses override if needed

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow component's input dimension.

        For NeuralRegion, this adds columns to existing synaptic weight matrices.

        Args:
            n_new: Number of input neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity for new inputs
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.grow_input() not yet implemented. "
            f"Regions typically grow inputs by growing specific sources. "
            f"Override this method if your region supports generic input growth."
        )

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
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
