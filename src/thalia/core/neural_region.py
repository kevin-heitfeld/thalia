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

from thalia.learning import create_strategy
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses.weight_init import WeightInitializer


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


class NeuralRegion(nn.Module):
    """Base class for brain regions with biologically accurate synaptic inputs.

    This is a NEW hierarchy for v2.0 architecture, independent of the legacy
    component system (LearnableComponent, etc.). Regions are just nn.Module
    with specialized structure for multi-source synaptic inputs.

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
            **kwargs: Additional arguments for NeuralComponent compatibility
        """
        # Initialize as nn.Module (new v2.0 hierarchy)
        super().__init__()

        self.n_neurons = n_neurons
        self.n_input = 0  # Updated as sources are added
        self.n_output = n_neurons
        self.device = device
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
            scale=weight_scale,
            device=self.device,
        )

        # Register as parameter for gradient tracking and state_dict
        # Use ParameterDict to ensure proper device movement and state saving
        self.synaptic_weights[source_name] = nn.Parameter(weights)

        # Track input size
        self.input_sources[source_name] = n_input

        # Update total n_input for NeuralComponent compatibility
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

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
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


__all__ = ["NeuralRegion"]
