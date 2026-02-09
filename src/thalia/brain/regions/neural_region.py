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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar, Generic

import torch
import torch.nn as nn

from thalia.brain.configs import NeuralRegionConfig
from thalia.components.synapses import WeightInitializer
from thalia.typing import (
    LayerName,
    PortName,
    RegionLayerSizes,
    RegionSpikesDict,
    SpikesSourceKey,
)
from thalia.utils import validate_finite, validate_spike_tensor

if TYPE_CHECKING:
    from thalia.learning import LearningStrategy


ConfigT = TypeVar('ConfigT', bound=NeuralRegionConfig)


@dataclass
class NeuromodulatorState:
    """Tracks neuromodulator levels for a region, which modulate learning and processing."""

    dopamine: float = 0.2
    """Dopamine level (reward signal, gates learning). Baseline tonic level."""

    acetylcholine: float = 0.3
    """Acetylcholine level (attention, encoding/retrieval mode)."""

    norepinephrine: float = 0.3
    """Norepinephrine level (arousal, gain modulation)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize common state fields."""
        return {
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
        }


class NeuralRegion(nn.Module, ABC, Generic[ConfigT]):
    """Base class for brain regions with biologically accurate synaptic inputs.

    Regions:
    1. Own their synaptic weights (one weight matrix per input source)
    2. Define learning rules (per-source plasticity)
    3. Integrate multi-source inputs naturally

    Subclassing:
        Regions with internal structure (like Cortex) should:
        1. Call super().__init__() to get synaptic_weights dict
        2. Define internal neurons/weights for within-region processing
        3. Override forward() to apply synaptic weights then internal processing
        4. Set OUTPUT_PORTS class attribute for auto port registration
    """

    # Type annotations for config
    config: ConfigT

    OUTPUT_PORTS: Dict[PortName, str] = {}
    """Port name → size attribute name mapping for auto-registration."""

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Device where tensors are located."""
        return torch.device(self.config.device)

    @property
    def dt_ms(self) -> float:
        """Timestep duration in milliseconds."""
        return self.config.dt_ms

    @property
    def dtype(self) -> torch.dtype:
        """Data type for floating point tensors."""
        return torch.float32

    @property
    def n_input(self) -> int:
        """Total number of input neurons across all sources."""
        import warnings
        warnings.warn("NeuralRegion.n_input is deprecated and will be removed in future versions.")
        return sum(self.input_sources.values())

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: ConfigT, region_layer_sizes: RegionLayerSizes):
        """Initialize NeuralRegion with config and layer sizes."""
        super().__init__()

        # ====================================================================
        # VALIDATE INPUT SIZES
        # ====================================================================
        for source_name, size in region_layer_sizes.items():
            if not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"NeuralRegion.__init__: Invalid input size for source '{source_name}': "
                    f"expected non-negative int, got {size}."
                )

        # ====================================================================
        # INITIALIZE INSTANCE VARIABLES
        # ====================================================================
        self.config: ConfigT = config
        self.region_layer_sizes: RegionLayerSizes = region_layer_sizes

        self.neuromodulator_state: NeuromodulatorState = NeuromodulatorState()

        # Synaptic weights: one weight matrix per input source
        # These are the TARGET dendrites receiving from each source
        self.synaptic_weights: nn.ParameterDict = nn.ParameterDict()

        # Track which sources have been added
        self.input_sources: RegionLayerSizes = {}

        # Port-based routing infrastructure (ADR-015)
        self._port_outputs: Dict[str, torch.Tensor] = {}
        self._port_sizes: Dict[str, int] = {}
        self._registered_ports: set[str] = set()

        # Map each input source to its target layer (for runtime routing)
        # e.g., {"thalamus:relay": "l4", "hippocampus:ca1": "l5"}
        self._source_target_layers: Dict[str, str] = {}

        # Initialize oscillator phase tracking (will be updated by Brain via set_oscillator_phases)
        self._oscillator_phases: Dict[str, float] = {}
        self._oscillator_signals: Dict[str, float] = {}
        self._coupled_amplitudes: Dict[str, float] = {}

        # Strategy instance (set by subclass)
        self.learning_strategy: Optional[LearningStrategy] = None

        # Output spikes from last forward pass
        self.output_spikes: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        """Post-initialization to ensure parameters are on correct device and ports registered."""
        # =====================================================================
        # AUTO-REGISTER OUTPUT PORTS
        # =====================================================================
        for port_name, size_attr in self.__class__.OUTPUT_PORTS.items():
            size = getattr(self, size_attr, None)
            assert size is not None and isinstance(size, int) and size > 0, (
                f"Invalid size for output port '{port_name}' in {self.__class__.__name__}. "
            )
            self._register_output_port(port_name, size)

        # =====================================================================
        # ENSURE PARAMETERS ON CORRECT DEVICE
        # =====================================================================
        self.to(self.device)

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        source_name: SpikesSourceKey,
        target_layer: LayerName,
        n_input: int,
        sparsity: float = 0.5,
        weight_scale: float = 2.5,
    ) -> None:
        """Add synaptic weights for a new input source.

        This creates the dendritic synapses that receive from the specified source.
        Biologically: These are synapses ON this region's neurons, not in the axons.

        **Source-Based Routing with Target Layer**:
        For multi-layer regions (Cortex, Hippocampus), inputs target specific layers.
        The weight matrix size is determined by the target layer (e.g., "l4", "l23", "l5").

        Args:
            source_name: Name of source region (e.g., "thalamus:relay", "hippocampus:ca1")
            target_layer: Target layer name (e.g., "l4", "l23", "l5")
            n_input: Number of neurons in source region
            sparsity: Connection sparsity (0.5 = 50% connected, biological default)
            weight_scale: Initial weight scale (2.5 = strong enough to reliably drive postsynaptic firing)

        Raises:
            ValueError: If source_name already exists
        """
        if source_name in self.input_sources:
            raise ValueError(f"Input source '{source_name}' already exists")

        # Determine target layer size from layer name
        # Different inputs go to different layers (e.g., Cortex: thalamus→L4, hippocampus→L5)
        n_output = self._get_target_layer_size(target_layer)

        # Create weight matrix [n_output, n_input]
        # n_output is layer-specific (e.g., L4 for Cortex) or total neurons for simple regions
        weights = WeightInitializer.sparse_random(
            n_output=n_output,
            n_input=n_input,
            sparsity=sparsity,
            weight_scale=weight_scale,
            device=self.device,
        )

        # Use ParameterDict to ensure proper device movement
        self.synaptic_weights[source_name] = nn.Parameter(weights, requires_grad=False)

        # Track source registration
        self.input_sources[source_name] = n_input

        # Store target layer for runtime routing (biological: synapses define the pathway)
        self._source_target_layers[source_name] = target_layer

    def _get_target_layer_size(self, target_layer: LayerName) -> int:
        """Get target layer size directly from layer name."""
        size_attr = f"{target_layer}_size"
        if not hasattr(self, size_attr):
            raise ValueError(
                f"Target layer '{target_layer}' not found in {self.__class__.__name__}. "
                f"Expected attribute '{size_attr}'."
            )
        return getattr(self, size_attr)

    # =========================================================================
    # PORT-BASED ROUTING MANAGEMENT
    # =========================================================================

    def _register_output_port(self, port_name: PortName, size: int) -> None:
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

    def set_port_output(self, port_name: PortName, spikes: torch.Tensor) -> None:
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

    def get_port_output(self, port_name: PortName) -> torch.Tensor:
        """Get output from a specific port.

        Used by AxonalProjection to route from specific outputs.

        Args:
            port_name: Name of the port (required). Must specify explicit port.

        Returns:
            Spike tensor from the specified port. Returns zeros if port is
            registered but hasn't been set yet (e.g., first timestep or
            circular dependencies).

        Raises:
            ValueError: If port not registered
        """
        # Check if port is registered
        if port_name not in self._registered_ports:
            raise ValueError(
                f"Port '{port_name}' not registered in {self.__class__.__name__}. "
                f"Available ports: {sorted(self._registered_ports)}"
            )

        # If port is registered but not set yet, return zeros
        if port_name not in self._port_outputs:
            size = self._port_sizes[port_name]
            return torch.zeros(size, dtype=torch.bool, device=self.device)

        return self._port_outputs[port_name]

    def get_all_port_outputs(self) -> RegionSpikesDict:
        """Get outputs from all registered ports.

        Returns:
            Dict mapping port names to their output spike tensors. Ports that
            haven't been set yet will return zero tensors.
        """
        outputs: RegionSpikesDict = {}
        for port_name in self._registered_ports:
            outputs[port_name] = self.get_port_output(port_name)
        return outputs

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Forward pass with input validation and port output management."""
        for source_name, spikes in inputs.items():
            validate_spike_tensor(spikes, tensor_name=source_name)
            if source_name not in self.input_sources:
                raise ValueError(
                    f"Input source '{source_name}' not registered in {self.__class__.__name__}. "
                    f"Registered sources: {sorted(self.input_sources.keys())}"
                )

        self._port_outputs.clear()
        self._forward_internal(inputs)

        return self.get_all_port_outputs()

    @abstractmethod
    def _forward_internal(self, inputs: RegionSpikesDict) -> None:
        """Internal forward method to be implemented by subclasses.

        This is where the actual processing happens. The base forward() method
        handles input validation and port output management, while this internal
        method focuses on the specific computations for the region.

        Args:
            inputs: Dict mapping source names to spike vectors
        """

    def _integrate_multi_source_synaptic_inputs(
        self,
        inputs: RegionSpikesDict,
        n_neurons: int,
        *,
        weight_key_suffix: str = "",
        apply_stp: bool = False,
        modulation_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Integrate synaptic currents from multiple input sources.

        This is the standard pattern for multi-source integration in biological neurons:
        1. Loop over each presynaptic source in inputs dict
        2. Apply source-specific synaptic weights (independent plasticity)
        3. Optionally apply per-source STP (short-term facilitation/depression)
        4. Optionally apply per-source modulation (attention, gating, etc.)
        5. Accumulate all synaptic currents for somatic integration

        Biological Rationale:
            Real neurons integrate currents from multiple dendritic branches, each receiving
            input from different sources with independent synaptic weights. This method
            models that dendritic integration, where each source creates a conductance
            change that sums at the soma.

        Args:
            inputs: Dict mapping source names to spike tensors [n_source]
            n_neurons: Number of post-synaptic neurons (target population size)
            weight_key_suffix: Suffix for weight keys (e.g., "_d1" for striatum D1 pathway,
                              "_l23" for cortex L2/3 direct input)
            apply_stp: Whether to apply per-source STP if stp_modules exist
            modulation_fn: Optional function to modulate spikes from each source
                          Signature: (spikes: Tensor, source_name: str) -> Tensor
                          Example: lambda s, name: s * alpha_suppression

        Returns:
            Total synaptic current [n_neurons] (float tensor, ready for neuronal integration)

        Raises:
            AssertionError: If input tensors violate ADR-005 (not 1D)
        """
        current = torch.zeros(n_neurons, device=self.device)

        for source_name, source_spikes in inputs.items():
            # ================================================================
            # INPUT VALIDATION (ADR-005: 1D spike tensors)
            # ================================================================
            validate_spike_tensor(source_spikes, tensor_name=source_name)

            # ================================================================
            # TYPE CONVERSION (ADR-004: bool spikes → float for computation)
            # ================================================================
            source_spikes_float = (
                source_spikes.float()
                if source_spikes.dtype == torch.bool
                else source_spikes
            )

            # ================================================================
            # OPTIONAL PER-SOURCE MODULATION
            # ================================================================
            # Examples: alpha attention gating, neuromodulator scaling, etc.
            if modulation_fn is not None:
                source_spikes_float = modulation_fn(source_spikes_float, source_name)

            # ================================================================
            # GET SOURCE-SPECIFIC SYNAPTIC WEIGHTS
            # ================================================================
            weight_key = f"{source_name}{weight_key_suffix}"
            if weight_key not in self.synaptic_weights:
                # Source not connected to this target (graceful skip)
                # This is normal: not all sources connect to all targets
                continue

            weights = self.synaptic_weights[weight_key]

            # ================================================================
            # OPTIONAL PER-SOURCE STP (SHORT-TERM PLASTICITY)
            # ================================================================
            # Models synaptic facilitation/depression (millisecond timescale)
            # Different sources can have different STP dynamics
            if apply_stp and hasattr(self, 'stp_modules') and weight_key in self.stp_modules:
                # Get STP efficacy modulation for this source
                stp_efficacy = self.stp_modules[weight_key](source_spikes_float)
                # Apply as multiplicative modulation of weights
                effective_weights = weights * stp_efficacy
                source_current = effective_weights @ source_spikes_float
            else:
                # Standard weighted summation (no STP)
                source_current = weights @ source_spikes_float

            # ================================================================
            # ACCUMULATE SYNAPTIC CURRENT
            # ================================================================
            # Biology: Dendritic currents sum at soma (linear superposition)
            current += source_current

        return current

    # =========================================================================
    # LEARNING MANAGEMENT
    # =========================================================================

    def _apply_strategy_learning(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        weights: torch.Tensor,
        modulator: Optional[float] = None,
        target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Apply learning strategy to update weights.

        This is the main interface for region learning. It:
        1. Checks plasticity enabled flag
        2. Gets dopamine-modulated learning rate
        3. Applies strategy compute_update()
        4. Updates weights in-place
        5. Returns learning metrics

        Args:
            pre_activity: Presynaptic activity (input spikes)
            post_activity: Postsynaptic activity (output spikes)
            weights: Weight matrix to update [n_post, n_pre]
            modulator: Optional modulator (dopamine, reward, etc.)
                      If None, uses self.neuromodulator_state.dopamine
            target: Optional target for supervised learning
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dict of learning metrics (ltp, ltd, net_change, etc.)
        """
        # Check if strategy is configured
        if self.learning_strategy is None:
            return {}

        # Get dopamine modulation if not explicitly provided
        if modulator is None:
            modulator = self.neuromodulator_state.dopamine

        # Get effective learning rate (dopamine modulated)
        base_lr = self.learning_strategy.config.learning_rate
        effective_lr = self.get_effective_learning_rate(base_lr)

        # Early exit if learning rate too small
        if effective_lr < 1e-8:
            return {}

        # Temporarily adjust strategy learning rate
        original_lr = self.learning_strategy.config.learning_rate
        self.learning_strategy.config.learning_rate = effective_lr

        # Apply strategy
        new_weights, metrics = self.learning_strategy.compute_update(
            weights=weights,
            pre_spikes=pre_activity,
            post_spikes=post_activity,
            modulator=modulator if modulator is not None else 0.0,
            target=target,
            **kwargs,
        )

        # Update weights in-place
        weights.data.copy_(new_weights)

        # Restore original learning rate
        self.learning_strategy.config.learning_rate = original_lr

        return metrics

    # =========================================================================
    # NEUROMODULATOR MANAGEMENT
    # =========================================================================

    def set_neuromodulators(
        self,
        dopamine: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None,
    ) -> None:
        """Set neuromodulator levels atomically (efficient broadcast).

        This consolidated method is more efficient than calling individual setters
        when updating multiple neuromodulators simultaneously (3x reduction in
        function calls and hasattr checks).

        Args:
            dopamine: DA level, typically in [-1, 1].
                      Positive = reward, consolidate current patterns
                      Negative = punishment, reduce current patterns
                      Zero = baseline learning rate
            norepinephrine: NE level, typically in [0, 1].
                           High = arousal, increase neural gain
                           Low = baseline gain
            acetylcholine: ACh level, typically in [0, 1].
                          High = encoding mode, enhance sensory processing
                          Low = retrieval mode, suppress interference

        Raises:
            ValueError: If any neuromodulator value is NaN or Inf
                       (when validation enabled)
        """
        # Validate inputs (NaN/Inf checks)
        if dopamine is not None:
            validate_finite(dopamine, "dopamine", valid_range=(-2.0, 2.0))
            self.neuromodulator_state.dopamine = dopamine
        if norepinephrine is not None:
            validate_finite(norepinephrine, "norepinephrine", valid_range=(0.0, 2.0))
            self.neuromodulator_state.norepinephrine = norepinephrine
        if acetylcholine is not None:
            validate_finite(acetylcholine, "acetylcholine", valid_range=(0.0, 2.0))
            self.neuromodulator_state.acetylcholine = acetylcholine

    def get_effective_learning_rate(
        self,
        base_lr: Optional[float] = None,
        dopamine_sensitivity: float = 1.0,
    ) -> float:
        """Compute learning rate modulated by dopamine.

        The effective learning rate is:
            base_lr * (1 + dopamine_sensitivity * dopamine)

        This means:
            - dopamine = 0: baseline learning
            - dopamine = 1: (1 + sensitivity)x learning rate (strong consolidation)
            - dopamine = -0.5: (1 - 0.5*sensitivity)x learning rate (reduced learning)
            - dopamine = -1: (1 - sensitivity)x learning rate (suppressed if sensitivity=1)

        Args:
            base_lr: Base learning rate (uses self.base_learning_rate if None)
            dopamine_sensitivity: How much dopamine affects learning (0-1)
                                  1.0 = full modulation, 0.0 = no modulation

        Returns:
            Modulated learning rate
        """
        if base_lr is None:
            base_lr = float(getattr(self, "base_learning_rate", 0.01))

        modulation = 1.0 + dopamine_sensitivity * self.neuromodulator_state.dopamine
        # Clamp to non-negative (can't have negative learning rate)
        modulation = max(0.0, modulation)

        return base_lr * modulation

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for neurons and learning strategies.

        Args:
            dt_ms: New timestep in milliseconds
        """
        self.config.dt_ms = dt_ms

        # Update neuron decay factors
        if hasattr(self, "neurons") and self.neurons is not None:
            self.neurons.update_temporal_parameters(dt_ms)

        # Update single learning strategy if present
        if hasattr(self.learning_strategy, "update_temporal_parameters"):
            self.learning_strategy.update_temporal_parameters(dt_ms)

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Dict[str, float],
        coupled_amplitudes: Dict[str, float],
    ) -> None:
        """Default: store oscillator info but don't require usage."""
        self._oscillator_phases = phases
        self._oscillator_signals = signals or {}
        self._coupled_amplitudes = coupled_amplitudes or {}

    @property
    def _theta_phase(self) -> float:
        """Current theta phase in radians [0, 2π)."""
        return float(self._oscillator_phases.get("theta", 0.0))

    @property
    def _gamma_phase(self) -> float:
        """Current gamma phase in radians [0, 2π)."""
        return float(self._oscillator_phases.get("gamma", 0.0))

    @property
    def _alpha_phase(self) -> float:
        """Current alpha phase in radians [0, 2π)."""
        return float(self._oscillator_phases.get("alpha", 0.0))

    @property
    def _beta_phase(self) -> float:
        """Current beta phase in radians [0, 2π)."""
        return float(self._oscillator_phases.get("beta", 0.0))

    @property
    def _delta_phase(self) -> float:
        """Current delta phase in radians [0, 2π)."""
        return float(self._oscillator_phases.get("delta", 0.0))

    @property
    def _alpha_signal(self) -> float:
        """Current alpha signal strength."""
        return float(self._oscillator_signals.get("alpha", 0.0))

    @property
    def _gamma_amplitude_effective(self) -> float:
        """Effective gamma amplitude (with cross-frequency coupling)."""
        return float(self._coupled_amplitudes.get("gamma", 1.0))

    @property
    def _beta_amplitude_effective(self) -> float:
        """Effective beta amplitude (with cross-frequency coupling)."""
        return float(self._coupled_amplitudes.get("beta", 1.0))

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region.

        This method should return a dictionary of relevant metrics and internal states
        that can be used for monitoring, debugging, and analysis. Subclasses should
        override this method to include region-specific diagnostics.

        Returns:
            Dict of diagnostic metrics (e.g., firing rates, weight norms, neuromodulator levels)
        """
