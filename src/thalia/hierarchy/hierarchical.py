"""
Hierarchical Thinking SNN with Multiple Time Constants.

This module implements a hierarchical spiking neural network where different
levels operate at different temporal scales:

- Sensory layer: Fast (τ=5ms) - raw input processing
- Feature layer: Medium-fast (τ=10ms) - feature extraction
- Concept layer: Medium-slow (τ=50ms) - concept formation
- Abstract layer: Slow (τ=200ms) - abstract reasoning

The architecture supports:
- Bottom-up pathways (feedforward)
- Top-down predictions (feedback)
- Lateral connections within layers

This creates a "temporal hierarchy" where abstract thoughts evolve slowly
and guide faster perceptual processing through top-down modulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn

from thalia.core.neuron import LIFNeuron, LIFConfig


@dataclass
class LayerConfig:
    """Configuration for a single hierarchical layer.

    Attributes:
        name: Human-readable layer name
        n_neurons: Number of neurons in this layer
        tau_mem: Membrane time constant in ms (defines temporal scale)
        threshold: Firing threshold
        noise_std: Noise level for spontaneous activity
        recurrent: Whether layer has recurrent connections
        recurrent_strength: Strength of recurrent connections
    """
    name: str = "layer"
    n_neurons: int = 256
    tau_mem: float = 20.0
    threshold: float = 1.0
    noise_std: float = 0.0
    recurrent: bool = True
    recurrent_strength: float = 0.5


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical SNN.

    Provides default layer configurations for a 4-level hierarchy with
    increasing time constants at higher levels.

    Attributes:
        layers: List of layer configurations
        dt: Simulation timestep in ms
        feedforward_strength: Strength of bottom-up connections
        feedback_strength: Strength of top-down connections
        lateral_strength: Strength of within-layer connections
        enable_feedback: Whether to enable top-down connections
        enable_lateral: Whether to enable lateral connections
    """
    layers: List[LayerConfig] = field(default_factory=lambda: [
        LayerConfig(name="sensory", n_neurons=256, tau_mem=5.0, noise_std=0.01),
        LayerConfig(name="feature", n_neurons=128, tau_mem=10.0, noise_std=0.02),
        LayerConfig(name="concept", n_neurons=64, tau_mem=50.0, noise_std=0.05),
        LayerConfig(name="abstract", n_neurons=32, tau_mem=200.0, noise_std=0.1),
    ])
    dt: float = 1.0
    feedforward_strength: float = 1.0
    feedback_strength: float = 0.3
    lateral_strength: float = 0.2
    enable_feedback: bool = True
    enable_lateral: bool = True

    @property
    def n_layers(self) -> int:
        """Number of layers in hierarchy."""
        return len(self.layers)

    @property
    def total_neurons(self) -> int:
        """Total neurons across all layers."""
        return sum(layer.n_neurons for layer in self.layers)


class HierarchicalLayer(nn.Module):
    """A single layer in the hierarchical SNN.

    Each layer has:
    - LIF neurons with configurable time constant
    - Optional recurrent connections
    - Feedforward input weights (from lower layer)
    - Feedback input weights (from higher layer)
    - Lateral connections (within layer)

    The time constant determines the "temporal scale" of the layer -
    layers with larger τ integrate information over longer periods
    and change state more slowly.

    Args:
        config: Layer configuration
        input_size: Size of feedforward input (from layer below)
        feedback_size: Size of feedback input (from layer above)
        dt: Simulation timestep

    Example:
        >>> config = LayerConfig(name="concept", n_neurons=64, tau_mem=50.0)
        >>> layer = HierarchicalLayer(config, input_size=128, feedback_size=32)
        >>> layer.reset_state(batch_size=1)
        >>> spikes, membrane = layer(bottom_up_input, top_down_feedback)
    """

    def __init__(
        self,
        config: LayerConfig,
        input_size: Optional[int] = None,
        feedback_size: Optional[int] = None,
        dt: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.n_neurons = config.n_neurons
        self.dt = dt

        # Create LIF neurons with layer-specific time constant
        lif_config = LIFConfig(
            tau_mem=config.tau_mem,
            v_threshold=config.threshold,
            dt=dt,
        )
        self.neurons = LIFNeuron(config.n_neurons, lif_config)

        # Feedforward weights (bottom-up)
        if input_size is not None and input_size > 0:
            self.ff_weights = nn.Parameter(
                torch.randn(input_size, config.n_neurons) * 0.1
            )
        else:
            self.register_parameter("ff_weights", None)

        # Feedback weights (top-down)
        if feedback_size is not None and feedback_size > 0:
            self.fb_weights = nn.Parameter(
                torch.randn(feedback_size, config.n_neurons) * 0.1
            )
        else:
            self.register_parameter("fb_weights", None)

        # Recurrent weights (within layer)
        if config.recurrent:
            recurrent = torch.randn(config.n_neurons, config.n_neurons) * 0.1
            # Zero self-connections
            recurrent.fill_diagonal_(0)
            self.recurrent_weights = nn.Parameter(recurrent * config.recurrent_strength)
        else:
            self.register_parameter("recurrent_weights", None)

        # State
        self._last_spikes: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset layer state."""
        self.neurons.reset_state(batch_size)
        # Get device from any available parameter
        device: torch.device | str = "cpu"
        for param in self.parameters():
            device = param.device
            break
        self._last_spikes = torch.zeros(
            batch_size, self.n_neurons, device=device
        )

    def forward(
        self,
        ff_input: Optional[torch.Tensor] = None,
        fb_input: Optional[torch.Tensor] = None,
        lateral_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.

        Args:
            ff_input: Feedforward (bottom-up) input from lower layer
            fb_input: Feedback (top-down) input from higher layer
            lateral_input: Lateral input from other neurons in same layer

        Returns:
            Tuple of (spikes, membrane_potential)
        """
        # Compute total input current
        total_input = torch.zeros(
            self._last_spikes.shape[0], self.n_neurons,
            device=self._last_spikes.device
        )

        # Bottom-up input
        if ff_input is not None and self.ff_weights is not None:
            total_input = total_input + ff_input @ self.ff_weights

        # Top-down input
        if fb_input is not None and self.fb_weights is not None:
            total_input = total_input + fb_input @ self.fb_weights

        # Recurrent input
        if self.recurrent_weights is not None:
            recurrent = self._last_spikes @ self.recurrent_weights
            total_input = total_input + recurrent

        # Lateral input (external, e.g., from attention)
        if lateral_input is not None:
            total_input = total_input + lateral_input

        # Add noise for spontaneous activity
        if self.config.noise_std > 0:
            noise = torch.randn_like(total_input) * self.config.noise_std
            total_input = total_input + noise

        # Step neurons
        spikes, membrane = self.neurons(total_input)
        self._last_spikes = spikes

        return spikes, membrane

    @property
    def membrane(self) -> Optional[torch.Tensor]:
        """Current membrane potential."""
        return self.neurons.membrane

    @property
    def last_spikes(self) -> Optional[torch.Tensor]:
        """Last spike output."""
        return self._last_spikes


class HierarchicalSNN(nn.Module):
    """Multi-level hierarchical spiking neural network.

    Implements a hierarchy of layers with different time constants, where:
    - Lower layers (fast τ) process rapid sensory changes
    - Higher layers (slow τ) maintain abstract representations
    - Bottom-up connections carry sensory information up
    - Top-down connections carry predictions/context down

    The key insight is that thoughts at different levels evolve at
    different speeds - perceptions are fleeting while abstract concepts
    persist. This temporal hierarchy enables both rapid reaction and
    sustained reasoning.

    Args:
        config: Hierarchical configuration

    Example:
        >>> config = HierarchicalConfig()
        >>> net = HierarchicalSNN(config)
        >>> net.reset_state(batch_size=1)
        >>>
        >>> # Process sensory input
        >>> input = torch.randn(1, 256)
        >>> outputs = net(input)
        >>> print(f"Layer outputs: {[o.shape for o in outputs['spikes']]}")
    """

    def __init__(self, config: Optional[HierarchicalConfig] = None):
        super().__init__()
        self.config = config or HierarchicalConfig()

        # Create layers
        self.layers = nn.ModuleList()

        for i, layer_config in enumerate(self.config.layers):
            # Determine input sizes
            if i == 0:
                input_size = None  # Sensory layer gets external input
            else:
                input_size = self.config.layers[i - 1].n_neurons

            if i < self.config.n_layers - 1:
                feedback_size = self.config.layers[i + 1].n_neurons
            else:
                feedback_size = None  # Top layer has no feedback

            layer = HierarchicalLayer(
                layer_config,
                input_size=input_size,
                feedback_size=feedback_size,
                dt=self.config.dt,
            )
            self.layers.append(layer)

        # Input projection for sensory layer
        self._input_size: Optional[int] = None
        self.input_projection: Optional[nn.Parameter] = None

        # State tracking
        self._timestep: int = 0
        self._layer_activities: List[List[torch.Tensor]] = [[] for _ in self.layers]

    def set_input_size(self, input_size: int) -> None:
        """Set input size and create input projection.

        Call this before processing if input size differs from sensory layer.
        """
        if input_size == self.config.layers[0].n_neurons:
            self.input_projection = None
        else:
            self.input_projection = nn.Parameter(
                torch.randn(input_size, self.config.layers[0].n_neurons) * 0.1
            )
        self._input_size = input_size

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all layers and tracking."""
        for layer in self.layers:
            layer.reset_state(batch_size)

        self._timestep = 0
        self._layer_activities = [[] for _ in self.layers]

    def forward(
        self,
        external_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Process one timestep through the hierarchy.

        Information flows:
        1. Bottom-up: external_input → sensory → feature → concept → abstract
        2. Top-down (if enabled): abstract → concept → feature → sensory

        Args:
            external_input: Input to sensory layer, shape (batch, input_size)

        Returns:
            Dict with:
            - spikes: List of spike tensors for each layer
            - membrane: List of membrane potentials for each layer
            - layer_names: Names of layers
        """
        self._timestep += 1

        # Project input if needed
        if external_input is not None and self.input_projection is not None:
            sensory_input = external_input @ self.input_projection
        else:
            sensory_input = external_input

        # First pass: bottom-up
        layer_spikes = []
        layer_membranes = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                # Sensory layer receives external input
                ff_input = sensory_input
            else:
                # Higher layers receive output from layer below
                ff_input = layer_spikes[i - 1] * self.config.feedforward_strength

            spikes, membrane = layer(ff_input=ff_input)
            layer_spikes.append(spikes)
            layer_membranes.append(membrane)

        # Second pass: top-down (if enabled)
        if self.config.enable_feedback and len(self.layers) > 1:
            # Go from top to bottom, modulating lower layers
            for i in range(len(self.layers) - 2, -1, -1):
                fb_input = layer_spikes[i + 1] * self.config.feedback_strength

                # Add feedback as additional input (modulation)
                # Re-run the forward pass with feedback
                if i == 0:
                    ff_input = sensory_input
                else:
                    ff_input = layer_spikes[i - 1] * self.config.feedforward_strength

                spikes, membrane = self.layers[i](
                    ff_input=ff_input,
                    fb_input=fb_input,
                )
                layer_spikes[i] = spikes
                layer_membranes[i] = membrane

        # Record activities
        for i, spikes in enumerate(layer_spikes):
            self._layer_activities[i].append(spikes.detach().clone())

        return {
            "spikes": layer_spikes,
            "membrane": layer_membranes,
            "layer_names": [cfg.name for cfg in self.config.layers],
            "timestep": self._timestep,
        }

    def get_layer(self, name_or_index: str | int) -> HierarchicalLayer:
        """Get a layer by name or index."""
        if isinstance(name_or_index, int):
            return self.layers[name_or_index]

        for i, cfg in enumerate(self.config.layers):
            if cfg.name == name_or_index:
                return self.layers[i]

        raise ValueError(f"Layer '{name_or_index}' not found")

    def get_layer_activity(
        self,
        layer: str | int,
        smoothing: Optional[int] = None,
    ) -> torch.Tensor:
        """Get activity history for a specific layer.

        Args:
            layer: Layer name or index
            smoothing: Optional window size for smoothing

        Returns:
            Activity tensor of shape (timesteps, batch, n_neurons)
        """
        if isinstance(layer, str):
            for i, cfg in enumerate(self.config.layers):
                if cfg.name == layer:
                    layer = i
                    break

        if not self._layer_activities[layer]:
            return torch.tensor([])

        activity = torch.stack(self._layer_activities[layer])

        if smoothing is not None and smoothing > 1 and activity.shape[0] >= smoothing:
            # Simple moving average
            kernel = torch.ones(smoothing) / smoothing
            kernel = kernel.view(1, 1, -1).to(activity.device)

            # Reshape for conv1d
            t, b, n = activity.shape
            activity_reshaped = activity.permute(1, 2, 0).reshape(b * n, 1, t)

            # Apply smoothing
            padded = nn.functional.pad(
                activity_reshaped, (smoothing - 1, 0), mode='replicate'
            )
            smoothed = nn.functional.conv1d(padded, kernel)

            activity = smoothed.reshape(b, n, -1).permute(2, 0, 1)

        return activity

    def get_temporal_profile(self) -> Dict[str, Dict[str, float]]:
        """Get temporal statistics for each layer.

        Returns dict with layer names mapping to:
        - mean_rate: Mean firing rate
        - variance: Activity variance over time
        - tau: Effective time constant
        """
        profile = {}

        for i, cfg in enumerate(self.config.layers):
            activity = self.get_layer_activity(i)

            if activity.numel() == 0:
                profile[cfg.name] = {
                    "mean_rate": 0.0,
                    "variance": 0.0,
                    "tau": cfg.tau_mem,
                }
            else:
                mean_rate = activity.mean().item()
                variance = activity.var().item()

                profile[cfg.name] = {
                    "mean_rate": mean_rate,
                    "variance": variance,
                    "tau": cfg.tau_mem,
                }

        return profile

    def inject_to_layer(
        self,
        layer: str | int,
        pattern: torch.Tensor,
        strength: float = 1.0,
    ) -> None:
        """Inject a pattern directly into a layer.

        Useful for seeding thoughts at a particular level of abstraction.

        Args:
            layer: Target layer name or index
            pattern: Activation pattern to inject
            strength: Injection strength multiplier
        """
        target = self.get_layer(layer)

        if target._last_spikes is not None:
            target._last_spikes = target._last_spikes + pattern * strength
