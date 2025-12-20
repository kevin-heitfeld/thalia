"""
Generic Event Adapter for User-Defined Components.

This adapter provides automatic event-driven execution support for:
- User-defined regions/pathways
- Components without custom adapters
- Plugin components from external packages

Works with any component following the NeuralComponent protocol.

Author: Thalia Project
Date: December 15, 2025
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING, Any

import torch

from .base import EventDrivenRegionBase, EventRegionConfig

if TYPE_CHECKING:
    from thalia.regions.base import NeuralComponent
    from thalia.config import GlobalConfig


class GenericEventAdapter(EventDrivenRegionBase):
    """Generic event adapter for any NeuralComponent.

    This adapter enables event-driven execution for user-defined components
    without requiring custom adapter implementation. It provides:

    - Generic membrane decay for components with standard neuron attributes
    - Simple pass-through to component's forward() method
    - Automatic output event creation with delays
    - Compatible with any NeuralComponent

    **Limitations:**
    - Generic decay (may not match region-specific decay dynamics)
    - Simple forward() call (no special input routing)
    - No multi-source input buffering

    **Custom Adapter Recommended When:**
    - Region has multiple input pathways requiring routing (e.g., cortex L4/L2/3)
    - Region has layer-specific processing
    - Region has specialized decay dynamics
    - Region requires multi-source input buffering (e.g., striatum D1/D2)

    Example - Auto-wrapping User Region:
        from thalia.core.neural_region import NeuralRegion

        @register_region("my_region", config_class=MyConfig)
        class MyRegion(NeuralRegion):
            def forward(self, inputs: Dict[str, Tensor]):
                return self.process(inputs)

        # GenericEventAdapter automatically wraps it
        brain = (
            BrainBuilder(config)
            .add_component("custom", "my_region", n_neurons=256)
            .build()
        )

        # Event-driven execution works automatically!
        result = brain.forward(input_data, n_timesteps=100)

    Example - Plugin Component:
        # External package registers component
        @register_region("advanced_vision", config_class=VisionConfig)
        class AdvancedVision(NeuralComponent):
            ...

        # GenericEventAdapter enables event-driven execution
        brain.add_component("vision", "advanced_vision", ...)
    """

    def __init__(
        self,
        region: "NeuralComponent",
        config: Optional[Any] = None,
        global_config: Optional["GlobalConfig"] = None,
    ):
        """Initialize adapter with component.

        Args:
            region: NeuralComponent instance to wrap
            config: Event region configuration (optional, inferred from region)
            global_config: Global configuration (optional)
        """
        # Create event config if not provided
        if config is None:
            config = EventRegionConfig(
                name=getattr(region, "name", "generic_region"),
                output_targets=[],  # Will be filled by DynamicBrain
                membrane_tau_ms=20.0,  # Default membrane time constant
                device=getattr(global_config, "device", "cpu") if global_config else "cpu",
            )

        # Initialize base adapter with config only
        super().__init__(config)

        # Store region and register as nn.Module submodule for parameter tracking
        self.impl_module = region
        self._component = region

    @property
    def impl(self) -> "NeuralComponent":
        """Return the underlying component."""
        return self._component

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply generic membrane decay.

        Attempts to decay membrane potentials and traces for components
        with standard neuron attributes. Safe to call even if component
        doesn't have these attributes (no-op).

        Decay Strategy:
        1. Decay neuron membrane potentials (if present)
        2. Decay eligibility traces (if present)
        3. Decay input/output traces (if present)

        Custom adapters should override this for region-specific decay.

        Args:
            dt_ms: Time elapsed since last update (milliseconds)
        """
        # Skip if no neurons attribute
        if not hasattr(self._component, 'neurons'):
            return

        neurons = self._component.neurons
        if neurons is None:
            return

        # Calculate decay factor
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay membrane potentials
        if hasattr(neurons, 'membrane') and neurons.membrane is not None:
            neurons.membrane *= decay_factor

        # Decay eligibility traces (for three-factor learning)
        if hasattr(self._component, 'eligibility'):
            trace = getattr(self._component, 'eligibility')
            if trace is not None and isinstance(trace, torch.Tensor):
                trace *= decay_factor

        # Decay input/output traces (for STDP)
        for trace_attr in ['input_trace', 'output_trace', 'pre_trace', 'post_trace']:
            if hasattr(self._component, trace_attr):
                trace = getattr(self._component, trace_attr)
                if trace is not None and isinstance(trace, torch.Tensor):
                    trace *= decay_factor

    def _process_spikes(
        self,
        input_spikes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        source: str,
    ) -> Optional[torch.Tensor]:
        """Forward spikes through component.

        Simple pass-through to component's forward() method. For complex
        input routing (e.g., layer-specific, multi-source buffering),
        create a custom adapter.

        Args:
            input_spikes: Input spike tensor [n_neurons] or dict of tensors
            source: Source component name (unused in generic adapter)

        Returns:
            Output spikes from component, or None on error
        """
        try:
            # Call component's forward method
            # Note: source is ignored - custom adapters can use it for routing
            # Pass through dict or tensor as-is
            output = self._component.forward(input_spikes)

            # Validate output
            if output is not None and not isinstance(output, torch.Tensor):
                print(f"Warning: {self._name} forward() returned non-tensor: {type(output)}")
                return None

            return output

        except Exception as e:
            # Log error but don't crash - return no output
            # This allows other components to continue processing
            print(f"Warning: {self._name} forward() failed: {e}")
            import traceback
            traceback.print_exc()
            return None


__all__ = [
    "GenericEventAdapter",
]
