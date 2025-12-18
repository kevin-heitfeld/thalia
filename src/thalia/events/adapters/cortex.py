"""
Event-Driven Cortex Adapter.

Wraps LayeredCortex for event-driven simulation.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import torch

from ..system import Event, EventType, SpikePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenCortex(EventDrivenRegionBase):
    """Event-driven wrapper for LayeredCortex.

    Adapts the existing LayeredCortex to work with the event-driven
    simulation framework. Handles:
    - Layer-specific input routing
    - Membrane decay between events
    - Dual output: L2/3 → cortical targets, L5 → subcortical targets
    - Top-down modulation from PFC (via attention pathway)

    Architecture:
        Sensory Input → L4 → L2/3 → L5
                              ↓      ↓
                          (hippocampus, pfc)  (striatum)

        PFC → attention pathway → L2/3 (top-down modulation)
    """

    def __init__(
        self,
        config: EventRegionConfig,
        cortex: Any,  # LayeredCortex instance
    ):
        super().__init__(config)
        self.impl_module = cortex  # Register as public attribute for nn.Module

    @property
    def impl(self) -> Any:
        """Return the underlying cortex implementation."""
        return self.impl_module

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to cortex neurons.

        Directly decay the membrane potentials of the LIF neurons
        in each cortical layer.
        """
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay each layer's neurons
        for layer_name in ["l4_neurons", "l23_neurons", "l5_neurons"]:
            neurons = getattr(self.impl, layer_name, None)
            if neurons is not None and hasattr(neurons, "membrane"):
                if neurons.membrane is not None:
                    neurons.membrane *= decay_factor

        # Also decay the recurrent activity trace (if the cortex type has it)
        if hasattr(self.impl, "state") and self.impl.state is not None:
            if hasattr(self.impl.state, "l23_recurrent_activity"):
                if self.impl.state.l23_recurrent_activity is not None:
                    self.impl.state.l23_recurrent_activity *= decay_factor

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process input through cortex layers."""
        # ADR-005: Keep 1D tensors, no batch dimension
        # input_spikes should be [n_neurons]

        # Handle top-down input from PFC
        # ADR-013: Pathways handle all dimensional transformations
        # PFC→Cortex attention pathway already projects to L2/3 size
        top_down = None
        if source == "pfc":
            # Verify pathway transformed to correct size
            assert input_spikes.shape[0] == getattr(self.impl, 'l23_size', 0), (
                f"PFC input must match L2/3 size ({getattr(self.impl, 'l23_size', 0)}), "
                f"got {input_spikes.shape[0]}. Check attention pathway configuration."
            )
            # Convert to modulatory signal (between 0 and 1)
            top_down = torch.sigmoid(input_spikes.float())
            return None  # Top-down alone doesn't drive output

        # Forward through cortex (theta modulation computed internally)
        output = self.impl.forward(
            input_spikes,
            top_down=top_down,
        )

        # Output should be 1D
        return output.squeeze() if output.dim() > 1 else output

    def _create_output_events(self, spikes: torch.Tensor) -> List[Event]:
        """Create layer-specific output events.

        L5 output → subcortical targets (striatum)
        L2/3 output → cortical targets (hippocampus, pfc)
        """
        events = []

        # Get layer-specific spikes if available
        l23_spikes = None
        l5_spikes = None

        if hasattr(self.impl, "state") and self.impl.state is not None:
            l23_spikes = self.impl.state.l23_spikes
            l5_spikes = self.impl.state.l5_spikes

        # If we don't have separate layer outputs, use the combined output
        if l23_spikes is None or l5_spikes is None:
            # Fall back to base implementation
            return super()._create_output_events(spikes)

        # Create events for each connection with appropriate layer routing
        for conn in self._connections:
            target = conn.target

            # Choose appropriate layer output for target
            if target in ["striatum", "motor"]:
                # Subcortical targets get L5 output
                output_spikes = l5_spikes
                source_layer = "L5"
            else:
                # Cortical targets (hippocampus, pfc) get L2/3 output
                output_spikes = l23_spikes
                source_layer = "L23"

            if output_spikes is not None and output_spikes.sum() > 0:
                event = Event(
                    time=self._current_time + conn.delay_ms,
                    event_type=EventType.SPIKE,
                    source=self._name,
                    target=target,
                    payload=SpikePayload(
                        spikes=output_spikes.squeeze().clone(),
                        source_layer=source_layer,
                    ),
                )
                events.append(event)

        return events

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from the underlying cortex implementation."""
        if hasattr(self.impl, "get_diagnostics"):
            return self.impl.get_diagnostics()
        return {}

    def get_state(self) -> Dict[str, Any]:
        """Return cortex state."""
        state = super().get_state()
        # Add cortex-specific diagnostics
        if hasattr(self.impl, "get_diagnostics"):
            state["impl"] = self.impl.get_diagnostics()
        return state
