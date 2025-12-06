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

from ..event_system import Event, EventType, SpikePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenCortex(EventDrivenRegionBase):
    """Event-driven wrapper for LayeredCortex.

    Adapts the existing LayeredCortex to work with the event-driven
    simulation framework. Handles:
    - Layer-specific input routing
    - Membrane decay between events
    - Dual output: L2/3 → cortical targets, L5 → subcortical targets
    - Top-down projection from PFC

    Architecture:
        Sensory Input → L4 → L2/3 → L5
                              ↓      ↓
                          (hippocampus, pfc)  (striatum)

        PFC → top-down projection → L2/3
    """

    def __init__(
        self,
        config: EventRegionConfig,
        cortex: Any,  # LayeredCortex instance
        pfc_size: int = 0,  # Size of PFC output for top-down projection
    ):
        super().__init__(config)
        self._cortex = cortex
        self._pfc_size = pfc_size

        # Track pending top-down modulation
        self._pending_top_down: Optional[torch.Tensor] = None

        # Accumulated input (for handling multiple sources)
        self._accumulated_input: Optional[torch.Tensor] = None

        # Top-down projection from PFC to L2/3
        # Only create if PFC size is provided
        self._top_down_projection: Optional[torch.nn.Linear] = None
        if pfc_size > 0 and hasattr(cortex, "l23_size"):
            self._top_down_projection = torch.nn.Linear(
                pfc_size, cortex.l23_size, bias=False
            )
            # Initialize with small weights (modulatory, not driving)
            torch.nn.init.normal_(
                self._top_down_projection.weight,
                mean=0.0,
                std=0.1 / pfc_size**0.5,
            )

    @property
    def impl(self) -> Any:
        """Return the underlying cortex implementation."""
        return self._cortex

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
        # Ensure batch dimension
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # Handle top-down input from PFC
        if source == "pfc":
            # Project PFC spikes to L2/3 size if projection exists
            if self._top_down_projection is not None:
                projected = self._top_down_projection(input_spikes.float())
                # Convert to modulatory signal (between 0 and 1)
                self._pending_top_down = torch.sigmoid(projected)
            else:
                # No projection - skip top-down (sizes don't match)
                self._pending_top_down = None
            return None  # Top-down alone doesn't drive output

        # Forward through cortex with current theta modulation
        output = self.impl.forward(
            input_spikes,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            top_down=self._pending_top_down,
        )

        # Clear pending top-down after use
        self._pending_top_down = None

        # Output is typically L5 activity (for subcortical targets)
        # But we might want L2/3 for cortical targets
        return output.squeeze()

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
