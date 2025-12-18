"""
Event-Driven Thalamus Adapter.

Wraps ThalamicRelay for event-driven simulation.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import torch

from ..system import Event, EventType, SpikePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenThalamus(EventDrivenRegionBase):
    """Event-driven wrapper for ThalamicRelay.

    Adapts the ThalamicRelay to work with the event-driven
    simulation framework. Handles:
    - Sensory input relay to cortex
    - Alpha oscillation-based attentional gating
    - Burst vs tonic mode switching
    - TRN-mediated inhibitory coordination
    - Membrane decay between events

    Architecture:
        Sensory Input → Thalamus → Cortex L4
                         ↓
                       (TRN inhibition)

    The thalamus acts as a sensory gateway, modulating
    input based on attention (alpha) and arousal (NE).
    """

    def __init__(
        self,
        config: EventRegionConfig,
        thalamus: Any,  # ThalamicRelay instance
    ):
        super().__init__(config)
        self.impl_module = thalamus  # Register as public attribute for nn.Module
        self._thalamus = thalamus  # Keep reference for backwards compatibility

        # No accumulated input needed - thalamus has single input source (sensory)

    @property
    def impl(self) -> Any:
        """Return the underlying thalamus implementation."""
        return self._thalamus

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to thalamus neurons.

        Directly decay the membrane potentials of the relay and TRN neurons.
        """
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay relay neurons
        if hasattr(self.impl, "relay_neurons") and self.impl.relay_neurons is not None:
            if hasattr(self.impl.relay_neurons, "membrane"):
                if self.impl.relay_neurons.membrane is not None:
                    self.impl.relay_neurons.membrane *= decay_factor

        # Decay TRN neurons
        if hasattr(self.impl, "trn_neurons") and self.impl.trn_neurons is not None:
            if hasattr(self.impl.trn_neurons, "membrane"):
                if self.impl.trn_neurons.membrane is not None:
                    self.impl.trn_neurons.membrane *= decay_factor

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process input through thalamus relay.

        Args:
            input_spikes: Sensory input spikes [n_neurons] (1D, ADR-005)
            source: Input source name (typically "sensory_input")

        Returns:
            Relay output spikes [n_relay] (bool, ADR-004/005)
        """
        # ADR-005: Keep 1D tensors, no batch dimension
        # ADR-004: Spikes are bool
        assert input_spikes.dim() == 1, (
            f"Thalamus adapter expects 1D input (ADR-005), got {input_spikes.shape}"
        )

        # Forward through thalamus (oscillator phases set by brain)
        output = self.impl.forward(input_spikes)

        # Output is already 1D bool (ADR-004/005)
        return output

    def _create_output_events(self, spikes: torch.Tensor) -> List[Event]:
        """Create relay output events to cortex.

        Args:
            spikes: Relay output spikes [n_relay] (1D bool, ADR-004/005)

        Returns:
            List of spike events with appropriate delays
        """
        events = []

        # Check if we have any spikes
        if spikes.any():
            # Create spike event for each target
            for conn in self._connections:
                events.append(
                    Event(
                        time=self._current_time + conn.delay_ms,
                        source=self.name,
                        target=conn.target,
                        event_type=EventType.SPIKE,
                        payload=SpikePayload(spikes=spikes),
                    )
                )

        return events

    def handle_event(self, event: Event) -> List[Event]:
        """Process incoming event and generate output events.

        Args:
            event: Incoming event (typically SPIKE from sensory input)

        Returns:
            List of output events to cortex
        """
        # Update time tracking
        dt = event.time - self._last_update_time
        if dt > 0:
            self._apply_decay(dt)

        self._last_update_time = event.time
        self._current_time = event.time

        # Process based on event type
        if event.event_type == EventType.SPIKE:
            # Extract spikes from payload
            payload = event.payload
            if not isinstance(payload, SpikePayload):
                return []

            input_spikes = payload.spikes

            # Process through thalamus
            output_spikes = self._process_spikes(input_spikes, event.source)

            if output_spikes is not None:
                return self._create_output_events(output_spikes)

        elif event.event_type == EventType.THETA_PHASE:
            # Theta phase updates (not directly used by thalamus)
            pass

        elif event.event_type == EventType.NEUROMODULATOR:
            # Handle neuromodulator updates
            payload = event.payload
            if hasattr(payload, "neuromodulator_levels"):
                levels = payload.neuromodulator_levels

                # Set norepinephrine (arousal/gain modulation)
                if "norepinephrine" in levels:
                    self.impl.set_neuromodulators(norepinephrine=levels["norepinephrine"])

        return []

    def reset(self) -> None:
        """Reset thalamus state."""
        self.impl.reset_state()
        self._last_update_time = 0.0
        self._current_time = 0.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get thalamus diagnostic information."""
        diagnostics = {
            "name": self.name,
            "current_time": self._current_time,
            "last_update_time": self._last_update_time,
        }

        # Get underlying implementation diagnostics
        if hasattr(self.impl, "get_diagnostics"):
            impl_diags = self.impl.get_diagnostics()
            diagnostics.update(impl_diags)

        return diagnostics
