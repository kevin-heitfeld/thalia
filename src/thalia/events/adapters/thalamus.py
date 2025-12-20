"""Event-driven adapter for ThalamicRelay with multi-port input handling.

The thalamus receives inputs on multiple anatomical pathways that target
different neuron populations:
- Sensory input → Relay neurons (thalamocortical projection)
- L6 cortical feedback → TRN neurons (corticothalamic attention modulation)

These inputs should NOT be concatenated - they go to separate post-synaptic targets!

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any, Union

import torch

from ..system import Event, EventType, SpikePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenThalamus(EventDrivenRegionBase):
    """Event-driven adapter for thalamic relay with multi-port input handling.

    Unlike striatum (which concatenates cortex+hippocampus+PFC), thalamus has
    inputs that target DIFFERENT neuron populations:
    - "external" (sensory) → relay neurons
    - "cortex" (L6 feedback) → TRN neurons

    This adapter buffers both inputs and packages them into a named dict for
    the thalamus, avoiding incorrect concatenation.

    Input Ports:
    ===========
    - sensory (external): Sensory input to relay neurons [n_input]
    - l6_feedback (cortex): L6 corticothalamic feedback to TRN [n_l6]

    Architecture:
    ============
    Sensory → Relay Neurons → Cortex
                ↓ (collateral)
               TRN ← L6 Cortical Feedback
                ↓ (inhibition)
            Relay Neurons (attentional gating)
    """

    def __init__(
        self,
        config: EventRegionConfig,
        thalamus: Any,  # ThalamicRelay instance
    ):
        """Initialize thalamus adapter.

        Args:
            config: Event region configuration
            thalamus: ThalamicRelay instance to wrap
        """
        super().__init__(config)
        self.impl_module = thalamus  # Register as nn.Module submodule

    @property
    def impl(self) -> Any:
        """Return the underlying thalamus implementation."""
        return self.impl_module

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to relay and TRN neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay relay neurons
        if (
            hasattr(self.impl, "relay_neurons")
            and self.impl.relay_neurons.membrane is not None
        ):
            self.impl.relay_neurons.membrane *= decay_factor

        # Decay TRN neurons
        if (
            hasattr(self.impl, "trn_neurons")
            and self.impl.trn_neurons.membrane is not None
        ):
            self.impl.trn_neurons.membrane *= decay_factor

    def _process_spikes(
        self,
        input_spikes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        source: str,
    ) -> Optional[torch.Tensor]:
        """Forward multi-port inputs to thalamus.

        With target_port routing, input_spikes is already a dict with proper keys:
        - {"input": sensory_tensor} for single-source (sensory only)
        - {"input": sensory, "l6_feedback": l6} for multi-source

        Args:
            input_spikes: Input dict with target_port keys or single tensor
            source: Source label (may be "multi:external,cortex")

        Returns:
            Relay output spikes
        """
        # DynamicBrain now uses target_port to create proper dict keys
        # Just pass through to thalamus (it handles dict/tensor)
        try:
            output = self.impl.forward(input_spikes)
            return output
        except Exception as e:
            raise RuntimeError(
                f"EventDrivenThalamus: Error processing spikes from {source}: {e}"
            ) from e

    def reset_state(self) -> None:
        """Reset adapter and thalamus state."""
        super().reset_state()
        if hasattr(self.impl, "reset_state"):
            self.impl.reset_state()

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
