"""
Event-Driven Cerebellum Adapter.

Wraps Cerebellum for event-driven simulation.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Any

import torch

from ..event_system import DopaminePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenCerebellum(EventDrivenRegionBase):
    """Event-driven wrapper for Cerebellum.

    Handles motor refinement through error-corrective learning.

    Key features:
    - Receives motor commands from striatum
    - Learns via climbing fiber error signals
    - Refines motor output through supervised learning

    The Cerebellum.forward() method handles:
    - input_spikes: Motor commands from striatum
    - encoding_mod/retrieval_mod: Theta modulation

    Learning requires explicit error signals via learn() method.
    """

    def __init__(
        self,
        config: EventRegionConfig,
        cerebellum: Any,  # Cerebellum instance
    ):
        super().__init__(config)
        self.cerebellum = cerebellum

        # Track recent activity for learning
        self._recent_input: Optional[torch.Tensor] = None
        self._recent_output: Optional[torch.Tensor] = None
        self._pending_error: Optional[torch.Tensor] = None

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to cerebellar neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        if (
            hasattr(self.cerebellum, "neurons")
            and self.cerebellum.neurons.membrane is not None
        ):
            self.cerebellum.neurons.membrane *= decay_factor

    def _on_dopamine(self, payload: DopaminePayload) -> None:
        """Handle dopamine signals.

        In the cerebellum, dopamine modulates climbing fiber sensitivity.
        We use the dopamine signal as a proxy for error feedback.
        
        Note: Cerebellum uses error-corrective learning which requires an
        explicit target signal. Dopamine in the cerebellum primarily modulates
        plasticity rate, not direction. For proper cerebellar learning, use
        learn_with_error() with an explicit target.
        """
        # Set dopamine level for plasticity modulation
        self.cerebellum.state.dopamine = payload.level
        
        # Only apply immediate learning if we have both input/output AND a clear error signal
        if self._recent_input is not None and self._recent_output is not None:
            if abs(payload.level) > 0.1 and hasattr(self.cerebellum, "learn"):
                # Create target based on dopamine direction
                # Positive: reinforce current output
                # Negative: suppress current output
                if payload.is_burst:
                    # Reward - reinforce what we did
                    target = self._recent_output.clone()
                elif payload.is_dip:
                    # Error - suppress what we did
                    target = torch.zeros_like(self._recent_output)
                else:
                    # Neutral - no learning
                    return

                self.cerebellum.learn(
                    input_spikes=self._recent_input,
                    output_spikes=self._recent_output,
                    target=target,
                )

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process motor command spikes through cerebellum."""
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # Store for learning
        self._recent_input = input_spikes.clone()

        # Forward through cerebellum
        output = self.cerebellum.forward(
            input_spikes,
            dt=1.0,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
        )

        # Store output for learning
        self._recent_output = output.clone()

        return output.squeeze()

    def learn_with_error(
        self,
        target: torch.Tensor,
    ) -> Dict[str, Any]:
        """Apply error-corrective learning with explicit target.

        This is the primary learning interface for the cerebellum.
        Call this when the correct output is known (e.g., from sensory feedback).

        Args:
            target: Target output pattern

        Returns:
            Learning metrics
        """
        if self._recent_input is None or self._recent_output is None:
            return {"error": "No recent activity to learn from"}

        return self.cerebellum.learn(
            input_spikes=self._recent_input,
            output_spikes=self._recent_output,
            target=target,
        )

    def get_state(self) -> Dict[str, Any]:
        """Return cerebellum state."""
        state = super().get_state()

        if hasattr(self.cerebellum, "climbing_fiber"):
            state["error"] = self.cerebellum.climbing_fiber.error.clone()

        if hasattr(self.cerebellum, "get_diagnostics"):
            state["cerebellum"] = self.cerebellum.get_diagnostics()

        return state
