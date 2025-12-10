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
        self.impl_module = cerebellum  # Register as public attribute for nn.Module
        self._cerebellum = cerebellum  # Keep reference for backwards compatibility

        # Track recent activity for learning
        self._recent_input: Optional[torch.Tensor] = None
        self._recent_output: Optional[torch.Tensor] = None
        self._pending_error: Optional[torch.Tensor] = None

    @property
    def impl(self) -> Any:
        """Return the underlying cerebellum implementation."""
        return self._cerebellum

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from the underlying cerebellum implementation."""
        if hasattr(self.impl, "get_diagnostics"):
            return self.impl.get_diagnostics()
        return {}

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to cerebellar neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        if (
            hasattr(self.impl, "neurons")
            and self.impl.neurons.membrane is not None
        ):
            self.impl.neurons.membrane *= decay_factor

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process motor command spikes through cerebellum."""
        # ADR-005: Keep 1D tensors, no batch dimension
        # input_spikes should be [n_neurons]

        # Store for learning
        self._recent_input = input_spikes.clone()

        # Forward through cerebellum
        output = self.impl.forward(
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

        return self.impl.learn(
            input_spikes=self._recent_input,
            output_spikes=self._recent_output,
            target=target,
        )

    def get_state(self) -> Dict[str, Any]:
        """Return cerebellum state."""
        state = super().get_state()

        if hasattr(self.impl, "climbing_fiber"):
            state["error"] = self.impl.climbing_fiber.error.clone()

        if hasattr(self.impl, "get_diagnostics"):
            state["impl"] = self.impl.get_diagnostics()

        return state
