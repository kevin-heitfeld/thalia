"""
Event-Driven Striatum Adapter.

Wraps Striatum for event-driven simulation.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Any

import torch

from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenStriatum(EventDrivenRegionBase):
    """Event-driven wrapper for Striatum.

    Handles action selection and reinforcement learning.

    Key features:
    - D1/D2 pathway balance for action selection
    - Dopamine-modulated learning (three-factor rule)
    - RPE-based updates via dopamine system
    - Input buffering: Accumulates inputs from cortex, hippocampus, and PFC

    The Striatum.forward() method handles:
    - input_spikes: Concatenated input from cortex (L5) + hippocampus + PFC
    - encoding_mod/retrieval_mod: Theta modulation
    - explore: Whether to use exploration

    Input Buffering:
    ================
    Striatum receives inputs from cortex L5, hippocampus, and PFC on separate pathways.
    These are buffered and concatenated before processing.
    Order: [cortex_l5 | hippocampus | pfc]
    """

    def __init__(
        self,
        config: EventRegionConfig,
        striatum: Any,  # Striatum instance
        cortex_input_size: int = 0,
        hippocampus_input_size: int = 0,
        pfc_input_size: int = 0,
        pfc_region: Optional[Any] = None,  # PFC region for goal context
    ):
        super().__init__(config)
        self.impl_module = striatum  # Register as public attribute for nn.Module
        self._striatum = striatum  # Keep reference for backwards compatibility
        self._pfc_region = pfc_region  # Store PFC reference for goal-conditioned values

        # Configure input buffering using base class
        if cortex_input_size > 0:
            input_config = {"cortex": cortex_input_size}
            if hippocampus_input_size > 0:
                input_config["hippocampus"] = hippocampus_input_size
            if pfc_input_size > 0:
                input_config["pfc"] = pfc_input_size
            self.configure_input_sources(**input_config)

        # Track recent activity for learning
        self._recent_input: Optional[torch.Tensor] = None
        self._recent_output: Optional[torch.Tensor] = None
        self._selected_action: Optional[int] = None

    @property
    def impl(self) -> Any:
        """Return the underlying striatum implementation."""
        return self._striatum

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from the underlying striatum implementation."""
        if hasattr(self.impl, "get_diagnostics"):
            return self.impl.get_diagnostics()
        return {}

    def configure_inputs(
        self,
        cortex_input_size: int,
        hippocampus_input_size: int,
        pfc_input_size: int,
    ) -> None:
        """Configure input sizes for buffering."""
        input_config = {"cortex": cortex_input_size}
        if hippocampus_input_size > 0:
            input_config["hippocampus"] = hippocampus_input_size
        if pfc_input_size > 0:
            input_config["pfc"] = pfc_input_size
        self.configure_input_sources(**input_config)

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to striatal neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay D1 and D2 pathway neurons
        if (
            hasattr(self.impl, "d1_neurons")
            and self.impl.d1_neurons.membrane is not None
        ):
            self.impl.d1_neurons.membrane *= decay_factor
        if (
            hasattr(self.impl, "d2_neurons")
            and self.impl.d2_neurons.membrane is not None
        ):
            self.impl.d2_neurons.membrane *= decay_factor

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Buffer input spikes and process when we have all inputs.

        Striatum requires concatenated input from cortex L5, hippocampus, and PFC.
        """
        # ADR-005: Keep 1D tensors, no batch dimension
        # input_spikes should be [n_neurons]

        # Buffer input using base class method
        if source in ["cortex", "hippocampus", "pfc"]:
            self._buffer_input(source, input_spikes)
        else:
            # Unknown source - try to process directly if sizes match
            return self._forward_striatum(input_spikes)

        # Check if we should process now
        # If sizes not configured, pass through (legacy mode)
        if not self._input_sizes:
            return self._forward_striatum(input_spikes)

        # Determine source order based on what's configured
        source_order = ["cortex"]  # Cortex is always first
        if "hippocampus" in self._input_sizes:
            source_order.append("hippocampus")
        if "pfc" in self._input_sizes:
            source_order.append("pfc")

        # Build combined input using base class method
        combined = self._build_combined_input(
            source_order=source_order,
            require_sources=["cortex"],  # Cortex is required, others optional
        )

        if combined is not None:
            result = self._forward_striatum(combined)
            self._clear_input_buffers()
            return result

        # Don't have complete input yet
        return None

    def _forward_striatum(self, combined_input: torch.Tensor) -> torch.Tensor:
        """Forward combined input through striatum."""
        # Store for learning
        self._recent_input = combined_input.clone()

        # Get goal context from PFC if available
        pfc_goal_context = None
        if self._pfc_region is not None and hasattr(self._pfc_region, "get_goal_context"):
            pfc_goal_context = self._pfc_region.get_goal_context()

        # Forward through striatum (theta modulation computed internally)
        # Enable exploration in event-driven mode
        output = self.impl.forward(combined_input, pfc_goal_context=pfc_goal_context, explore=True)

        # Store output for learning
        self._recent_output = output.clone()

        # Track selected action
        if hasattr(self.impl, "get_selected_action"):
            self._selected_action = self.impl.get_selected_action()
        elif output is not None:
            self._selected_action = int(output.argmax().item())

        return output.squeeze()

    def get_selected_action(self) -> Optional[int]:
        """Get the currently selected action."""
        return self._selected_action

    def get_state(self) -> Dict[str, Any]:
        """Return striatum state."""
        state = super().get_state()
        state["selected_action"] = self._selected_action

        if hasattr(self.impl, "get_diagnostics"):
            state["impl"] = self.impl.get_diagnostics()

        return state
