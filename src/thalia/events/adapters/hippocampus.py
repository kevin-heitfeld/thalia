"""
Event-Driven Hippocampus Adapter.

Wraps Hippocampus for event-driven simulation.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import torch

from ..system import SpikePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenHippocampus(EventDrivenRegionBase):
    """Event-driven wrapper for Hippocampus.

    Adapts the hippocampus for event-driven simulation. Handles:
    - Phase determination from theta (ENCODE/DELAY/RETRIEVE)
    - EC direct input pathway (raw sensory for comparison)
    - STP on mossy fibers (facilitating)

    Architecture:
        Cortex (EC L2) → DG → CA3 → CA1 → Output
                         ↑    ↑
                 EC L3 direct path
    """

    def __init__(
        self,
        config: EventRegionConfig,
        hippocampus: Any,  # Hippocampus instance
    ):
        super().__init__(config)
        self.impl_module = hippocampus  # Register as public attribute for nn.Module

        # Track EC direct input (from sensory, bypasses cortex)
        self._ec_direct_input: Optional[torch.Tensor] = None

    @property
    def impl(self) -> Any:
        """Return the underlying hippocampus implementation."""
        return self.impl_module

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from the underlying hippocampus implementation."""
        if hasattr(self.impl, "get_diagnostics"):
            return self.impl.get_diagnostics()
        return {}

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to hippocampal neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay neurons in each subregion
        for layer_name in ["dg_neurons", "ca3_neurons", "ca1_neurons"]:
            neurons = getattr(self.impl, layer_name, None)
            if neurons is not None and hasattr(neurons, "membrane"):
                if neurons.membrane is not None:
                    neurons.membrane *= decay_factor

        # Decay NMDA trace (slower time constant)
        if (
            hasattr(self.impl, "state")
            and self.impl.state is not None
        ):
            if self.impl.state.nmda_trace is not None:
                nmda_decay = math.exp(-dt_ms / 100.0)  # ~100ms NMDA time constant
                self.impl.state.nmda_trace *= nmda_decay

    def _handle_spikes(self, event: Any) -> List[Any]:
        """Override to handle EC direct input specially."""
        if isinstance(event.payload, SpikePayload):
            # Check if this is EC direct input (from sensory or special pathway)
            if (
                event.source == "sensory_direct"
                or event.payload.source_layer == "EC_L3"
            ):
                self._ec_direct_input = event.payload.spikes
                return []  # Don't process yet, wait for main input

            # Process main input
            output_spikes = self._process_spikes(
                event.payload.spikes,
                event.source,
            )

            # Clear EC direct input after use
            self._ec_direct_input = None

            if output_spikes is not None and output_spikes.sum() > 0:
                return self._create_output_events(output_spikes)

        return []

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Process input through hippocampal circuit."""
        # Ensure 1D (no batch dimension in 1D architecture)
        input_spikes = input_spikes.squeeze()
        assert input_spikes.dim() == 1, (
            f"EventDrivenHippocampus._process_spikes: input must be 1D [size], "
            f"got shape {input_spikes.shape}"
        )

        # Forward through hippocampus (already expects 1D after ADR-005 update)
        # Theta modulation computed internally by Hippocampus
        output = self.impl.forward(input_spikes, ec_direct_input=self._ec_direct_input)

        return output

    def new_trial(self) -> None:
        """Signal new trial to hippocampus."""
        if hasattr(self.impl, "new_trial"):
            self.impl.new_trial()
        self._ec_direct_input = None

    def get_state(self) -> Dict[str, Any]:
        """Return hippocampus state."""
        state = super().get_state()
        if hasattr(self.impl, "get_diagnostics"):
            state["impl"] = self.impl.get_diagnostics()
        return state
