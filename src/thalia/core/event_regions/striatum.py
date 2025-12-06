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

from ..event_system import DopaminePayload
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
    ):
        super().__init__(config)
        self.striatum = striatum

        # Input sizes for buffering
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size
        self._pfc_input_size = pfc_input_size

        # Input buffers - accumulate spikes from different sources
        self._cortex_buffer: Optional[torch.Tensor] = None
        self._hippocampus_buffer: Optional[torch.Tensor] = None
        self._pfc_buffer: Optional[torch.Tensor] = None

        # Time window for accumulating inputs (ms)
        self._accumulation_window: float = 5.0
        self._last_cortex_time: float = -1000.0
        self._last_hippocampus_time: float = -1000.0
        self._last_pfc_time: float = -1000.0

        # Track recent activity for learning
        self._recent_input: Optional[torch.Tensor] = None
        self._recent_output: Optional[torch.Tensor] = None
        self._selected_action: Optional[int] = None

    def configure_inputs(
        self,
        cortex_input_size: int,
        hippocampus_input_size: int,
        pfc_input_size: int,
    ) -> None:
        """Configure input sizes for buffering."""
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size
        self._pfc_input_size = pfc_input_size

    def _clear_buffers(self) -> None:
        """Clear input buffers."""
        self._cortex_buffer = None
        self._hippocampus_buffer = None
        self._pfc_buffer = None

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to striatal neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay D1 and D2 pathway neurons
        if (
            hasattr(self.striatum, "d1_neurons")
            and self.striatum.d1_neurons.membrane is not None
        ):
            self.striatum.d1_neurons.membrane *= decay_factor
        if (
            hasattr(self.striatum, "d2_neurons")
            and self.striatum.d2_neurons.membrane is not None
        ):
            self.striatum.d2_neurons.membrane *= decay_factor

    def _on_dopamine(self, payload: DopaminePayload) -> None:
        """Handle dopamine for reinforcement learning.

        Dopamine drives learning via the three-factor rule:
        - Positive DA (burst) → strengthen eligible synapses (D1 LTP, D2 LTD)
        - Negative DA (dip) → weaken eligible synapses (D1 LTD, D2 LTP)

        With continuous plasticity, we set the dopamine level on the region
        and let the next forward pass apply the plasticity. Alternatively,
        deliver_reward() can be called for immediate learning.
        """
        # Set dopamine level on the underlying region for continuous plasticity
        self.striatum.state.dopamine = payload.level
        
        # Also trigger immediate learning via deliver_reward if there's a clear reward signal
        if hasattr(self.striatum, "deliver_reward") and abs(payload.level) > 0.1:
            self.striatum.deliver_reward(reward=payload.level)

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Buffer input spikes and process when we have all inputs.

        Striatum requires concatenated input from cortex L5, hippocampus, and PFC.
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # Buffer input based on source
        if source == "cortex":
            self._cortex_buffer = input_spikes
            self._last_cortex_time = self._current_time
        elif source == "hippocampus":
            self._hippocampus_buffer = input_spikes
            self._last_hippocampus_time = self._current_time
        elif source == "pfc":
            self._pfc_buffer = input_spikes
            self._last_pfc_time = self._current_time
        else:
            # Unknown source - try to process directly if sizes match
            return self._forward_striatum(input_spikes)

        # Check if we should process now
        # If sizes not configured, pass through (legacy mode)
        total_expected = (
            self._cortex_input_size
            + self._hippocampus_input_size
            + self._pfc_input_size
        )
        if total_expected == 0:
            # Sizes not configured - just pass through
            return self._forward_striatum(input_spikes)

        # Build combined input
        combined = self._build_combined_input()
        if combined is not None:
            result = self._forward_striatum(combined)
            self._clear_buffers()
            return result

        # Don't have complete input yet
        return None

    def _build_combined_input(self) -> Optional[torch.Tensor]:
        """Build combined input from buffers if ready.

        Combines in order: [cortex_l5 | hippocampus | pfc]
        Uses zeros for missing components if they've timed out.
        """
        device = None
        batch_size = 1

        # Determine device and batch size from any available buffer
        for buf in [self._cortex_buffer, self._hippocampus_buffer, self._pfc_buffer]:
            if buf is not None:
                device = buf.device
                batch_size = buf.shape[0]
                break

        if device is None:
            return None

        # Check if we have at least cortex (the primary input)
        if self._cortex_buffer is None:
            return None

        # Build components, using zeros for missing ones if timed out
        parts = []

        # Cortex L5
        parts.append(self._cortex_buffer)

        # Hippocampus
        if self._hippocampus_buffer is not None:
            parts.append(self._hippocampus_buffer)
        elif self._hippocampus_input_size > 0:
            # Use zeros if timed out or size is configured
            parts.append(
                torch.zeros(batch_size, self._hippocampus_input_size, device=device)
            )

        # PFC
        if self._pfc_buffer is not None:
            parts.append(self._pfc_buffer)
        elif self._pfc_input_size > 0:
            # Use zeros if timed out or size is configured
            parts.append(torch.zeros(batch_size, self._pfc_input_size, device=device))

        return torch.cat(parts, dim=-1)

    def _forward_striatum(self, combined_input: torch.Tensor) -> torch.Tensor:
        """Forward combined input through striatum."""
        # Store for learning
        self._recent_input = combined_input.clone()

        # Forward through striatum
        output = self.striatum.forward(
            combined_input,
            dt=1.0,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            explore=True,  # Enable exploration in event-driven mode
        )

        # Store output for learning
        self._recent_output = output.clone()

        # Track selected action
        if hasattr(self.striatum, "get_selected_action"):
            self._selected_action = self.striatum.get_selected_action()
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

        if hasattr(self.striatum, "get_diagnostics"):
            state["striatum"] = self.striatum.get_diagnostics()

        return state
