"""
Event-Driven PFC (Prefrontal Cortex) Adapter.

Wraps Prefrontal for event-driven simulation.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Any

import torch

from ..event_system import DopaminePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenPFC(EventDrivenRegionBase):
    """Event-driven wrapper for Prefrontal Cortex.

    Handles working memory maintenance and dopamine-gated learning.

    Key features:
    - Working memory: Maintained via recurrent activity in the PFC
    - Dopamine gating: DA controls what enters working memory
    - Top-down control: Sends modulatory signals to cortex
    - Input buffering: Accumulates inputs from cortex and hippocampus

    The PFC.forward() method handles:
    - input_spikes: Concatenated input from cortex (L2/3) and hippocampus
    - encoding_mod/retrieval_mod: Theta modulation
    - dopamine_signal: For gating working memory updates

    Input Buffering:
    ================
    PFC receives inputs from both cortex and hippocampus on separate pathways.
    These are buffered and concatenated before processing. This allows for
    event-driven operation where cortex and hippocampus spikes arrive at
    different times.
    """

    def __init__(
        self,
        config: EventRegionConfig,
        pfc: Any,  # Prefrontal instance
        cortex_input_size: int = 0,
        hippocampus_input_size: int = 0,
    ):
        super().__init__(config)
        self.pfc = pfc

        # Input sizes for buffering (set via configure_inputs)
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size

        # Input buffers - accumulate spikes from different sources
        self._cortex_buffer: Optional[torch.Tensor] = None
        self._hippocampus_buffer: Optional[torch.Tensor] = None

        # Time window for accumulating inputs (ms)
        self._accumulation_window: float = 5.0
        self._last_cortex_time: float = -1000.0
        self._last_hippocampus_time: float = -1000.0

        # Accumulate dopamine signal for next forward pass
        self._pending_dopamine_signal: float = 0.0

    def configure_inputs(
        self,
        cortex_input_size: int,
        hippocampus_input_size: int,
    ) -> None:
        """Configure input sizes for buffering.

        Call this after construction if sizes weren't provided.
        """
        self._cortex_input_size = cortex_input_size
        self._hippocampus_input_size = hippocampus_input_size

    def _clear_buffers(self) -> None:
        """Clear input buffers."""
        self._cortex_buffer = None
        self._hippocampus_buffer = None

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to PFC neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay PFC neurons
        if hasattr(self.pfc, "neurons") and hasattr(self.pfc.neurons, "membrane"):
            if self.pfc.neurons.membrane is not None:
                self.pfc.neurons.membrane *= decay_factor

        # Dopamine decays via update() call with 0 signal
        # The update() method handles decay internally
        if hasattr(self.pfc, "dopamine_system"):
            self.pfc.dopamine_system.update(0.0, dt_ms)

    def _on_dopamine(self, payload: DopaminePayload) -> None:
        """Handle dopamine signal for PFC.

        DA level gets passed to PFC.forward() which uses it to:
        - Gate what enters working memory (high DA = update WM)
        - Modulate learning (via dopamine-gated STDP)
        
        With continuous plasticity, dopamine modulates the learning rate
        in forward(). We store it both on the PFC state and for gating.
        """
        # Set dopamine on region state for continuous plasticity modulation
        self.pfc.state.dopamine = payload.level
        
        # Store dopamine signal for next forward pass (WM gating)
        self._pending_dopamine_signal = payload.level

    def _process_spikes(
        self,
        input_spikes: torch.Tensor,
        source: str,
    ) -> Optional[torch.Tensor]:
        """Buffer input spikes and process when we have both inputs.

        PFC requires concatenated input from cortex and hippocampus.
        This method buffers inputs and processes when:
        1. We have input from both sources, OR
        2. One input times out (process with zeros for missing source)
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
        else:
            # Unknown source - try to process directly if sizes match
            return self._forward_pfc(input_spikes)

        # Check if we should process now
        # Process if we have both inputs, or if input sizes weren't configured
        if self._cortex_input_size == 0 and self._hippocampus_input_size == 0:
            # Sizes not configured - just pass through (legacy mode)
            return self._forward_pfc(input_spikes)

        # Build combined input
        combined = self._build_combined_input()
        if combined is not None:
            result = self._forward_pfc(combined)
            self._clear_buffers()
            return result

        # Don't have complete input yet - return None (no output)
        return None

    def _build_combined_input(self) -> Optional[torch.Tensor]:
        """Build combined input from buffers if ready."""
        # If we have both buffers, combine them
        if self._cortex_buffer is not None and self._hippocampus_buffer is not None:
            return torch.cat([self._cortex_buffer, self._hippocampus_buffer], dim=-1)

        # If only cortex arrived and hippocampus timed out, use zeros
        if self._cortex_buffer is not None:
            time_since_hippocampus = self._current_time - self._last_hippocampus_time
            if time_since_hippocampus > self._accumulation_window:
                # Hippocampus timed out - use zeros
                batch_size = self._cortex_buffer.shape[0]
                zeros = torch.zeros(
                    batch_size,
                    self._hippocampus_input_size,
                    device=self._cortex_buffer.device,
                )
                return torch.cat([self._cortex_buffer, zeros], dim=-1)

        # If only hippocampus arrived and cortex timed out, use zeros
        if self._hippocampus_buffer is not None:
            time_since_cortex = self._current_time - self._last_cortex_time
            if time_since_cortex > self._accumulation_window:
                # Cortex timed out - use zeros
                batch_size = self._hippocampus_buffer.shape[0]
                zeros = torch.zeros(
                    batch_size,
                    self._cortex_input_size,
                    device=self._hippocampus_buffer.device,
                )
                return torch.cat([zeros, self._hippocampus_buffer], dim=-1)

        # Not ready yet
        return None

    def _forward_pfc(self, combined_input: torch.Tensor) -> torch.Tensor:
        """Forward combined input through PFC."""
        # Forward through PFC with theta modulation and dopamine
        output = self.pfc.forward(
            combined_input,
            dt=1.0,  # Event-driven doesn't use fixed dt
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            dopamine_signal=self._pending_dopamine_signal,
        )

        # Clear pending dopamine after use
        self._pending_dopamine_signal = 0.0

        return output.squeeze()

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: float = 0.0,
    ) -> None:
        """Apply dopamine-gated STDP learning."""
        if hasattr(self.pfc, "learn"):
            self.pfc.learn(
                input_spikes=input_spikes,
                output_spikes=output_spikes,
                reward=reward,
            )

    def get_state(self) -> Dict[str, Any]:
        """Return PFC state."""
        state = super().get_state()
        if hasattr(self.pfc, "state") and self.pfc.state is not None:
            state["wm_active"] = self.pfc.state.working_memory is not None
            if self.pfc.state.working_memory is not None:
                state["wm_mean"] = float(self.pfc.state.working_memory.mean())
            state["gate_value"] = (
                float(self.pfc.dopamine_system.get_gate())
                if hasattr(self.pfc, "dopamine_system")
                else 0.0
            )
        return state
