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
        self.impl_module = pfc  # Register as public attribute for nn.Module
        self._pfc = pfc  # Keep reference for backwards compatibility

        # Configure input buffering using base class
        if cortex_input_size > 0 and hippocampus_input_size > 0:
            self.configure_input_sources(
                cortex=cortex_input_size,
                hippocampus=hippocampus_input_size,
            )

        # Accumulate dopamine signal for next forward pass
        self._pending_dopamine_signal: float = 0.0

    @property
    def impl(self) -> Any:
        """Return the underlying PFC implementation."""
        return self._pfc

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from the underlying PFC implementation."""
        if hasattr(self.impl, "get_diagnostics"):
            return self.impl.get_diagnostics()
        return {}

    def configure_inputs(
        self,
        cortex_input_size: int,
        hippocampus_input_size: int,
    ) -> None:
        """Configure input sizes for buffering.

        Call this after construction if sizes weren't provided.
        """
        self.configure_input_sources(
            cortex=cortex_input_size,
            hippocampus=hippocampus_input_size,
        )

    def _apply_decay(self, dt_ms: float) -> None:
        """Apply decay to PFC neurons."""
        decay_factor = math.exp(-dt_ms / self._membrane_tau)

        # Decay PFC neurons
        if hasattr(self.impl, "neurons") and hasattr(self.impl.neurons, "membrane"):
            if self.impl.neurons.membrane is not None:
                self.impl.neurons.membrane *= decay_factor

        # Dopamine decays via update() call with 0 signal
        # The update() method handles decay internally
        if hasattr(self.impl, "dopamine_system"):
            self.impl.dopamine_system.update(0.0, dt_ms)

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
        # ADR-005: Keep 1D tensors, no batch dimension
        # input_spikes should be [n_neurons]

        # Buffer input using base class method
        if source in ["cortex", "hippocampus"]:
            self._buffer_input(source, input_spikes)
        else:
            # Unknown source - try to process directly if sizes match
            return self._forward_pfc(input_spikes)

        # Check if we should process now
        # If sizes not configured, pass through (legacy mode)
        if not self._input_sizes:
            return self._forward_pfc(input_spikes)

        # Build combined input using base class method
        combined = self._build_combined_input(
            source_order=["cortex", "hippocampus"],
            require_sources=["cortex"],  # Cortex is required, hippocampus optional
        )

        if combined is not None:
            result = self._forward_pfc(combined)
            self._clear_input_buffers()
            return result

        # Don't have complete input yet - return None (no output)
        return None

    def _forward_pfc(self, combined_input: torch.Tensor) -> torch.Tensor:
        """Forward combined input through PFC."""
        # Forward through PFC (theta modulation computed internally)
        output = self.impl.forward(combined_input, dopamine_signal=self._pending_dopamine_signal)

        # Clear pending dopamine after use
        self._pending_dopamine_signal = 0.0

        return output.squeeze() if output.dim() > 1 else output

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: float = 0.0,
    ) -> None:
        """Apply dopamine-gated STDP learning."""
        if hasattr(self.impl, "learn"):
            self.impl.learn(
                input_spikes=input_spikes,
                output_spikes=output_spikes,
                reward=reward,
            )

    def get_state(self) -> Dict[str, Any]:
        """Return PFC state."""
        state = super().get_state()
        if hasattr(self.impl, "state") and self.impl.state is not None:
            state["wm_active"] = self.impl.state.working_memory is not None
            if self.impl.state.working_memory is not None:
                state["wm_mean"] = float(self.impl.state.working_memory.mean())
            state["gate_value"] = (
                float(self.impl.dopamine_system.get_gate())
                if hasattr(self.impl, "dopamine_system")
                else 0.0
            )
        return state
