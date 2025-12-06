"""
Event-Driven Hippocampus Adapter.

Wraps TrisynapticHippocampus for event-driven simulation.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import torch

from ..event_system import SpikePayload
from .base import EventDrivenRegionBase, EventRegionConfig


class EventDrivenHippocampus(EventDrivenRegionBase):
    """Event-driven wrapper for TrisynapticHippocampus.

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
        hippocampus: Any,  # TrisynapticHippocampus instance
    ):
        super().__init__(config)
        self._hippocampus = hippocampus

        # Track EC direct input (from sensory, bypasses cortex)
        self._ec_direct_input: Optional[torch.Tensor] = None

        # Import TrialPhase for phase determination
        from thalia.regions.theta_dynamics import TrialPhase

        self._TrialPhase = TrialPhase

    @property
    def impl(self) -> Any:
        """Return the underlying hippocampus implementation."""
        return self._hippocampus

    @property
    def state(self) -> Any:
        """Delegate state access to the underlying implementation."""
        return getattr(self.impl, "state", None)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics from the underlying hippocampus implementation."""
        if hasattr(self.impl, "get_diagnostics"):
            return self.impl.get_diagnostics()
        return {}

    def _get_trial_phase(self) -> Any:
        """Determine trial phase from theta modulation.

        Encoding strength high → ENCODE phase
        Retrieval strength high → RETRIEVE phase
        Neither dominant → DELAY phase
        """
        if self._encoding_strength > 0.6:
            return self._TrialPhase.ENCODE
        elif self._retrieval_strength > 0.6:
            return self._TrialPhase.RETRIEVE
        else:
            return self._TrialPhase.DELAY

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
        # Ensure batch dimension
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # Determine phase from theta
        phase = self._get_trial_phase()

        # Forward through hippocampus
        output = self.impl.forward(
            input_spikes,
            phase=phase,
            encoding_mod=self._encoding_strength,
            retrieval_mod=self._retrieval_strength,
            dt=1.0,  # Event-driven doesn't use fixed dt
            ec_direct_input=self._ec_direct_input,
        )

        return output.squeeze()

    def new_trial(self) -> None:
        """Signal new trial to hippocampus."""
        if hasattr(self.impl, "new_trial"):
            self.impl.new_trial()
        self._ec_direct_input = None

    def get_state(self) -> Dict[str, Any]:
        """Return hippocampus state."""
        state = super().get_state()
        state["trial_phase"] = self._get_trial_phase().name
        if hasattr(self.impl, "get_diagnostics"):
            state["impl"] = self.impl.get_diagnostics()
        return state
