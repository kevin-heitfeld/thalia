"""State representation for thalamic relay nucleus region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from thalia.core.region_state import BaseRegionState


@dataclass
class ThalamicRelayState(BaseRegionState):
    """State for thalamic relay nucleus with RegionState protocol compliance.

    Extends BaseRegionState with thalamus-specific state:
    - Relay and TRN neuron states (spikes, membrane potentials)
    - Burst/tonic mode state
    - Alpha oscillation gating state
    - Short-term plasticity (STP) state for sensory and L6 feedback pathways

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.
    """

    # Relay neuron state
    relay_spikes: Optional[torch.Tensor] = None
    relay_membrane: Optional[torch.Tensor] = None

    # TRN state
    trn_spikes: Optional[torch.Tensor] = None
    trn_membrane: Optional[torch.Tensor] = None

    # Mode state
    current_mode: Optional[torch.Tensor] = None  # 0=burst, 1=tonic
    burst_counter: Optional[torch.Tensor] = None  # Tracks spikes in burst

    # Gating state
    alpha_gate: Optional[torch.Tensor] = None  # Current gating factor [0, 1]

    # Short-term plasticity state (HIGH PRIORITY for sensory gating)
    stp_sensory_relay_state: Optional[Dict[str, torch.Tensor]] = None  # Sensory → relay STP
    stp_l6_feedback_state: Optional[Dict[str, torch.Tensor]] = None  # L6 → relay STP

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing.

        Returns:
            Dictionary with all state fields, including nested STP states.
        """
        return {
            # Base region state
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            # Relay neuron state
            "relay_spikes": self.relay_spikes,
            "relay_membrane": self.relay_membrane,
            # TRN state
            "trn_spikes": self.trn_spikes,
            "trn_membrane": self.trn_membrane,
            # Mode state
            "current_mode": self.current_mode,
            "burst_counter": self.burst_counter,
            # Gating state
            "alpha_gate": self.alpha_gate,
            # STP state (nested dicts)
            "stp_sensory_relay_state": self.stp_sensory_relay_state,
            "stp_l6_feedback_state": self.stp_l6_feedback_state,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],  # Changed from state_dict to data to match base class
        device: str = "cpu",  # Changed from Optional[torch.device] to str to match base class
    ) -> ThalamicRelayState:
        """Deserialize state from dictionary.

        Args:
            data: Dictionary with state fields
            device: Target device string (e.g., 'cpu', 'cuda', 'cuda:0')

        Returns:
            ThalamicRelayState instance with restored state
        """

        def transfer_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return t
            return t.to(device)

        def transfer_nested_dict(
            d: Optional[Dict[str, torch.Tensor]],
        ) -> Optional[Dict[str, torch.Tensor]]:
            """Transfer nested dict of tensors to device."""
            if d is None:
                return d
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

        return cls(
            # Base region state
            dopamine=data.get("dopamine", 0.2),
            acetylcholine=data.get("acetylcholine", 0.0),
            norepinephrine=data.get("norepinephrine", 0.0),
            # Relay neuron state
            relay_spikes=transfer_tensor(data.get("relay_spikes")),
            relay_membrane=transfer_tensor(data.get("relay_membrane")),
            # TRN state
            trn_spikes=transfer_tensor(data.get("trn_spikes")),
            trn_membrane=transfer_tensor(data.get("trn_membrane")),
            # Mode state
            current_mode=transfer_tensor(data.get("current_mode")),
            burst_counter=transfer_tensor(data.get("burst_counter")),
            # Gating state
            alpha_gate=transfer_tensor(data.get("alpha_gate")),
            # STP state (nested dicts)
            stp_sensory_relay_state=transfer_nested_dict(data.get("stp_sensory_relay_state")),
            stp_l6_feedback_state=transfer_nested_dict(data.get("stp_l6_feedback_state")),
        )

    def reset(self) -> None:
        """Reset state to default values (in-place mutation)."""
        # Reset base state (neuromodulators with DA_BASELINE_STANDARD)
        super().reset()

        # Reset thalamus-specific state
        self.relay_spikes = None
        self.relay_membrane = None
        self.trn_spikes = None
        self.trn_membrane = None
        self.current_mode = None
        self.burst_counter = None
        self.alpha_gate = None
        self.stp_sensory_relay_state = None
        self.stp_l6_feedback_state = None
