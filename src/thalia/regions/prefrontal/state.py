"""State definition for prefrontal cortex region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from thalia.core.region_state import BaseRegionState


@dataclass
class PrefrontalState(BaseRegionState):
    """State for prefrontal cortex region.

    Implements RegionState protocol for checkpoint compatibility.
    Inherits from BaseRegionState for common fields (spikes, membrane, neuromodulators).

    PFC-specific state:
    - working_memory: Active maintenance of task-relevant information
    - update_gate: Dopamine-gated update signals
    - active_rule: Current task rule representation
    """

    STATE_VERSION: int = 1

    # Inherited from BaseRegionState:
    # - spikes: Optional[torch.Tensor] = None
    # - membrane: Optional[torch.Tensor] = None
    # - dopamine: float = 0.0
    # - acetylcholine: float = 0.0
    # - norepinephrine: float = 0.0

    # PFC-specific state fields
    working_memory: Optional[torch.Tensor] = None
    """Working memory contents [n_neurons]."""

    update_gate: Optional[torch.Tensor] = None
    """Gate state for WM updates [n_neurons]."""

    active_rule: Optional[torch.Tensor] = None
    """Rule representation [n_neurons]."""

    # STP state (recurrent connections)
    stp_recurrent_state: Optional[Dict[str, Any]] = None
    """Short-term plasticity state for recurrent connections."""

    stp_feedforward_state: Optional[Dict[str, Any]] = None
    """Short-term plasticity state for feedforward connections."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "working_memory": self.working_memory,
                "update_gate": self.update_gate,
                "active_rule": self.active_rule,
                "dopamine": self.dopamine,
                "acetylcholine": self.acetylcholine,
                "norepinephrine": self.norepinephrine,
                "stp_recurrent_state": self.stp_recurrent_state,
                "stp_feedforward_state": self.stp_feedforward_state,
            }
        )
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str) -> PrefrontalState:
        """Deserialize state from dictionary."""
        # Future: Handle version migration if needed

        # Get base state
        base_state = BaseRegionState.from_dict(data, device)

        # Transfer PFC-specific tensors
        wm = data.get("working_memory")
        if wm is not None and isinstance(wm, torch.Tensor):
            wm = wm.to(device)

        gate = data.get("update_gate")
        if gate is not None and isinstance(gate, torch.Tensor):
            gate = gate.to(device)

        rule = data.get("active_rule")
        if rule is not None and isinstance(rule, torch.Tensor):
            rule = rule.to(device)

        # Transfer nested STP state tensors
        stp_recurrent = data.get("stp_recurrent_state")
        if stp_recurrent is not None:
            stp_recurrent = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in stp_recurrent.items()
            }

        stp_feedforward = data.get("stp_feedforward_state")
        if stp_feedforward is not None:
            stp_feedforward = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in stp_feedforward.items()
            }

        return cls(
            spikes=base_state.spikes,
            membrane=base_state.membrane,
            working_memory=wm,
            update_gate=gate,
            active_rule=rule,
            dopamine=data.get("dopamine", 0.2),
            acetylcholine=data.get("acetylcholine", 0.0),
            norepinephrine=data.get("norepinephrine", 0.0),
            stp_recurrent_state=stp_recurrent,
            stp_feedforward_state=stp_feedforward,
        )

    def reset(self) -> None:
        """Reset state to initial conditions."""
        # Reset base fields (spikes, membrane, neuromodulators with DA_BASELINE_STANDARD)
        super().reset()

        # Reset PFC-specific state
        self.working_memory = None
        self.update_gate = None
        self.active_rule = None
        self.stp_recurrent_state = None
