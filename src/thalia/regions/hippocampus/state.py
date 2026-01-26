"""State representation for the hippocampus region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from thalia.core.region_state import BaseRegionState


@dataclass
class HippocampusState(BaseRegionState):
    """State for hippocampus (DG→CA3→CA2→CA1 circuit) with RegionState protocol compliance.

    Extends BaseRegionState with hippocampus-specific state:
    - DG/CA3/CA2/CA1 layer activities and traces
    - CA3 persistent activity (attractor dynamics)
    - Sample trace for memory encoding
    - STDP traces for multiple pathways
    - NMDA trace for temporal integration
    - Stored DG pattern for match/mismatch detection
    - Feedforward inhibition strength
    - Short-term plasticity (STP) state for 7 pathways

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.

    The CA1 spikes ARE the output - no interpretation needed!
    Different CA1 spike patterns naturally emerge for match vs mismatch
    through the coincidence detection between CA3 (memory) and EC (current).
    """

    # Layer activities (current spikes)
    dg_spikes: Optional[torch.Tensor] = None
    ca3_spikes: Optional[torch.Tensor] = None
    ca2_spikes: Optional[torch.Tensor] = None  # CA2: Social memory and temporal context
    ca1_spikes: Optional[torch.Tensor] = None

    # CA3 recurrent state
    ca3_membrane: Optional[torch.Tensor] = None

    # CA1 membrane voltages (for gap junction coupling)
    ca1_membrane: Optional[torch.Tensor] = None

    # CA3 bistable persistent activity trace
    # Models I_NaP/I_CAN currents that allow neurons to maintain firing
    # without continuous external input. This is essential for stable
    # attractor states during delay periods.
    ca3_persistent: Optional[torch.Tensor] = None

    # Memory trace (for STDP learning during sample phase)
    sample_trace: Optional[torch.Tensor] = None

    # STDP traces
    dg_trace: Optional[torch.Tensor] = None
    ca3_trace: Optional[torch.Tensor] = None
    ca2_trace: Optional[torch.Tensor] = None

    # NMDA trace for temporal integration (slow kinetics)
    nmda_trace: Optional[torch.Tensor] = None

    # Stored DG pattern from sample phase (for match/mismatch detection)
    stored_dg_pattern: Optional[torch.Tensor] = None

    # Current feedforward inhibition strength
    ffi_strength: float = 0.0

    # Spontaneous replay (sharp-wave ripple) detection
    ripple_detected: bool = False

    # Short-term plasticity state for 7 pathways
    stp_mossy_state: Optional[Dict[str, torch.Tensor]] = None  # DG→CA3 facilitation
    stp_ca3_ca2_state: Optional[Dict[str, torch.Tensor]] = None  # CA3→CA2 depression
    stp_ca2_ca1_state: Optional[Dict[str, torch.Tensor]] = None  # CA2→CA1 facilitation
    stp_ec_ca2_state: Optional[Dict[str, torch.Tensor]] = None  # EC→CA2 direct
    stp_schaffer_state: Optional[Dict[str, torch.Tensor]] = None  # CA3→CA1 depression
    stp_ec_ca1_state: Optional[Dict[str, torch.Tensor]] = None  # EC→CA1 direct
    stp_ca3_recurrent_state: Optional[Dict[str, torch.Tensor]] = None  # CA3 recurrent

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing.

        Returns:
            Dictionary with all state fields, including nested STP states for 7 pathways.
        """
        return {
            # Base region state
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            # Layer activities
            "dg_spikes": self.dg_spikes,
            "ca3_spikes": self.ca3_spikes,
            "ca2_spikes": self.ca2_spikes,
            "ca1_spikes": self.ca1_spikes,
            # CA3 state
            "ca3_membrane": self.ca3_membrane,
            "ca3_persistent": self.ca3_persistent,
            "ripple_detected": self.ripple_detected,
            # CA1 state (gap junctions)
            "ca1_membrane": self.ca1_membrane,
            # Memory and traces
            "sample_trace": self.sample_trace,
            "dg_trace": self.dg_trace,
            "ca3_trace": self.ca3_trace,
            "ca2_trace": self.ca2_trace,
            "nmda_trace": self.nmda_trace,
            "stored_dg_pattern": self.stored_dg_pattern,
            "ffi_strength": self.ffi_strength,
            # STP state (nested dicts for 7 pathways)
            "stp_mossy_state": self.stp_mossy_state,
            "stp_ca3_ca2_state": self.stp_ca3_ca2_state,
            "stp_ca2_ca1_state": self.stp_ca2_ca1_state,
            "stp_ec_ca2_state": self.stp_ec_ca2_state,
            "stp_schaffer_state": self.stp_schaffer_state,
            "stp_ec_ca1_state": self.stp_ec_ca1_state,
            "stp_ca3_recurrent_state": self.stp_ca3_recurrent_state,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        device: str = "cpu",
    ) -> HippocampusState:
        """Deserialize state from dictionary.

        Args:
            data: Dictionary with state fields
            device: Target device string (e.g., 'cpu', 'cuda', 'cuda:0')

        Returns:
            HippocampusState instance with restored state
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
            # Layer activities
            dg_spikes=transfer_tensor(data.get("dg_spikes")),
            ca3_spikes=transfer_tensor(data.get("ca3_spikes")),
            ca2_spikes=transfer_tensor(
                data.get("ca2_spikes")
            ),  # Backward compatible (None if missing)
            ca1_spikes=transfer_tensor(data.get("ca1_spikes")),
            # CA3 state
            ca3_membrane=transfer_tensor(data.get("ca3_membrane")),
            ca3_persistent=transfer_tensor(data.get("ca3_persistent")),
            ripple_detected=data.get("ripple_detected", False),
            # CA1 state (gap junctions, added 2025-01, backward compatible)
            ca1_membrane=transfer_tensor(data.get("ca1_membrane")),
            # Memory and traces
            sample_trace=transfer_tensor(data.get("sample_trace")),
            dg_trace=transfer_tensor(data.get("dg_trace")),
            ca3_trace=transfer_tensor(data.get("ca3_trace")),
            ca2_trace=transfer_tensor(
                data.get("ca2_trace")
            ),  # Backward compatible (None if missing)
            nmda_trace=transfer_tensor(data.get("nmda_trace")),
            stored_dg_pattern=transfer_tensor(data.get("stored_dg_pattern")),
            ffi_strength=data.get("ffi_strength", 0.0),
            # STP state (nested dicts for 7 pathways, backward compatible)
            stp_mossy_state=transfer_nested_dict(data.get("stp_mossy_state")),
            stp_ca3_ca2_state=transfer_nested_dict(data.get("stp_ca3_ca2_state")),
            stp_ca2_ca1_state=transfer_nested_dict(data.get("stp_ca2_ca1_state")),
            stp_ec_ca2_state=transfer_nested_dict(data.get("stp_ec_ca2_state")),
            stp_schaffer_state=transfer_nested_dict(data.get("stp_schaffer_state")),
            stp_ec_ca1_state=transfer_nested_dict(data.get("stp_ec_ca1_state")),
            stp_ca3_recurrent_state=transfer_nested_dict(data.get("stp_ca3_recurrent_state")),
        )

    def reset(self) -> None:
        """Reset state to default values (in-place mutation)."""
        # Reset base state (spikes, membrane, neuromodulators with DA_BASELINE_STANDARD)
        super().reset()

        # Reset hippocampus-specific state
        self.dg_spikes = None
        self.ca3_spikes = None
        self.ca2_spikes = None
        self.ca1_spikes = None
        self.ca3_membrane = None
        self.ca3_persistent = None
        self.sample_trace = None
        self.dg_trace = None
        self.ca3_trace = None
        self.ca2_trace = None
        self.nmda_trace = None
        self.stored_dg_pattern = None
        self.ffi_strength = 0.0
        self.stp_mossy_state = None
        self.stp_ca3_ca2_state = None
        self.stp_ca2_ca1_state = None
        self.stp_ec_ca2_state = None
        self.stp_schaffer_state = None
        self.stp_ec_ca1_state = None
        self.stp_ca3_recurrent_state = None
