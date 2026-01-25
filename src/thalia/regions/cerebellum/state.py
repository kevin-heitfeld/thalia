"""State definitions for Cerebellum region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from thalia.constants.neuromodulation import (
    ACH_BASELINE,
    DA_BASELINE_STANDARD,
    NE_BASELINE,
)
from thalia.core.region_state import BaseRegionState


@dataclass
class GranuleLayerState:
    """State for granule cell layer.

    Attributes:
        mossy_to_granule: Synaptic weights from mossy fibers to granule cells
        granule_neurons: State dict from granule neuron model (ConductanceLIF)
    """

    mossy_to_granule: torch.Tensor
    granule_neurons: Dict[str, Any]  # State from ConductanceLIF.get_state()


@dataclass
class PurkinjeCellState:
    """State for Purkinje cell component.

    Attributes:
        dendrite_voltage: Voltage of each dendritic compartment [n_dendrites]
        dendrite_calcium: Calcium level in each dendrite [n_dendrites]
        soma_neurons: State dict from the soma neuron model (ConductanceLIF)
        last_complex_spike_time: Timestep of last complex spike (for refractory period)
        timestep: Current timestep counter
    """

    dendrite_voltage: torch.Tensor
    dendrite_calcium: torch.Tensor
    soma_neurons: Dict[str, Any]  # State from ConductanceLIF.get_state()
    last_complex_spike_time: int
    timestep: int


@dataclass
class CerebellumState(BaseRegionState):
    """Complete state for Cerebellum region.

    Stores all cerebellar state including:
    - Eligibility traces (input, output, STDP)
    - Climbing fiber error signal
    - Neuron state (classic mode) OR enhanced microcircuit state
    - Short-term plasticity state

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.

    Classic Mode Fields (use_enhanced_microcircuit=False):
    - v_mem, g_exc, g_inh: Direct Purkinje cell neuron state
    - stp_pf_purkinje_state: STP for parallel fiber→Purkinje synapses

    Enhanced Mode Fields (use_enhanced_microcircuit=True):
    - granule_layer_state: Granule cell layer state
    - purkinje_cells_state: List of enhanced Purkinje cell states
    - deep_nuclei_state: Deep cerebellar nuclei state
    - stp_mf_granule_state: STP for mossy fiber→granule synapses
    """

    # ========================================================================
    # TRACE MANAGER STATE (both modes)
    # ========================================================================
    input_trace: Optional[torch.Tensor] = None  # [n_input] or [n_granule] if enhanced
    output_trace: Optional[torch.Tensor] = None  # [n_output]
    stdp_eligibility: Optional[torch.Tensor] = None  # [n_output, n_input/n_granule]

    # ========================================================================
    # CLIMBING FIBER ERROR (both modes)
    # ========================================================================
    climbing_fiber_error: Optional[torch.Tensor] = None  # [n_output] - Error signal from IO
    io_membrane: Optional[torch.Tensor] = None  # [n_output] - IO membrane for gap junctions

    # ========================================================================
    # CLASSIC MODE NEURON STATE (use_enhanced=False)
    # ========================================================================
    v_mem: Optional[torch.Tensor] = None  # [n_output] - Membrane voltage
    g_exc: Optional[torch.Tensor] = None  # [n_output] - Excitatory conductance
    g_inh: Optional[torch.Tensor] = None  # [n_output] - Inhibitory conductance

    # ========================================================================
    # SHORT-TERM PLASTICITY STATE
    # ========================================================================
    stp_pf_purkinje_state: Optional[Dict[str, torch.Tensor]] = None  # Classic mode STP
    stp_mf_granule_state: Optional[Dict[str, torch.Tensor]] = None  # Enhanced mode STP

    # ========================================================================
    # ENHANCED MICROCIRCUIT STATE (use_enhanced=True)
    # ========================================================================
    granule_layer_state: Optional[Dict[str, Any]] = None  # GranuleCellLayer state
    purkinje_cells_state: Optional[list] = None  # List of EnhancedPurkinjeCell states
    deep_nuclei_state: Optional[Dict[str, Any]] = None  # DeepCerebellarNuclei state

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary.

        Returns:
            Dictionary containing all state fields.
        """
        return {
            # Traces
            "input_trace": self.input_trace,
            "output_trace": self.output_trace,
            "stdp_eligibility": self.stdp_eligibility,
            # Error signal
            "climbing_fiber_error": self.climbing_fiber_error,
            "io_membrane": self.io_membrane,
            # Classic neuron state
            "v_mem": self.v_mem,
            "g_exc": self.g_exc,
            "g_inh": self.g_inh,
            # STP state
            "stp_pf_purkinje_state": self.stp_pf_purkinje_state,
            "stp_mf_granule_state": self.stp_mf_granule_state,
            # Neuromodulators
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            # Enhanced microcircuit
            "granule_layer_state": self.granule_layer_state,
            "purkinje_cells_state": self.purkinje_cells_state,
            "deep_nuclei_state": self.deep_nuclei_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> CerebellumState:
        """Deserialize state from dictionary.

        Args:
            data: Dictionary from to_dict()
            device: Target device for tensors

        Returns:
            CerebellumState instance with tensors on specified device.
        """
        dev = torch.device(device)

        # Helper to transfer tensors to device
        def to_device(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return tensor.to(dev) if tensor is not None else None

        # Helper for STP state dicts
        def stp_to_device(
            stp_state: Optional[Dict[str, torch.Tensor]],
        ) -> Optional[Dict[str, torch.Tensor]]:
            if stp_state is None:
                return None
            return {k: v.to(dev) if v is not None else None for k, v in stp_state.items()}

        return cls(
            # Traces (may be None if state is uninitialized)
            input_trace=to_device(data.get("input_trace")),
            output_trace=to_device(data.get("output_trace")),
            stdp_eligibility=to_device(data.get("stdp_eligibility")),
            # Error signal (may be None if state is uninitialized)
            climbing_fiber_error=to_device(data.get("climbing_fiber_error")),
            # Classic neuron state (optional)
            v_mem=to_device(data.get("v_mem")),
            g_exc=to_device(data.get("g_exc")),
            g_inh=to_device(data.get("g_inh")),
            # STP state (optional)
            stp_pf_purkinje_state=stp_to_device(data.get("stp_pf_purkinje_state")),
            stp_mf_granule_state=stp_to_device(data.get("stp_mf_granule_state")),
            # Neuromodulators (base tonic baseline)
            dopamine=data.get("dopamine", DA_BASELINE_STANDARD),
            acetylcholine=data.get("acetylcholine", ACH_BASELINE),
            norepinephrine=data.get("norepinephrine", NE_BASELINE),
            # Enhanced microcircuit (optional)
            granule_layer_state=data.get("granule_layer_state"),
            purkinje_cells_state=data.get("purkinje_cells_state"),
            deep_nuclei_state=data.get("deep_nuclei_state"),
        )

    def reset(self) -> None:
        """Reset state in-place (clears traces, resets neurons)."""
        # Reset base state (spikes, membrane, neuromodulators)
        super().reset()

        # Clear traces (if initialized)
        if self.input_trace is not None:
            self.input_trace.zero_()
        if self.output_trace is not None:
            self.output_trace.zero_()
        if self.stdp_eligibility is not None:
            self.stdp_eligibility.zero_()

        # Clear error (if initialized)
        if self.climbing_fiber_error is not None:
            self.climbing_fiber_error.zero_()

        # Reset classic neuron state
        if self.v_mem is not None:
            self.v_mem.fill_(-70.0)  # Resting potential
        if self.g_exc is not None:
            self.g_exc.zero_()
        if self.g_inh is not None:
            self.g_inh.zero_()

        # Reset STP state
        if self.stp_pf_purkinje_state is not None:
            if "u" in self.stp_pf_purkinje_state:
                self.stp_pf_purkinje_state["u"].zero_()
            if "x" in self.stp_pf_purkinje_state:
                self.stp_pf_purkinje_state["x"].fill_(1.0)  # Resources fully available

        if self.stp_mf_granule_state is not None:
            if "u" in self.stp_mf_granule_state:
                self.stp_mf_granule_state["u"].zero_()
            if "x" in self.stp_mf_granule_state:
                self.stp_mf_granule_state["x"].fill_(1.0)

        # NOTE: Enhanced microcircuit states are complex nested structures
        # They should be reset through their respective subsystems
