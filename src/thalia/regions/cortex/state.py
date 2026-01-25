"""State dataclasses for cortex regions."""

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
class LayeredCortexState(BaseRegionState):
    """State for layered cortex with RegionState protocol compliance.

    Extends BaseRegionState with cortex-specific state:
    - 6-layer architecture (L4, L2/3, L5, L6a, L6b) with spikes and traces
    - L2/3 recurrent activity accumulation
    - Top-down modulation and attention gating
    - Feedforward inhibition and alpha suppression
    - Short-term plasticity (STP) state for L2/3 recurrent pathway

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.

    The 6-layer structure reflects canonical cortical microcircuit:
    - L4: Main input layer (thalamic recipient)
    - L2/3: Cortico-cortical output and recurrent processing
    - L5: Subcortical output (motor, striatum)
    - L6a: TRN feedback for attentional gating
    - L6b: Relay feedback for gain control
    """

    # Input stored for continuous plasticity
    input_spikes: Optional[torch.Tensor] = None
    source_inputs: Optional[Dict[str, torch.Tensor]] = None

    # Per-layer spike states (6 layers)
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None
    l6a_spikes: Optional[torch.Tensor] = None  # L6a → TRN pathway
    l6b_spikes: Optional[torch.Tensor] = None  # L6b → relay pathway

    # L2/3 membrane potential (for gap junction coupling)
    l23_membrane: Optional[torch.Tensor] = None

    # L2/3 recurrent activity (accumulated over time)
    l23_recurrent_activity: Optional[torch.Tensor] = None

    # STDP traces per layer (5 layers)
    l4_trace: Optional[torch.Tensor] = None
    l23_trace: Optional[torch.Tensor] = None
    l5_trace: Optional[torch.Tensor] = None
    l6a_trace: Optional[torch.Tensor] = None  # L6a trace for TRN feedback plasticity
    l6b_trace: Optional[torch.Tensor] = None  # L6b trace for relay feedback plasticity

    # Top-down modulation state
    top_down_modulation: Optional[torch.Tensor] = None

    # Feedforward inhibition strength (0-1, 1 = max suppression)
    ffi_strength: float = 0.0

    # Alpha oscillation suppression (0-1, 1 = no suppression, 0.5 = max suppression)
    alpha_suppression: float = 1.0

    # Gamma attention state (spike-native phase gating)
    gamma_attention_phase: Optional[float] = None  # Current gamma phase
    gamma_attention_gate: Optional[torch.Tensor] = None  # Per-neuron gating [l23_size]

    # Last plasticity delta (for monitoring continuous learning)
    last_plasticity_delta: float = 0.0

    # Short-term plasticity state for L2/3 recurrent pathway
    stp_l23_recurrent_state: Optional[Dict[str, torch.Tensor]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing.

        Returns:
            Dictionary with all state fields, including nested STP state for L2/3 recurrent.
        """
        return {
            # Base region state
            "spikes": self.spikes,
            "membrane": self.membrane,
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            # Input
            "input_spikes": self.input_spikes,
            # Layer spike states
            "l4_spikes": self.l4_spikes,
            "l23_spikes": self.l23_spikes,
            "l5_spikes": self.l5_spikes,
            "l6a_spikes": self.l6a_spikes,
            "l6b_spikes": self.l6b_spikes,
            # L2/3 membrane for gap junctions
            "l23_membrane": self.l23_membrane,
            # L2/3 recurrent activity
            "l23_recurrent_activity": self.l23_recurrent_activity,
            # STDP traces
            "l4_trace": self.l4_trace,
            "l23_trace": self.l23_trace,
            "l5_trace": self.l5_trace,
            "l6a_trace": self.l6a_trace,
            "l6b_trace": self.l6b_trace,
            # Modulation state
            "top_down_modulation": self.top_down_modulation,
            "ffi_strength": self.ffi_strength,
            "alpha_suppression": self.alpha_suppression,
            # Gamma attention
            "gamma_attention_phase": self.gamma_attention_phase,
            "gamma_attention_gate": self.gamma_attention_gate,
            # Plasticity monitoring
            "last_plasticity_delta": self.last_plasticity_delta,
            # STP state
            "stp_l23_recurrent_state": self.stp_l23_recurrent_state,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        device: str = "cpu",
    ) -> LayeredCortexState:
        """Deserialize state from dictionary.

        Args:
            data: Dictionary with state fields
            device: Target device string (e.g., 'cpu', 'cuda', 'cuda:0')

        Returns:
            LayeredCortexState instance with restored state
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
                return None
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

        return cls(
            # Base region state
            spikes=transfer_tensor(data.get("spikes")),
            membrane=transfer_tensor(data.get("membrane")),
            dopamine=data.get("dopamine", DA_BASELINE_STANDARD),
            acetylcholine=data.get("acetylcholine", ACH_BASELINE),
            norepinephrine=data.get("norepinephrine", NE_BASELINE),
            # Input
            input_spikes=transfer_tensor(data.get("input_spikes")),
            # Layer spike states
            l4_spikes=transfer_tensor(data.get("l4_spikes")),
            l23_spikes=transfer_tensor(data.get("l23_spikes")),
            l5_spikes=transfer_tensor(data.get("l5_spikes")),
            l6a_spikes=transfer_tensor(data.get("l6a_spikes")),
            l6b_spikes=transfer_tensor(data.get("l6b_spikes")),
            # L2/3 recurrent activity
            l23_recurrent_activity=transfer_tensor(data.get("l23_recurrent_activity")),
            # STDP traces
            l4_trace=transfer_tensor(data.get("l4_trace")),
            l23_trace=transfer_tensor(data.get("l23_trace")),
            l5_trace=transfer_tensor(data.get("l5_trace")),
            l6a_trace=transfer_tensor(data.get("l6a_trace")),
            l6b_trace=transfer_tensor(data.get("l6b_trace")),
            # Modulation state
            top_down_modulation=transfer_tensor(data.get("top_down_modulation")),
            ffi_strength=data.get("ffi_strength", 0.0),
            alpha_suppression=data.get("alpha_suppression", 1.0),
            # Gamma attention
            gamma_attention_phase=data.get("gamma_attention_phase"),
            gamma_attention_gate=transfer_tensor(data.get("gamma_attention_gate")),
            # Plasticity monitoring
            last_plasticity_delta=data.get("last_plasticity_delta", 0.0),
            # STP state
            stp_l23_recurrent_state=transfer_nested_dict(data.get("stp_l23_recurrent_state")),
            # Gap junction state (added 2025-01, backward compatible)
            l23_membrane=transfer_tensor(data.get("l23_membrane")),
        )

    def reset(self) -> None:
        """Reset state to initial values (in-place mutation).

        Zeros all tensors and resets scalars to defaults.
        This is called when starting a new simulation or resetting the region.
        """
        # Reset base state (spikes, membrane, neuromodulators)
        super().reset()

        # Reset input spikes
        if self.input_spikes is not None:
            self.input_spikes.zero_()

        # Reset layer spikes
        if self.l4_spikes is not None:
            self.l4_spikes.zero_()
        if self.l23_spikes is not None:
            self.l23_spikes.zero_()
        if self.l5_spikes is not None:
            self.l5_spikes.zero_()
        if self.l6a_spikes is not None:
            self.l6a_spikes.zero_()
        if self.l6b_spikes is not None:
            self.l6b_spikes.zero_()

        # Reset L2/3 recurrent activity
        if self.l23_recurrent_activity is not None:
            self.l23_recurrent_activity.zero_()

        # Reset traces
        if self.l4_trace is not None:
            self.l4_trace.zero_()
        if self.l23_trace is not None:
            self.l23_trace.zero_()
        if self.l5_trace is not None:
            self.l5_trace.zero_()
        if self.l6a_trace is not None:
            self.l6a_trace.zero_()
        if self.l6b_trace is not None:
            self.l6b_trace.zero_()

        # Reset modulation state
        if self.top_down_modulation is not None:
            self.top_down_modulation.zero_()
        if self.gamma_attention_gate is not None:
            self.gamma_attention_gate.zero_()

        # Reset scalars (neuromodulators handled by BaseRegionState.reset())
        self.ffi_strength = 0.0
        self.alpha_suppression = 1.0  # Reset to no suppression
        self.gamma_attention_phase = None
        self.last_plasticity_delta = 0.0

        # NOTE: STP state is NOT reset here - it's managed by the STP module


@dataclass
class PredictiveCortexState(BaseRegionState):
    """State for predictive cortex with RegionState protocol compliance.

    Extends BaseRegionState with predictive cortex-specific fields:
    - Layer-specific spike states (L4, L2/3, L5, L6a, L6b)
    - Predictive coding state (prediction, error, precision, free energy)
    - Attention weights
    - Oscillator signals (for alpha suppression, theta modulation, etc.)

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.
    """

    # Layer-specific spikes (cortex has L4→L2/3→L5+L6 microcircuit)
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None
    l6a_spikes: Optional[torch.Tensor] = None
    l6b_spikes: Optional[torch.Tensor] = None

    # Multi-source inputs for learning (dict mapping source names to spike tensors)
    source_inputs: Optional[Dict[str, torch.Tensor]] = None

    # Predictive coding
    prediction: Optional[torch.Tensor] = None
    error: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    free_energy: float = 0.0

    # Attention
    attention_weights: Optional[torch.Tensor] = None

    # Oscillator signals (for alpha suppression, theta modulation, etc.)
    # Note: Actual alpha gating happens in inner LayeredCortex
    _oscillator_phases: Optional[Dict[str, float]] = None
    _oscillator_signals: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary.

        Returns:
            Dictionary with all state fields
        """
        # Start with base fields (spikes, membrane, neuromodulators)
        data = super().to_dict()

        # Add predictive cortex-specific fields
        data.update(
            {
                # Layer spikes
                "l4_spikes": self.l4_spikes,
                "l23_spikes": self.l23_spikes,
                "l5_spikes": self.l5_spikes,
                "l6a_spikes": self.l6a_spikes,
                "l6b_spikes": self.l6b_spikes,
                # Multi-source inputs
                "source_inputs": self.source_inputs,
                # Predictive coding
                "prediction": self.prediction,
                "error": self.error,
                "precision": self.precision,
                "free_energy": self.free_energy,
                # Attention
                "attention_weights": self.attention_weights,
                # Oscillators (can be None)
                "_oscillator_phases": self._oscillator_phases,
                "_oscillator_signals": self._oscillator_signals,
            }
        )

        return data

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        device: str = "cpu",
    ) -> PredictiveCortexState:
        """Deserialize state from dictionary.

        Args:
            data: Dictionary with state fields
            device: Target device string (e.g., 'cpu', 'cuda', 'cuda:0')

        Returns:
            PredictiveCortexState instance with restored state
        """

        def transfer_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.to(device)

        return cls(
            # Base region state
            spikes=transfer_tensor(data.get("spikes")),
            membrane=transfer_tensor(data.get("membrane")),
            dopamine=data.get("dopamine", 0.2),
            acetylcholine=data.get("acetylcholine", 0.0),
            norepinephrine=data.get("norepinephrine", 0.0),
            # Layer spikes
            l4_spikes=transfer_tensor(data.get("l4_spikes")),
            l23_spikes=transfer_tensor(data.get("l23_spikes")),
            l5_spikes=transfer_tensor(data.get("l5_spikes")),
            l6a_spikes=transfer_tensor(data.get("l6a_spikes")),
            l6b_spikes=transfer_tensor(data.get("l6b_spikes")),
            # Multi-source inputs (backwards compat: also check old "input_spikes")
            source_inputs=(
                {k: transfer_tensor(v) for k, v in data["source_inputs"].items()}
                if "source_inputs" in data and data["source_inputs"] is not None
                else (
                    {"input": transfer_tensor(data["input_spikes"])}
                    if "input_spikes" in data and data["input_spikes"] is not None
                    else None
                )
            ),
            # Predictive coding
            prediction=transfer_tensor(data.get("prediction")),
            error=transfer_tensor(data.get("error")),
            precision=transfer_tensor(data.get("precision")),
            free_energy=data.get("free_energy", 0.0),
            # Attention
            attention_weights=transfer_tensor(data.get("attention_weights")),
            # Oscillators
            _oscillator_phases=data.get("_oscillator_phases"),
            _oscillator_signals=data.get("_oscillator_signals"),
        )

    def reset(self) -> None:
        """Reset state to initial conditions.

        Clears all tensors and restores baseline neuromodulator levels.
        """
        # Reset base fields
        super().reset()

        # Reset predictive cortex-specific fields
        self.l4_spikes = None
        self.l23_spikes = None
        self.l5_spikes = None
        self.l6a_spikes = None
        self.l6b_spikes = None
        self.input_spikes = None
        self.prediction = None
        self.error = None
        self.precision = None
        self.free_energy = 0.0
        self.attention_weights = None
        self._oscillator_phases = None
        self._oscillator_signals = None


@dataclass
class PredictiveCodingState:
    """State of a predictive coding layer."""

    # Representations
    prediction: Optional[torch.Tensor] = None  # Current prediction of input
    representation: Optional[torch.Tensor] = None  # Internal representation
    error: Optional[torch.Tensor] = None  # Prediction error

    # For spiking implementation
    prediction_membrane: Optional[torch.Tensor] = None
    error_membrane: Optional[torch.Tensor] = None

    # Precision tracking
    precision: Optional[torch.Tensor] = None

    # Learning eligibility
    eligibility: Optional[torch.Tensor] = None
