"""
Predictive Cortex - Layered Cortex with Predictive Coding Integration.

This module extends the LayeredCortex with predictive coding capabilities,
creating a cortical hierarchy that learns by minimizing prediction error
rather than through backpropagation.

Architecture:
=============

    Higher Areas (PFC, etc.)
           │
           │ Predictions (top-down via L5/6)
           ▼
    ┌──────────────────────────────────────────────────────┐
    │                  PREDICTIVE CORTEX                    │
    │                                                       │
    │   L2/3 ────────────────────────────────────────────── │
    │    │     Error neurons (superficial pyramidal)        │
    │    │     Send errors UP to higher areas               │
    │    │                                                  │
    │    ▼                                                  │
    │   L4 ───────────────────────────────────────────────  │
    │    │     Input layer (spiny stellates)                │
    │    │     Receives sensory/lower area input            │
    │    │                                                  │
    │    ▼                                                  │
    │   L5/6 ─────────────────────────────────────────────  │
    │          Prediction neurons (deep pyramidal)          │
    │          Send predictions DOWN to L4                  │
    │          Also project to subcortical targets          │
    │                                                       │
    └──────────────────────────────────────────────────────┘
           ▲
           │ Bottom-up input
    Lower Areas / Thalamus

The Key Innovation:
===================
Traditional cortex: L4 → L2/3 → L5 (feedforward hierarchy)
Predictive cortex: L5+L6 → L4 (predictions) + L4 vs L5+L6 → L2/3 (errors)

L5 and L6 are BOTH prediction neurons (deep pyramidal layers):
- L5: Predicts cortical/subcortical activity
- L6: Predicts thalamic input (via TRN modulation)
- Together: Combined representation for predictive coding

Learning Rule:
- Error = Input - Prediction
- ΔW_prediction ∝ Error × Representation
- This is LOCAL - no need to propagate gradients!

Integration with Attention:
- Precision = inverse error variance
- High precision → attend to this input
- Attention modulates precision (PFC → precision weights)

FILE ORGANIZATION (790 lines)
==============================
Lines 1-80:    Module docstring, imports
Lines 81-135:  PredictiveCortexConfig dataclass
Lines 136-310: PredictiveCortex class __init__, layer setup
Lines 311-450: Forward pass (prediction, error computation)
Lines 451-580: Predictive learning (precision-weighted updates)
Lines 581-700: Growth and diagnostics
Lines 701-790: Utility methods (reset_state, checkpointing)

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) to jump between methods.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.config.region_configs import PredictiveCortexConfig
from thalia.core.neural_region import NeuralRegion
from thalia.core.region_state import BaseRegionState
from thalia.managers.component_registry import register_region
from thalia.regions.cortex.layered_cortex import LayeredCortex
from thalia.regions.cortex.predictive_coding import (
    PredictiveCodingConfig,
    PredictiveCodingLayer,
)


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


@register_region(
    "predictive_cortex",
    aliases=["predictive"],
    description="Layered cortex with predictive coding and precision-weighted prediction errors",
    version="1.0",
    author="Thalia Project",
    config_class=PredictiveCortexConfig,
)
class PredictiveCortex(NeuralRegion):
    """
    Layered cortex with integrated predictive coding.

    This combines:
    1. LayeredCortex: Biologically realistic L4 → L2/3 → L5+L6 microcircuit
    2. PredictiveCoding: L5+L6 generate predictions, L2/3 computes errors
    3. SpikingAttention: Precision weighting modulated by attention

    The result is a cortical module that:
    - Learns WITHOUT backpropagation (local prediction errors)
    - Naturally implements attention (precision weighting)
    - Maintains biological plausibility

    Credit Assignment Solution:
    ===========================
    Instead of backprop, learning happens via:
    1. L5+L6 predict expected L4 input (deep layers are prediction neurons)
    2. Error = L4_actual - L5+L6_prediction
    3. Error propagates to L2/3 (goes UP to next area)
    4. L5+L6 weights update based on local error

    All learning is LOCAL to each layer!

    Mixins Provide:
    ---------------
    From DiagnosticsMixin (via NeuralRegion):
        - check_health() → HealthMetrics
        - get_firing_rate(spikes) → float
        - check_weight_health(weights, name) → WeightHealth
        - detect_runaway_excitation(spikes) → bool

    From NeuralRegion:
        - forward(inputs: Dict[str, Tensor], **kwargs) → Tensor [delegates to cortex]
        - reset_state() → None [delegates to cortex]
        - get_diagnostics() → Dict

    Note: PredictiveCortex uses composition (has-a LayeredCortex)
          rather than inheritance, so many methods delegate.

    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
        docs/patterns/state-management.md for composition pattern

    Usage:
        config = PredictiveCortexConfig(
            n_input=256,
            n_output=128,
            prediction_enabled=True,
            use_attention=True,
        )
        cortex = PredictiveCortex(config)

        # Process input
        for t in range(n_timesteps):
            output = cortex(input_spikes)

        # Learning (after episode)
        metrics = cortex.learn(input_spikes, output_spikes)
    """

    def __init__(
        self,
        config: PredictiveCortexConfig,
        sizes: Optional[Dict[str, int]] = None,
        device: str = "cpu",
    ):
        """Initialize predictive cortex.

        Args:
            config: Behavioral configuration (learning rates, sparsity, etc.) - NO sizes
            sizes: Structural sizes dict with keys: l4_size, l23_size, l5_size, l6a_size, l6b_size, input_size
            device: Device for tensors ("cpu", "cuda", etc.)
        """
        self.config = config

        # Extract sizes from dict (required)
        if sizes is None:
            raise ValueError(
                "PredictiveCortex requires sizes dict (l4_size, l23_size, l5_size, l6a_size, l6b_size, input_size)"
            )

        l4_size = sizes["l4_size"]
        l23_size = sizes["l23_size"]
        l5_size = sizes["l5_size"]
        l6a_size = sizes["l6a_size"]
        l6b_size = sizes["l6b_size"]

        l6_total = l6a_size + l6b_size  # Total L6 size

        # Compute output size
        _output_size = l23_size + l5_size

        # Call NeuralRegion init (v3.0 architecture)
        # PredictiveCortex uses composition - actual neurons managed by inner LayeredCortex
        super().__init__(
            n_neurons=_output_size,
            neuron_config=None,  # Managed by inner cortex
            default_learning_rule=None,  # Managed by inner cortex
            device=device,
        )

        # =====================================================================
        # BASE LAYERED CORTEX (creates the L4→L2/3→L5 microcircuit)
        # =====================================================================
        self.cortex = LayeredCortex(config=config, sizes=sizes, device=device)

        # NOTE: LayeredCortex uses multi-source architecture - weights stored in synaptic_weights dict
        # No single "weights" attribute - each source has independent weights

        # Get layer sizes from cortex (verify our calculations)
        self.l4_size = self.cortex.l4_size
        self.l23_size = self.cortex.l23_size
        self.l5_size = self.cortex.l5_size
        self.l6a_size = self.cortex.l6a_size  # L6a → TRN pathway
        self.l6b_size = self.cortex.l6b_size  # L6b → thalamic relay
        self.l6_size = self.l6a_size + self.l6b_size  # Total L6 for prediction neurons
        self._output_size = _output_size  # Track output size for growth
        assert self.l4_size == l4_size, f"L4 size mismatch: {self.l4_size} != {l4_size}"
        assert self.l23_size == l23_size, f"L2/3 size mismatch: {self.l23_size} != {l23_size}"
        assert self.l5_size == l5_size, f"L5 size mismatch: {self.l5_size} != {l5_size}"
        assert self.l6a_size == l6a_size, f"L6a size mismatch: {self.l6a_size} != {l6a_size}"
        assert self.l6b_size == l6b_size, f"L6b size mismatch: {self.l6b_size} != {l6b_size}"
        assert self.l6_size == l6_total, f"L6 total mismatch: {self.l6_size} != {l6_total}"

        # =====================================================================
        # PREDICTIVE CODING MODULE
        # =====================================================================
        self.prediction_layer: Optional[PredictiveCodingLayer] = None

        if config.prediction_enabled:
            # L5+L6 → L4 prediction pathway
            # Both deep layers (L5 and L6) are prediction neurons:
            # - L5: Projects to subcortical targets and other cortex
            # - L6: Projects to thalamus TRN (modulates sensory input)
            # Together they form the "prediction representation"
            prediction_repr_size = self.l5_size + self.l6_size
            self.prediction_layer = PredictiveCodingLayer(
                PredictiveCodingConfig(
                    n_input=self.l4_size,  # Predicts L4 input
                    n_representation=prediction_repr_size,  # From L5+L6 combined
                    prediction_tau_ms=config.prediction_tau_ms,
                    error_tau_ms=config.error_tau_ms,
                    learning_rate=config.prediction_learning_rate,
                    initial_precision=config.initial_precision,
                    precision_learning_rate=config.precision_learning_rate,
                    device=device,  # Use device argument, not config.device string
                )
            )

        # =====================================================================
        # GAMMA ATTENTION (inherited from LayeredCortex base)
        # =====================================================================
        # NOTE: Gamma attention is now implemented in LayeredCortex base class.
        # Both cortex types (layered and predictive) get gamma-phase gating
        # automatically through the self.cortex composition.
        # No need to duplicate initialization here.

        # =====================================================================
        # PRECISION MODULATION (attention → prediction)
        # =====================================================================
        if config.use_precision_weighting:
            # Maps attention output to precision modulation
            self.precision_modulator: Optional[nn.Linear] = nn.Linear(self.l23_size, self.l4_size)
            nn.init.zeros_(self.precision_modulator.weight)
            nn.init.ones_(self.precision_modulator.bias)
        else:
            self.precision_modulator = None  # type: ignore[assignment]

        # =====================================================================
        # STATE
        # =====================================================================
        self.state: PredictiveCortexState = PredictiveCortexState()  # type: ignore[assignment]

        # Metrics for monitoring
        self._total_free_energy = 0.0
        self._timesteps = 0
        self._last_plasticity_delta = 0.0

        # Cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0

    def reset_state(self) -> None:
        """Reset all states for new sequence."""
        self.cortex.reset_state()

        if self.prediction_layer is not None:
            self.prediction_layer.reset_state()

        # Note: Gamma attention reset handled by base cortex.reset_state()

        # Sync state from inner cortex (don't leave as None!)
        # The inner LayeredCortex initializes proper zero tensors
        self.state = PredictiveCortexState(
            l4_spikes=self.cortex.state.l4_spikes,
            l23_spikes=self.cortex.state.l23_spikes,
            l5_spikes=self.cortex.state.l5_spikes,
            l6a_spikes=self.cortex.state.l6a_spikes,
            l6b_spikes=self.cortex.state.l6b_spikes,
        )
        self._total_free_energy = 0.0
        self._timesteps = 0
        self._last_plasticity_delta = 0.0

        # Reset cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0
        self._cumulative_l6_spikes = 0

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters for new timestep.

        Args:
            dt_ms: New timestep in milliseconds
        """
        # Delegate to inner cortex (which has STP, neurons, strategies)
        if hasattr(self.cortex, "update_temporal_parameters"):
            self.cortex.update_temporal_parameters(dt_ms)

        # Update prediction layer if present
        if self.prediction_layer is not None:
            if hasattr(self.prediction_layer, "update_temporal_parameters"):
                self.prediction_layer.update_temporal_parameters(dt_ms)

    def get_state(self) -> PredictiveCortexState:
        """Get current predictive cortex state.

        Returns state with all layer spikes synchronized from inner cortex.

        Returns:
            PredictiveCortexState with current layer activities
        """
        # Sync from inner cortex to ensure we have latest state
        return PredictiveCortexState(
            # Base neuromodulator levels (from inner cortex)
            dopamine=self.cortex.state.dopamine,
            acetylcholine=self.cortex.state.acetylcholine,
            norepinephrine=self.cortex.state.norepinephrine,
            # Layer spikes
            l4_spikes=self.cortex.state.l4_spikes,
            l23_spikes=self.cortex.state.l23_spikes,
            l5_spikes=self.cortex.state.l5_spikes,
            l6a_spikes=self.cortex.state.l6a_spikes,
            l6b_spikes=self.cortex.state.l6b_spikes,
            # Input and predictive coding state
            source_inputs=(
                self.state.source_inputs if hasattr(self.state, "source_inputs") else None
            ),
        )

    def set_neuromodulators(
        self,
        dopamine: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None,
    ) -> None:
        """Set neuromodulator levels.

        Propagates to both PredictiveCortex state and inner cortex state
        to ensure consistency.

        Args:
            dopamine: Dopamine level (reward signal)
            norepinephrine: Norepinephrine level (arousal)
            acetylcholine: Acetylcholine level (attention)
        """
        # Update own state via parent mixin
        super().set_neuromodulators(
            dopamine=dopamine,
            norepinephrine=norepinephrine,
            acetylcholine=acetylcholine,
        )

        # Propagate to inner cortex
        self.cortex.set_neuromodulators(
            dopamine=dopamine,
            norepinephrine=norepinephrine,
            acetylcholine=acetylcholine,
        )

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set oscillator phases and pass through to inner LayeredCortex.

        PredictiveCortex delegates oscillator handling to its inner LayeredCortex,
        which implements alpha-based attention gating. This ensures that alpha
        suppression works correctly in predictive mode.

        Args:
            phases: Dict mapping oscillator name to phase [0, 2π)
            signals: Dict mapping oscillator name to signal [-1, 1]
            theta_slot: Current theta slot [0, n_slots-1] for sequence encoding
            coupled_amplitudes: Dict of cross-frequency coupled amplitudes
        """
        # Use base mixin implementation to store all oscillator data
        super().set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

        # Pass through to inner cortex (where alpha gating is implemented)
        self.cortex.set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow output dimension by delegating to base LayeredCortex.

        This expands all layers (L4, L2/3, L5, L6) proportionally while also
        updating the prediction layer (which uses L5+L6 combined) and attention modules.

        Args:
            n_new: Number of neurons to add to total cortex size
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
        # =====================================================================
        # 1. DELEGATE TO LAYERED CORTEX (handles all weight/neuron expansion)
        # =====================================================================
        self.cortex.grow_output(n_new, initialization, sparsity)

        # =====================================================================
        # 2. UPDATE SIZES AND CONFIG
        # =====================================================================
        # Update our cached layer sizes
        self.l4_size = self.cortex.l4_size
        self.l23_size = self.cortex.l23_size
        self.l5_size = self.cortex.l5_size
        self.l6a_size = self.cortex.l6a_size
        self.l6b_size = self.cortex.l6b_size
        self.l6_size = self.l6a_size + self.l6b_size

        # Update output size (note: _output_size initialized in __init__)
        self._output_size = self.l23_size + self.l5_size
        self.n_output = self._output_size  # Update NeuralRegion's n_output attribute
        self.n_neurons = self._output_size  # Keep n_neurons in sync

        # Note: Config no longer stores sizes - they're instance variables only

        # =====================================================================
        # 3. RECREATE PREDICTION LAYER with new sizes
        # =====================================================================
        if self.config.prediction_enabled:
            # Use L5+L6 combined as prediction representation (biologically accurate)
            prediction_repr_size = self.l5_size + self.l6_size
            self.prediction_layer = PredictiveCodingLayer(
                PredictiveCodingConfig(
                    n_input=self.l4_size,
                    n_representation=prediction_repr_size,
                    prediction_tau_ms=self.config.prediction_tau_ms,
                    error_tau_ms=self.config.error_tau_ms,
                    learning_rate=self.config.prediction_learning_rate,
                    initial_precision=self.config.initial_precision,
                    precision_learning_rate=self.config.precision_learning_rate,
                    device=str(self.device),
                )
            )

        # NOTE: Gamma attention resizing handled by base LayeredCortex.grow_output()

    def forward(  # type: ignore[override]
        self,
        inputs: Dict[str, torch.Tensor],
        top_down: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process input through predictive cortex (ADR-005: 1D tensors).

        Args:
            inputs: Multi-source input spikes - dict mapping source names to spike tensors [n_input] (1D)
            top_down: Optional top-down modulation from higher areas [l23_size] (1D)
                      NOTE: This is for L2/3 modulation, NOT L4 prediction!
            **kwargs: Additional arguments for compatibility

        Returns:
            output: Output spikes (L2/3 + L5) [l23_size + l5_size] (1D)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # Validate inputs is a dict (required for multi-source architecture)
        if not isinstance(inputs, dict):
            raise TypeError(
                f"PredictiveCortex.forward: inputs must be a Dict[str, Tensor], "
                f'got {type(inputs)}. Wrap single tensor in dict like {{"source_name": tensor}}'
            )

        # Validate all input tensors are 1D (ADR-005)
        for source_name, source_spikes in inputs.items():
            assert (
                source_spikes.dim() == 1
            ), f"PredictiveCortex.forward: Expected 1D input for source '{source_name}' (ADR-005), got shape {source_spikes.shape}"

        if top_down is not None:
            assert (
                top_down.dim() == 1
            ), f"PredictiveCortex.forward: Expected 1D top_down (ADR-005), got shape {top_down.shape}"
            assert top_down.shape[0] == self.l23_size, (
                f"PredictiveCortex.forward: top_down has shape {top_down.shape} "
                f"but must match l23_size={self.l23_size}. "
                f"top_down is for L2/3 modulation, not L4 prediction."
            )

        # Initialize if needed
        state = self.state  # type: PredictiveCortexState
        if state.l4_spikes is None:
            self.reset_state()

        # =====================================================================
        # STEP 1: Standard feedforward through cortex
        # =====================================================================
        # Pass through to the base LayeredCortex (computes theta modulation internally)
        # Multi-source inputs handled natively by LayeredCortex
        cortex_output = self.cortex.forward(
            inputs,
            top_down=top_down,
        )

        # Extract layer outputs from cortex state
        l4_output = self.cortex.state.l4_spikes
        l23_output = self.cortex.state.l23_spikes
        l5_output = self.cortex.state.l5_spikes
        # L6 is split into L6a and L6b - combine for prediction representation
        l6a_output = self.cortex.state.l6a_spikes
        l6b_output = self.cortex.state.l6b_spikes
        l6_output = self.cortex.get_l6_spikes()  # Combined L6a + L6b

        # Store in state (including original inputs dict for learning)
        state.source_inputs = inputs
        state.l4_spikes = l4_output
        state.l23_spikes = l23_output
        state.l5_spikes = l5_output
        state.l6a_spikes = l6a_output
        state.l6b_spikes = l6b_output

        # Update cumulative spike counters (for diagnostics)
        if l4_output is not None:
            self._cumulative_l4_spikes += int(l4_output.sum().item())
        if l23_output is not None:
            self._cumulative_l23_spikes += int(l23_output.sum().item())
        if l5_output is not None:
            self._cumulative_l5_spikes += int(l5_output.sum().item())
        if l6_output is not None:
            self._cumulative_l6_spikes += int(l6_output.sum().item())

        # =====================================================================
        # STEP 2: Predictive coding (L5+L6 → L4 prediction, compute error)
        # =====================================================================
        if self.prediction_layer is not None and l5_output is not None and l6_output is not None:
            # L5+L6 together generate prediction of what L4 should receive
            # L5: Cortical predictions (to other areas, subcortical)
            # L6: Thalamic predictions (via TRN, modulates sensory gating)
            # Both deep layers participate in predictive coding
            # NOTE: top_down is for L2/3 modulation, NOT for L4 prediction
            # Convert bool spikes to float for prediction layer (ADR-004)
            # Check for None before calling float()
            l5_float = (
                l5_output.float()
                if l5_output is not None
                else torch.zeros(self.l5_size, device=self.device)
            )
            l6_float = (
                l6_output.float()
                if l6_output is not None
                else torch.zeros(self.l6_size, device=self.device)
            )
            combined_representation = torch.cat(
                [
                    l5_float,
                    l6_float,
                ],
                dim=-1,
            )
            error, prediction, _ = self.prediction_layer(
                actual_input=(
                    l4_output.float()
                    if l4_output is not None
                    else torch.zeros(self.l4_size, device=self.device)
                ),
                representation=combined_representation,
                top_down_prediction=None,  # L5+L6→L4 prediction is generated internally
            )

            state.prediction = prediction
            state.error = error
            state.precision = self.prediction_layer.precision
            free_energy_tensor = self.prediction_layer.get_free_energy()
            state.free_energy = free_energy_tensor.item() if free_energy_tensor is not None else 0.0

            # Accumulate free energy for monitoring
            self._total_free_energy += state.free_energy
            self._timesteps += 1

        # =====================================================================
        # STEP 3: Gamma attention already applied by base LayeredCortex
        # =====================================================================
        # NOTE: Inner cortex applies gamma-phase gating to L2/3 if enabled
        # We can access the gating weights from cortex.state.gamma_attention_gate

        # =====================================================================
        # STEP 4: Precision modulation (gamma-gate → prediction weights)
        # =====================================================================
        if (
            self.precision_modulator is not None
            and hasattr(self.cortex.state, "gamma_attention_gate")
            and self.cortex.state.gamma_attention_gate is not None
        ):
            # Use gamma-phase gating from base cortex to modulate prediction trust
            avg_gate = float(self.cortex.state.gamma_attention_gate.mean().item())

            # Modulate precision based on average gating strength
            if self.prediction_layer is not None:
                precision_scale = avg_gate
                self.prediction_layer.log_precision.data = (
                    self.prediction_layer.log_precision.data
                    + 0.01
                    * torch.log(
                        torch.tensor(
                            precision_scale + 1e-6,
                            device=self.prediction_layer.log_precision.device,
                        )
                    )
                )

        # =====================================================================
        # COMBINE OUTPUT
        # =====================================================================
        if l23_output is not None and l5_output is not None:
            output = torch.cat([l23_output, l5_output], dim=-1)
        elif l23_output is not None:
            output = l23_output
        elif l5_output is not None:
            output = l5_output
        else:
            output = cortex_output

        # =====================================================================
        # CONTINUOUS LEARNING (plasticity happens as part of forward dynamics)
        # =====================================================================
        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # The underlying LayeredCortex already does continuous STDP in its forward()
        # Here we also update the prediction weights based on accumulated error
        if self.prediction_layer is not None:
            # Get dopamine-modulated learning rate from base class
            effective_lr = self.get_effective_learning_rate(self.config.prediction_learning_rate)
            if effective_lr > 1e-8:  # Only learn if not fully suppressed
                # Learn prediction weights based on current error
                # Pass None for reward_signal since we're doing continuous learning
                # Reward will modulate dopamine, which is handled via modulation
                pred_metrics = self.prediction_layer.learn(reward_signal=None)
                # Store plasticity delta on the prediction layer state
                self._last_plasticity_delta = pred_metrics.get("weight_update", 0.0)

        return output

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get layer-specific diagnostics using DiagnosticsMixin helpers.

        Uses the same format as LayeredCortex for consistency.

        Note: Reports both instantaneous (l4_active_count) and cumulative
        (l4_cumulative_spikes) counts. During consolidation phases with
        zero input, instantaneous L4 will be 0 but cumulative shows
        total activity since last reset.
        """
        diag: Dict[str, Any] = {
            "l4_size": self.l4_size,
            "l23_size": self.l23_size,
            "l5_size": self.l5_size,
            "l6_size": self.l6_size,
            "last_plasticity_delta": getattr(self, "_last_plasticity_delta", 0.0),
            # Cumulative spike counts (since last reset_state)
            "l4_cumulative_spikes": getattr(self, "_cumulative_l4_spikes", 0),
            "l23_cumulative_spikes": getattr(self, "_cumulative_l23_spikes", 0),
            "l5_cumulative_spikes": getattr(self, "_cumulative_l5_spikes", 0),
            "l6_cumulative_spikes": getattr(self, "_cumulative_l6_spikes", 0),
        }

        # Spike diagnostics for each layer (same format as LayeredCortex)
        # These are INSTANTANEOUS counts from the last forward pass
        state = self.state  # type: PredictiveCortexState
        if state.l4_spikes is not None:
            diag.update(self.spike_diagnostics(state.l4_spikes, "l4"))
        if state.l23_spikes is not None:
            diag.update(self.spike_diagnostics(state.l23_spikes, "l23"))
        if state.l5_spikes is not None:
            diag.update(self.spike_diagnostics(state.l5_spikes, "l5"))

        # Weight diagnostics from prediction layer (if available)
        if self.prediction_layer is not None:
            pred_diag = self.prediction_layer.get_diagnostics()
            diag.update({f"pred_{k}": v for k, v in pred_diag.items()})
            # Add weight diagnostics for consistency with LayeredCortex
            if hasattr(self.prediction_layer, "W_pred"):
                diag.update(self.weight_diagnostics(self.prediction_layer.W_pred.data, "pred"))
            if hasattr(self.prediction_layer, "W_encode"):
                diag.update(self.weight_diagnostics(self.prediction_layer.W_encode.data, "encode"))

        return diag

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Extends LayeredCortex state with prediction layer and attention state.

        Returns state dictionary with keys:
        - weights: All weights (inherited from LayeredCortex)
        - region_state: Neuron states, spikes, traces (inherited)
        - learning_state: BCM, STP (inherited)
        - prediction_state: Prediction layer weights and state
        - attention_state: Attention mechanism state
        - neuromodulator_state: Current neuromodulators
        - config: Configuration for validation
        """
        # Get base LayeredCortex state (from wrapped cortex, not super())
        state_dict = self.cortex.get_full_state()

        # Add prediction layer state
        if self.prediction_layer is not None:
            pred_state = self.prediction_layer.get_state()
            state_dict["prediction_state"] = {
                "W_pred": (
                    pred_state["W_pred"].clone()
                    if "W_pred" in pred_state and pred_state["W_pred"] is not None
                    else None
                ),
                "W_encode": (
                    pred_state["W_encode"].clone()
                    if "W_encode" in pred_state and pred_state["W_encode"] is not None
                    else None
                ),
                "log_precision": (
                    pred_state["log_precision"].clone()
                    if "log_precision" in pred_state and pred_state["log_precision"] is not None
                    else None
                ),
                "prediction": (
                    pred_state["prediction"].clone()
                    if "prediction" in pred_state and pred_state["prediction"] is not None
                    else None
                ),
                "error": (
                    pred_state["error"].clone()
                    if "error" in pred_state and pred_state["error"] is not None
                    else None
                ),
            }
        else:
            state_dict["prediction_state"] = None

        # NOTE: Gamma attention state saved by base LayeredCortex

        # Add config dict with PredictiveCortex-specific parameters
        if "config" not in state_dict:
            state_dict["config"] = {}
        state_dict["config"]["prediction_enabled"] = self.config.prediction_enabled

        # Add format identifier for hybrid checkpoints
        state_dict["format"] = "elastic_tensor"

        return state_dict

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dictionary from get_full_state()

        Raises:
            ValueError: If config dimensions don't match
        """
        # Load base LayeredCortex state (PredictiveCortex wraps LayeredCortex)
        self.cortex.load_full_state(state)

        # Load prediction layer state
        if "prediction_state" in state and state["prediction_state"] is not None:
            if self.prediction_layer is not None:
                pred_state = state["prediction_state"]
                self.prediction_layer.load_state(
                    {
                        "W_pred": (
                            pred_state["W_pred"].to(self.device)
                            if pred_state["W_pred"] is not None
                            else None
                        ),
                        "W_encode": (
                            pred_state["W_encode"].to(self.device)
                            if pred_state["W_encode"] is not None
                            else None
                        ),
                        "log_precision": (
                            pred_state["log_precision"].to(self.device)
                            if pred_state["log_precision"] is not None
                            else None
                        ),
                        "prediction": (
                            pred_state["prediction"].to(self.device)
                            if pred_state["prediction"] is not None
                            else None
                        ),
                        "error": (
                            pred_state["error"].to(self.device)
                            if pred_state["error"] is not None
                            else None
                        ),
                    }
                )

        # NOTE: Gamma attention state loaded by base LayeredCortex

    def get_l6_spikes(self) -> Optional[torch.Tensor]:
        """Get L6 corticothalamic feedback spikes.

        L6 is part of the prediction neurons in PredictiveCortex,
        but also serves the biological function of thalamic feedback
        via TRN for attentional modulation.

        Returns:
            L6 spikes [l6_size] or None if not available
        """
        return self.cortex.get_l6_spikes()

    def get_l6_feedback(self) -> Optional[torch.Tensor]:
        """Alias for get_l6_spikes for compatibility with dynamic_brain port extraction.

        Returns:
            L6 feedback spikes [l6_size] or None if not available
        """
        return self.cortex.get_l6_spikes()

    def get_effective_learning_rate(
        self,
        base_lr: Optional[float] = None,
        dopamine_sensitivity: float = 1.0,
    ) -> float:
        """Delegate to inner cortex for dopamine-modulated learning rate.

        Args:
            base_lr: Base learning rate
            dopamine_sensitivity: How much dopamine affects learning

        Returns:
            Effective learning rate modulated by dopamine
        """
        return self.cortex.get_effective_learning_rate(base_lr, dopamine_sensitivity)

    @property
    def output_size(self) -> int:
        """Total output size (L2/3 + L5)."""
        return self.l23_size + self.l5_size
