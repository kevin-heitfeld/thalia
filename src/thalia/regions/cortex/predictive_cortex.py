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
Predictive cortex: L5 → L4 (predictions) + L4 vs L5 → L2/3 (errors)

Learning Rule:
- Error = Input - Prediction
- ΔW_prediction ∝ Error × Representation
- This is LOCAL - no need to propagate gradients!

Integration with Attention:
- Precision = inverse error variance
- High precision → attend to this input
- Attention modulates precision (PFC → precision weights)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from thalia.core.component_config import NeuralComponentConfig
from thalia.core.component_registry import register_region
from thalia.regions.base import NeuralComponent, NeuralComponentState, LearningRule
from thalia.regions.cortex.layered_cortex import LayeredCortex, LayeredCortexConfig
from thalia.core.predictive_coding import (
    PredictiveCodingLayer,
    PredictiveCodingConfig,
)


@dataclass
class PredictiveCortexConfig(LayeredCortexConfig):
    """Configuration for predictive cortex.

    Extends LayeredCortexConfig with predictive coding parameters.
    """
    # Predictive coding parameters
    prediction_enabled: bool = True
    prediction_tau_ms: float = 50.0    # Slow predictions (NMDA-like)
    error_tau_ms: float = 5.0          # Fast errors (AMPA-like)
    prediction_learning_rate: float = 0.01

    # Precision (attention) parameters
    use_precision_weighting: bool = True
    initial_precision: float = 1.0
    precision_learning_rate: float = 0.001

    # Note: Gamma attention inherited from LayeredCortex base class (always enabled)
    # Configure width via gamma_attention_width


@dataclass
class PredictiveCortexState(NeuralComponentState):
    """State for predictive cortex.

    Extends NeuralComponentState with cortex-specific fields for layer spikes,
    predictive coding, and attention.
    """
    # Layer-specific spikes (cortex has L4→L2/3→L5 microcircuit)
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None

    # Store original input for learning
    input_spikes: Optional[torch.Tensor] = None

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


@register_region(
    "predictive_cortex",
    description="Layered cortex with predictive coding and precision-weighted prediction errors",
    version="1.0",
    author="Thalia Project"
)
class PredictiveCortex(NeuralComponent):
    """
    Layered cortex with integrated predictive coding.

    This combines:
    1. LayeredCortex: Biologically realistic L4 → L2/3 → L5 microcircuit
    2. PredictiveCoding: L5 generates predictions, L2/3 computes errors
    3. SpikingAttention: Precision weighting modulated by attention

    The result is a cortical module that:
    - Learns WITHOUT backpropagation (local prediction errors)
    - Naturally implements attention (precision weighting)
    - Maintains biological plausibility

    Credit Assignment Solution:
    ===========================
    Instead of backprop, learning happens via:
    1. L5 predicts expected L4 input
    2. Error = L4_actual - L5_prediction
    3. Error propagates to L2/3 (goes UP to next area)
    4. L5 weights update based on local error

    All learning is LOCAL to each layer!

    Mixins Provide:
    ---------------
    From DiagnosticsMixin:
        - check_health() → HealthMetrics
        - get_firing_rate(spikes) → float
        - check_weight_health(weights, name) → WeightHealth
        - detect_runaway_excitation(spikes) → bool

    From NeuralComponent (abstract base):
        - forward(input, **kwargs) → Tensor [delegates to cortex]
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

    def __init__(self, config: PredictiveCortexConfig):
        """Initialize predictive cortex."""
        self.predictive_config = config

        # Calculate layer sizes from config
        from thalia.regions.cortex.config import calculate_layer_sizes
        l4_size, l23_size, l5_size = calculate_layer_sizes(
            config.n_output, config.l4_ratio, config.l23_ratio, config.l5_ratio
        )
        _output_size = l23_size + l5_size

        # Create parent config for NeuralComponent
        parent_config = NeuralComponentConfig(
            n_input=config.n_input,
            n_output=_output_size,
            dt_ms=config.dt_ms,
            device=config.device,
        )

        # Call parent init (our _initialize_weights returns None for lazy init)
        super().__init__(parent_config)

        # =====================================================================
        # BASE LAYERED CORTEX (creates the L4→L2/3→L5 microcircuit)
        # =====================================================================
        self.cortex = LayeredCortex(config)

        # Set our weights to delegate to cortex
        # Note: LayeredCortex manages its own neurons internally (l4_neurons, l23_neurons, l5_neurons)
        # so we don't expose a top-level neurons attribute
        self.weights = self.cortex.weights

        # Get layer sizes from cortex (verify our calculations)
        self.l4_size = self.cortex.l4_size
        self.l23_size = self.cortex.l23_size
        self.l5_size = self.cortex.l5_size
        self._output_size = _output_size  # Track output size for growth
        assert self.l4_size == l4_size, f"L4 size mismatch: {self.l4_size} != {l4_size}"
        assert self.l23_size == l23_size, f"L2/3 size mismatch: {self.l23_size} != {l23_size}"
        assert self.l5_size == l5_size, f"L5 size mismatch: {self.l5_size} != {l5_size}"

        # =====================================================================
        # PREDICTIVE CODING MODULE
        # =====================================================================
        if config.prediction_enabled:
            # L5 → L4 prediction pathway
            self.prediction_layer = PredictiveCodingLayer(
                PredictiveCodingConfig(
                    n_input=self.l4_size,              # Predicts L4 input
                    n_representation=self.l5_size,      # From L5 representation
                    n_output=self.l4_size,
                    prediction_tau_ms=config.prediction_tau_ms,
                    error_tau_ms=config.error_tau_ms,
                    learning_rate=config.prediction_learning_rate,
                    initial_precision=config.initial_precision,
                    precision_learning_rate=config.precision_learning_rate,
                    use_spiking=True,
                    device=config.device,
                )
            )
        else:
            self.prediction_layer = None

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
            self.precision_modulator = nn.Linear(self.l23_size, self.l4_size)
            nn.init.zeros_(self.precision_modulator.weight)
            nn.init.ones_(self.precision_modulator.bias)
        else:
            self.precision_modulator = None

        # =====================================================================
        # STATE
        # =====================================================================
        self.state = PredictiveCortexState()

        # Metrics for monitoring
        self._total_free_energy = 0.0
        self._timesteps = 0
        self._last_plasticity_delta = 0.0

        # Cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from NeuralComponent)
    # =========================================================================

    def _get_learning_rule(self) -> LearningRule:
        """Predictive cortex uses predictive STDP (local errors + spike timing)."""
        return LearningRule.PREDICTIVE_STDP

    def _initialize_weights(self) -> Optional[torch.Tensor]:
        """Weights are managed by internal LayeredCortex (initialized after super().__init__).

        Returns None to signal lazy initialization - actual weights are set
        in __init__ after creating self.cortex.
        """
        return None

    def _create_neurons(self) -> Optional[Any]:
        """Neurons are managed by internal LayeredCortex (initialized after super().__init__).

        Returns None to signal lazy initialization - actual neurons are set
        in __init__ after creating self.cortex.
        """
        return None

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
        )
        self._total_free_energy = 0.0
        self._timesteps = 0
        self._last_plasticity_delta = 0.0

        # Reset cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0

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
        # Pass through to inner cortex (where alpha gating is implemented)
        self.cortex.set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

        # Also store in our own state for potential use
        # (though we delegate processing to inner cortex)
        if not hasattr(self.state, '_oscillator_phases'):
            self.state._oscillator_phases = {}
            self.state._oscillator_signals = {}
        self.state._oscillator_phases = phases
        self.state._oscillator_signals = signals

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to predictive cortex by delegating to internal LayeredCortex.

        This expands all layers (L4, L2/3, L5) proportionally while also
        updating the prediction layer and attention modules.

        Args:
            n_new: Number of neurons to add to total cortex size
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
        from dataclasses import replace

        # =====================================================================
        # 1. DELEGATE TO LAYERED CORTEX (handles all weight/neuron expansion)
        # =====================================================================
        self.cortex.add_neurons(n_new, initialization, sparsity)

        # =====================================================================
        # 2. UPDATE SIZES AND CONFIG
        # =====================================================================
        # Update our cached layer sizes
        self.l4_size = self.cortex.l4_size
        self.l23_size = self.cortex.l23_size
        self.l5_size = self.cortex.l5_size

        # Update output size (note: _output_size initialized in __init__)
        self._output_size = self.l23_size + self.l5_size

        # Update parent config
        self.config = replace(
            self.config,
            n_output=self._output_size
        )

        # =====================================================================
        # 3. RECREATE PREDICTION LAYER with new sizes
        # =====================================================================
        if self.predictive_config.prediction_enabled:
            self.prediction_layer = PredictiveCodingLayer(
                PredictiveCodingConfig(
                    n_input=self.l4_size,
                    n_representation=self.l5_size,
                    n_output=self.l4_size,
                    prediction_tau_ms=self.predictive_config.prediction_tau_ms,
                    error_tau_ms=self.predictive_config.error_tau_ms,
                    learning_rate=self.predictive_config.prediction_learning_rate,
                    initial_precision=self.predictive_config.initial_precision,
                    precision_learning_rate=self.predictive_config.precision_learning_rate,
                    use_spiking=True,
                    device=self.predictive_config.device,
                )
            )

        # Note: Gamma attention resizing handled by base LayeredCortex.add_neurons()

    def forward(
        self,
        input_spikes: torch.Tensor,
        top_down: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process input through predictive cortex (ADR-005: 1D tensors).

        Args:
            input_spikes: Input spike pattern [n_input] (1D)
            top_down: Optional top-down modulation from higher areas [l23_size] (1D)
                      NOTE: This is for L2/3 modulation, NOT L4 prediction!
            **kwargs: Additional arguments for compatibility

        Returns:
            output: Output spikes (L2/3 + L5) [l23_size + l5_size] (1D)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # ADR-005: Expect 1D tensors
        assert input_spikes.dim() == 1, (
            f"PredictiveCortex.forward: Expected 1D input (ADR-005), got shape {input_spikes.shape}"
        )
        assert input_spikes.shape[0] == self.predictive_config.n_input, (
            f"PredictiveCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.predictive_config.n_input}."
        )
        if top_down is not None:
            assert top_down.dim() == 1, (
                f"PredictiveCortex.forward: Expected 1D top_down (ADR-005), got shape {top_down.shape}"
            )
            assert top_down.shape[0] == self.l23_size, (
                f"PredictiveCortex.forward: top_down has shape {top_down.shape} "
                f"but must match l23_size={self.l23_size}. "
                f"top_down is for L2/3 modulation, not L4 prediction."
            )

        # Initialize if needed
        if self.state.l4_spikes is None:
            self.reset_state()

        # =====================================================================
        # STEP 1: Standard feedforward through cortex
        # =====================================================================
        # Pass through to the base LayeredCortex (computes theta modulation internally)
        cortex_output = self.cortex.forward(
            input_spikes,
            top_down=top_down,
        )

        # Extract layer outputs from cortex state
        l4_output = self.cortex.state.l4_spikes
        l23_output = self.cortex.state.l23_spikes
        l5_output = self.cortex.state.l5_spikes

        # Store in state (including original input for learning)
        self.state.input_spikes = input_spikes
        self.state.l4_spikes = l4_output
        self.state.l23_spikes = l23_output
        self.state.l5_spikes = l5_output

        # Update cumulative spike counters (for diagnostics)
        if l4_output is not None:
            self._cumulative_l4_spikes += int(l4_output.sum().item())
        if l23_output is not None:
            self._cumulative_l23_spikes += int(l23_output.sum().item())
        if l5_output is not None:
            self._cumulative_l5_spikes += int(l5_output.sum().item())

        # =====================================================================
        # STEP 2: Predictive coding (L5 → L4 prediction, compute error)
        # =====================================================================
        if self.prediction_layer is not None and l5_output is not None:
            # L5 generates prediction of what L4 should receive
            # NOTE: top_down is for L2/3 modulation, NOT for L4 prediction
            # The prediction_layer handles L5→L4 predictions internally
            # Convert bool spikes to float for prediction layer (ADR-004)
            error, prediction, _ = self.prediction_layer(
                actual_input=l4_output.float(),
                representation=l5_output.float(),
                top_down_prediction=None,  # L5→L4 prediction is generated internally
            )

            self.state.prediction = prediction
            self.state.error = error
            self.state.precision = self.prediction_layer.precision
            self.state.free_energy = self.prediction_layer.get_free_energy().item()

            # Accumulate free energy for monitoring
            self._total_free_energy += self.state.free_energy
            self._timesteps += 1

        # =====================================================================
        # STEP 3: Gamma attention already applied by base LayeredCortex
        # =====================================================================
        # Note: Inner cortex applies gamma-phase gating to L2/3 if enabled
        # We can access the gating weights from cortex.state.gamma_attention_gate

        # =====================================================================
        # STEP 4: Precision modulation (gamma-gate → prediction weights)
        # =====================================================================
        if (self.precision_modulator is not None and
            hasattr(self.cortex.state, 'gamma_attention_gate') and
            self.cortex.state.gamma_attention_gate is not None):
            # Use gamma-phase gating from base cortex to modulate prediction trust
            avg_gate = float(self.cortex.state.gamma_attention_gate.mean().item())

            # Modulate precision based on average gating strength
            if self.prediction_layer is not None:
                precision_scale = avg_gate
                with torch.no_grad():
                    self.prediction_layer.log_precision.data = (
                        self.prediction_layer.log_precision.data +
                        0.01 * torch.log(torch.tensor(precision_scale + 1e-6, device=self.prediction_layer.log_precision.device))
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
            effective_lr = self.get_effective_learning_rate(
                self.predictive_config.prediction_learning_rate
            )
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
            "last_plasticity_delta": getattr(self, "_last_plasticity_delta", 0.0),
            # Cumulative spike counts (since last reset_state)
            "l4_cumulative_spikes": getattr(self, "_cumulative_l4_spikes", 0),
            "l23_cumulative_spikes": getattr(self, "_cumulative_l23_spikes", 0),
            "l5_cumulative_spikes": getattr(self, "_cumulative_l5_spikes", 0),
        }

        # Spike diagnostics for each layer (same format as LayeredCortex)
        # These are INSTANTANEOUS counts from the last forward pass
        if self.state.l4_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l4_spikes, "l4"))
        if self.state.l23_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l23_spikes, "l23"))
        if self.state.l5_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l5_spikes, "l5"))

        # Weight diagnostics from prediction layer (if available)
        if self.prediction_layer is not None:
            pred_diag = self.prediction_layer.get_diagnostics()
            diag.update({f"pred_{k}": v for k, v in pred_diag.items()})
            # Add weight diagnostics for consistency with LayeredCortex
            if hasattr(self.prediction_layer, 'W_pred'):
                diag.update(self.weight_diagnostics(self.prediction_layer.W_pred.data, "pred"))
            if hasattr(self.prediction_layer, 'W_encode'):
                diag.update(self.weight_diagnostics(self.prediction_layer.W_encode.data, "encode"))

        # Gamma attention diagnostics (from base cortex)
        if hasattr(self.cortex, 'gamma_attention') and self.cortex.gamma_attention is not None:
            diag["gamma_attn_phase"] = self.cortex.state.gamma_attention_phase
            diag["gamma_attn_frequency_hz"] = self.cortex.gamma_attention.frequency_hz  # Direct property
            if self.cortex.state.gamma_attention_gate is not None:
                diag["gamma_attn_mean_gate"] = float(self.cortex.state.gamma_attention_gate.mean().item())
                diag["gamma_attn_max_gate"] = float(self.cortex.state.gamma_attention_gate.max().item())

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
                "W_pred": pred_state["W_pred"].clone() if "W_pred" in pred_state and pred_state["W_pred"] is not None else None,
                "W_encode": pred_state["W_encode"].clone() if "W_encode" in pred_state and pred_state["W_encode"] is not None else None,
                "log_precision": pred_state["log_precision"].clone() if "log_precision" in pred_state and pred_state["log_precision"] is not None else None,
                "prediction": pred_state["prediction"].clone() if "prediction" in pred_state and pred_state["prediction"] is not None else None,
                "error": pred_state["error"].clone() if "error" in pred_state and pred_state["error"] is not None else None,
            }
        else:
            state_dict["prediction_state"] = None

        # Note: Gamma attention state saved by base LayeredCortex

        # Update config with PredictiveCortex-specific parameters
        state_dict["config"]["prediction_enabled"] = self.predictive_config.prediction_enabled

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
                self.prediction_layer.load_state({
                    "W_pred": pred_state["W_pred"].to(self.device) if pred_state["W_pred"] is not None else None,
                    "W_encode": pred_state["W_encode"].to(self.device) if pred_state["W_encode"] is not None else None,
                    "log_precision": pred_state["log_precision"].to(self.device) if pred_state["log_precision"] is not None else None,
                    "prediction": pred_state["prediction"].to(self.device) if pred_state["prediction"] is not None else None,
                    "error": pred_state["error"].to(self.device) if pred_state["error"] is not None else None,
                })

        # Note: Gamma attention state loaded by base LayeredCortex

    @property
    def output_size(self) -> int:
        """Total output size (L2/3 + L5)."""
        return self.l23_size + self.l5_size


# =============================================================================
# MULTI-AREA PREDICTIVE HIERARCHY
# =============================================================================


class PredictiveHierarchy(nn.Module):
    """
    Multi-area cortical hierarchy with bidirectional predictions.

    Each area predicts the errors from the area below, creating
    increasingly abstract representations as you go up.

    This is the full implementation of hierarchical predictive coding:
    - V1 → V2 → V4 → IT (ventral stream analog)
    - Each area predicts the errors from below
    - Top-down predictions flow back down

    Information flow:
    =================

    Area 3 (most abstract)
        ↑ errors        ↓ predictions
    Area 2
        ↑ errors        ↓ predictions
    Area 1 (concrete)
        ↑ errors        ↓ predictions
    Sensory Input
    """

    def __init__(
        self,
        area_sizes: list[int],
        base_config: Optional[PredictiveCortexConfig] = None,
    ):
        super().__init__()

        n_areas = len(area_sizes) - 1

        if base_config is None:
            base_config = PredictiveCortexConfig(n_input=area_sizes[0], n_output=area_sizes[1])

        self.areas = nn.ModuleList()

        for i in range(n_areas):
            area_config = PredictiveCortexConfig(
                n_input=area_sizes[i],
                n_output=area_sizes[i + 1],
                prediction_enabled=True,
                # Gamma attention always enabled (inherited from LayeredCortexConfig)
                device=base_config.device,
            )
            self.areas.append(PredictiveCortex(area_config))

        self.n_areas = n_areas

    def reset_state(self) -> None:
        """Reset all areas."""
        for area in self.areas:
            area.reset_state()

    def forward(
        self,
        sensory_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[PredictiveCortexState]]:
        """
        Process through hierarchy.

        Args:
            sensory_input: Raw sensory input [batch, input_size]

        Returns:
            output: Highest-level output
            states: States from each area
        """
        states = []
        current = sensory_input

        # Bottom-up pass
        for area in self.areas:
            output, state = area(current)
            states.append(state)
            current = output

        return current, states

    def get_total_free_energy(self) -> float:
        """Sum of free energy across all areas."""
        total = 0.0
        for area in self.areas:
            if hasattr(area, 'prediction_layer') and area.prediction_layer is not None:
                total += area.prediction_layer.get_free_energy().item()
        return total
