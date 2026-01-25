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

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import WeightInitializer
from thalia.config.region_configs import (
    PredictiveCodingErrorType,
    PredictiveCodingConfig,
    PredictiveCortexConfig,
)
from thalia.core.neural_region import NeuralRegion
from thalia.managers.component_registry import register_region
from thalia.mixins.diagnostics_mixin import DiagnosticsMixin

from .layered_cortex import LayeredCortex
from .state import PredictiveCodingState, PredictiveCortexState


class PredictiveCodingLayer(DiagnosticsMixin, nn.Module):
    """
    Predictive Coding Layer with local, backprop-free learning.

    This layer learns to predict its inputs from a higher-level representation.
    Learning is driven entirely by local prediction errors - no gradients
    need to flow backwards through the network.

    The key insight: instead of propagating errors backward (backprop),
    we compute errors LOCALLY by comparing predictions to actual inputs.
    This is more biologically plausible and enables hierarchical learning.

    Credit Assignment Solution:
    ===========================
    Traditional backprop: Error at output → propagated to all weights
    Predictive coding: Error at EACH LAYER → updates THAT layer's weights

    This means:
    1. No weight transport problem (no need to know downstream weights)
    2. No non-local computation (all signals are local)
    3. Continuous learning (no separate forward/backward passes)

    Usage:
        layer = PredictiveCodingLayer(PredictiveCodingConfig(
            n_input=256,
            n_representation=64,
        ))

        # Reset for new sequence
        layer.reset_state(batch_size=1)

        # Each timestep
        for t in range(n_timesteps):
            # Bottom-up input (from lower layer or sensors)
            actual_input = get_input(t)

            # Top-down representation (from higher layer)
            representation = get_higher_representation(t)

            # Compute prediction error
            error, prediction = layer(actual_input, representation)

            # Error goes to higher layer (it's their input now!)
            # This creates hierarchical prediction

        # Learning happens continuously based on errors
        layer.learn()
    """

    def __init__(self, config: PredictiveCodingConfig):
        super().__init__()
        self.config = config
        # Don't cache device - use property instead to track actual module device

        # =================================================================
        # PREDICTION PATHWAY (top-down)
        # =================================================================
        # Learns to predict input from representation
        # W_pred: representation → predicted_input

        device = torch.device(config.device)  # Use local variable for initialization
        self.W_pred = nn.Parameter(
            WeightInitializer.gaussian(
                config.n_input,
                config.n_representation,
                mean=0.0,
                std=0.1,
                device=device,
            )
        )

        # Prediction bias (learned prior)
        self.b_pred = nn.Parameter(torch.zeros(config.n_input, device=device))

        # =================================================================
        # ERROR COMPUTATION (local, no backprop needed)
        # =================================================================
        # Error neurons receive:
        #   + excitation from actual input
        #   - inhibition from prediction
        # Net activity = error signal

        # Spiking error neurons (fast dynamics)
        error_config = ConductanceLIFConfig(
            g_L=1.0 / config.error_tau_ms,  # Convert tau to conductance
            tau_E=5.0,  # Fast excitatory
            tau_I=10.0,  # Fast inhibitory
            v_threshold=0.5,  # Lower threshold for sensitive error detection
        )
        self.error_neurons = ConductanceLIF(config.n_input, error_config, device=device)

        # Spiking prediction neurons (slow dynamics)
        pred_config = ConductanceLIFConfig(
            g_L=1.0 / config.prediction_tau_ms,  # Convert tau to conductance
            tau_E=10.0,  # Slower excitatory for temporal integration
            tau_I=15.0,
            v_threshold=1.0,
        )
        self.prediction_neurons = ConductanceLIF(config.n_input, pred_config, device=device)

        # =================================================================
        # PRECISION (attention/confidence weighting)
        # =================================================================
        # High precision = pay attention to errors
        # Low precision = ignore errors (unreliable input)

        self.log_precision = nn.Parameter(
            torch.full(
                (config.n_input,),
                fill_value=torch.log(torch.tensor(config.initial_precision)).item(),
                device=device,
            )
        )

        # =================================================================
        # REPRESENTATION → PREDICTION ENCODING
        # =================================================================
        # Optional: learn a compact representation
        self.W_encode = nn.Parameter(
            WeightInitializer.gaussian(
                config.n_representation,
                config.n_input,
                mean=0.0,
                std=0.1,
                device=device,
            )
        )

        # =================================================================
        # STATE
        # =================================================================
        self.state = PredictiveCodingState()

        # =================================================================
        # TEMPORAL VARIANCE TRACKING FOR PRECISION LEARNING
        # =================================================================
        # Maintain a circular buffer of recent errors for variance estimation
        # This allows precision to adapt based on how predictable inputs have been
        self._error_history: list[torch.Tensor] = []
        self._timestep_counter: int = 0

        # Cached decay factors (updated via update_temporal_parameters)
        self._prediction_decay: Optional[float] = None
        self._error_decay: Optional[float] = None

    @property
    def device(self) -> torch.device:
        """Get current device from parameters (tracks module.to() calls)."""
        return self.W_pred.device

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update cached decay factors when dt changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._prediction_decay = float(
            torch.exp(torch.tensor(-dt_ms / self.config.prediction_tau_ms)).item()
        )
        self._error_decay = float(
            torch.exp(torch.tensor(-dt_ms / self.config.error_tau_ms)).item()
        )

    def reset_state(self) -> None:
        """Reset layer state for new sequence."""
        # Ensure decay factors are initialized (use dt=1.0 if not set)
        if self._prediction_decay is None or self._error_decay is None:
            self.update_temporal_parameters(1.0)

        # ADR-005: Single-instance architecture (1D tensors, no batch dimension)
        self.state = PredictiveCodingState(
            prediction=torch.zeros(self.config.n_input, device=self.device),
            representation=torch.zeros(self.config.n_representation, device=self.device),
            error=torch.zeros(self.config.n_input, device=self.device),
            precision=self.precision,
            eligibility=torch.zeros(
                self.config.n_input, self.config.n_representation, device=self.device
            ),
        )

        # Clear temporal variance tracking
        self._error_history.clear()
        self._timestep_counter = 0

        # Reset spiking neurons
        self.error_neurons.reset_state()
        self.prediction_neurons.reset_state()

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        state_dict = {
            "W_pred": self.W_pred.data if hasattr(self, "W_pred") else None,
            "W_encode": self.W_encode.data if hasattr(self, "W_encode") else None,
            "log_precision": self.log_precision.data if hasattr(self, "log_precision") else None,
            "prediction": (
                self.state.prediction.clone()
                if (hasattr(self.state, "prediction") and self.state.prediction is not None)
                else None
            ),
            "error": (
                self.state.error.clone()
                if (hasattr(self.state, "error") and self.state.error is not None)
                else None
            ),
        }
        return state_dict

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        if state_dict.get("W_pred") is not None and hasattr(self, "W_pred"):
            self.W_pred.data.copy_(state_dict["W_pred"].to(self.device))
        if state_dict.get("W_encode") is not None and hasattr(self, "W_encode"):
            self.W_encode.data.copy_(state_dict["W_encode"].to(self.device))
        if state_dict.get("log_precision") is not None and hasattr(self, "log_precision"):
            self.log_precision.data.copy_(state_dict["log_precision"].to(self.device))
        if state_dict.get("prediction") is not None and hasattr(self.state, "prediction"):
            self.state.prediction = state_dict["prediction"].to(self.device)
        if state_dict.get("error") is not None and hasattr(self.state, "error"):
            self.state.error = state_dict["error"].to(self.device)

    @property
    def precision(self) -> torch.Tensor:
        """Get precision (clamped to valid range)."""
        return torch.clamp(
            torch.exp(self.log_precision), self.config.precision_min, self.config.precision_max
        )

    def predict(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction from representation (top-down).

        Args:
            representation: Higher-level representation [batch, n_representation]

        Returns:
            prediction: Predicted input [batch, n_input]
        """
        # Linear prediction (can be made nonlinear if needed)
        prediction = F.linear(representation, self.W_pred, self.b_pred)

        # Convert to conductance and get spikes (ConductanceLIF expects g_exc)
        pred_g_exc = F.relu(prediction)  # Clamp to positive conductance
        pred_spikes, pred_membrane = self.prediction_neurons(pred_g_exc, g_inh_input=None)
        return pred_membrane  # type: ignore[no-any-return]  # Use membrane potential as analog prediction

    def compute_error(
        self,
        actual: torch.Tensor,
        predicted: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prediction error (actual - predicted).

        This is the core of predictive coding:
        - Positive error: under-predicted (need to increase prediction)
        - Negative error: over-predicted (need to decrease prediction)

        Per ADR-005: Single-instance architecture (1D tensors, no batch dimension)

        Args:
            actual: Actual input [n_input]
            predicted: Predicted input [n_input]

        Returns:
            error: Precision-weighted prediction error [n_input]
        """
        # Raw error
        raw_error = actual - predicted

        # Error neurons: convert signed error to separate E/I conductances
        # Positive error → excitation, negative error → inhibition
        if self.config.error_type == PredictiveCodingErrorType.SIGNED:
            # Split signed error into excitatory (positive) and inhibitory (negative)
            error_g_exc = F.relu(raw_error)  # Positive errors
            error_g_inh = F.relu(-raw_error)  # Negative errors (flipped)
            error_spikes, error_membrane = self.error_neurons(error_g_exc, error_g_inh)
            error = error_membrane
        else:
            # Two populations: E+ for positive, E- for negative
            # Not implemented for simplicity, but would double neurons
            raise NotImplementedError("Separate E+/E- populations not yet implemented")

        # Apply precision weighting (high precision = amplify error)
        precision_weighted_error = error * self.precision

        return precision_weighted_error  # type: ignore[no-any-return]

    def encode(self, error: torch.Tensor) -> torch.Tensor:
        """
        Encode error into representation update.

        The representation is updated to reduce future prediction errors.
        This is the "explaining away" of prediction errors.

        Args:
            error: Prediction error [batch, n_input]

        Returns:
            representation_update: Update to representation [batch, n_representation]
        """
        # Project error into representation space
        # This tells the representation what to change
        return F.linear(error, self.W_encode)

    def forward(
        self,
        actual_input: torch.Tensor,
        representation: Optional[torch.Tensor] = None,
        top_down_prediction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one timestep of predictive coding.

        Per ADR-005: Single-instance architecture (1D tensors, no batch dimension)

        Args:
            actual_input: Bottom-up input (sensory or from lower layer) [n_input]
            representation: Current representation from higher layer [n_representation]
            top_down_prediction: Optional direct prediction from above [n_input]

        Returns:
            error: Prediction error (goes UP to higher layers)
            prediction: Current prediction (for diagnostics)
            representation_update: Suggested update to representation
        """
        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert actual_input.shape[-1] == self.config.n_input, (
            f"PredictiveCodingLayer.forward: actual_input has shape {actual_input.shape} "
            f"but n_input={self.config.n_input}. Check that you're passing L4 output."
        )
        if representation is not None:
            assert representation.shape[-1] == self.config.n_representation, (
                f"PredictiveCodingLayer.forward: representation has shape {representation.shape} "
                f"but n_representation={self.config.n_representation}. Check that you're passing L5 output."
            )
        if top_down_prediction is not None:
            assert top_down_prediction.shape[-1] == self.config.n_input, (
                f"PredictiveCodingLayer.forward: top_down_prediction has shape {top_down_prediction.shape} "
                f"but must match n_input={self.config.n_input}. "
                f"Did you pass L2/3 modulation (size {top_down_prediction.shape[-1]}) instead of L4 prediction?"
            )

        # Initialize state if needed
        if self.state.prediction is None:
            self.reset_state()

        # Get current representation (from input or stored)
        if representation is not None:
            self.state.representation = representation

        # Generate prediction (top-down)
        if top_down_prediction is not None:
            prediction = top_down_prediction
        else:
            prediction = self.predict(self.state.representation)  # type: ignore[arg-type]

        # Smooth prediction over time (temporal integration)
        assert self._prediction_decay is not None
        self.state.prediction = (
            self._prediction_decay * self.state.prediction + (1 - self._prediction_decay) * prediction  # type: ignore[operator]
        )

        # Compute prediction error
        error = self.compute_error(actual_input, self.state.prediction)

        # Store error
        self.state.error = error

        # Compute representation update from error
        representation_update = self.encode(error)

        # Update eligibility trace for learning
        self._update_eligibility(actual_input, self.state.representation, error)  # type: ignore[arg-type]

        # Error propagates upward with same dimensionality as input
        # (Per ADR-013: dimensional transformations handled by pathways)
        return error, self.state.prediction, representation_update

    def _update_eligibility(
        self,
        actual: torch.Tensor,
        representation: torch.Tensor,
        error: torch.Tensor,
    ) -> None:
        """
        Update eligibility traces for local learning.

        The eligibility trace records which weight contributed to which error.
        This enables proper credit assignment without backprop.

        Eligibility = outer_product(error, representation)
        This is a Hebbian-like trace: which inputs were active when error occurred.
        """
        # Outer product: [n_input] x [n_representation] → [n_input, n_representation]
        # Per ADR-005: single-instance architecture (1D tensors, no batch dimension)
        eligibility_update = torch.einsum("i,j->ij", error, representation)

        # Exponential moving average of eligibility
        eligibility_decay = 0.95
        self.state.eligibility = (
            eligibility_decay * self.state.eligibility  # type: ignore[operator]
            + (1 - eligibility_decay) * eligibility_update
        )

    def learn(self, reward_signal: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Update weights based on accumulated prediction errors.

        This is where the magic happens - LOCAL learning that doesn't need backprop!

        The learning rule:
            ΔW_pred = η * precision * error * representation^T

        This means:
        - Increase prediction weights for inputs that were under-predicted
        - Decrease prediction weights for inputs that were over-predicted
        - Modulate by precision (pay attention to reliable signals)

        Optionally modulated by reward signal for three-factor learning.

        Returns:
            metrics: Learning metrics for monitoring
        """
        if self.state.eligibility is None:
            return {"weight_update": 0.0}

        # Average eligibility over batch
        eligibility = self.state.eligibility.mean(dim=0)  # [n_input, n_representation]

        # Optional reward modulation (three-factor rule)
        if reward_signal is not None:
            eligibility = eligibility * reward_signal.unsqueeze(-1)

        # Apply precision weighting
        precision_weights = self.precision.unsqueeze(-1)  # [n_input, 1]
        weighted_eligibility = eligibility * precision_weights

        # Update prediction weights
        weight_update = self.config.learning_rate * weighted_eligibility
        self.W_pred.data += weight_update

        # Weight normalization (homeostatic)
        w_norm = torch.norm(self.W_pred, dim=1, keepdim=True)
        self.W_pred.data = (
            self.W_pred.data
            / (w_norm + 1e-8)
            * torch.sqrt(torch.tensor(self.config.n_representation, dtype=torch.float))
        )

        # Update precision based on error statistics
        self._update_precision()

        # Clear eligibility after learning
        self.state.eligibility.zero_()

        return {
            "weight_update": weight_update.abs().mean().item(),
            "precision_mean": self.precision.mean().item(),
        }

    def _update_precision(self) -> None:
        """
        Update precision (inverse variance) based on TEMPORAL error history.

        Precision reflects how predictable inputs have been recently:
        - High precision: errors have been consistently small (reliable predictions)
        - Low precision: errors have been variable (unreliable/surprising input)

        This is biologically plausible - synaptic precision should adapt based
        on recent prediction history, not parallel samples.

        Implementation:
        - Maintain a circular buffer of recent errors (across timesteps)
        - Compute variance over the temporal dimension
        - Update precision periodically (not every timestep)
        """
        if self.state.error is None:
            return

        # Add current error to history (mean across batch for single value per input)
        # Detach to avoid gradient accumulation
        error_snapshot = self.state.error.detach().mean(dim=0)  # [n_input]
        self._error_history.append(error_snapshot)

        # Keep buffer at configured size (circular buffer behavior)
        if len(self._error_history) > self.config.error_history_size:
            self._error_history.pop(0)

        # Increment timestep counter
        self._timestep_counter += 1

        # Only update precision periodically and when we have enough history
        if (
            self._timestep_counter % self.config.precision_update_interval != 0
            or len(self._error_history) < 10
        ):  # Need at least 10 samples for meaningful variance
            return

        # Stack history into tensor: [history_size, n_input]
        error_history_tensor = torch.stack(self._error_history, dim=0)

        # Compute variance over time dimension (dim=0)
        error_var = error_history_tensor.var(dim=0, correction=1)  # [n_input]

        # Update log precision (in log space for stability)
        # precision = 1 / variance, so log_precision = -log(variance)
        target_log_precision = -torch.log(error_var + 1e-6)

        self.log_precision.data = (
            1 - self.config.precision_learning_rate
        ) * self.log_precision.data + self.config.precision_learning_rate * target_log_precision

        # Clamp to valid range
        min_log = torch.log(torch.tensor(self.config.precision_min, device=self.device))
        max_log = torch.log(torch.tensor(self.config.precision_max, device=self.device))
        self.log_precision.data = torch.clamp(self.log_precision.data, min_log, max_log)

    def get_free_energy(self) -> torch.Tensor:
        """
        Compute free energy (prediction error + complexity).

        The brain minimizes free energy, which balances:
        - Accuracy: prediction error should be small
        - Complexity: model shouldn't be overly complex

        F = precision * error^2 + log(precision)

        The first term is weighted squared error.
        The second term penalizes high precision (overconfidence).
        """
        if self.state.error is None:
            return torch.tensor(0.0)

        error_term = (self.precision * self.state.error**2).sum()
        complexity_term = self.log_precision.sum()

        return error_term + complexity_term

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information using DiagnosticsMixin helpers."""
        diag: Dict[str, Any] = {
            "free_energy": self.get_free_energy().item(),
        }

        # Prediction state
        if self.state.prediction is not None:
            diag["prediction_mean"] = self.state.prediction.mean().item()
            diag["prediction_std"] = self.state.prediction.std().item()

        # Error diagnostics
        if self.state.error is not None:
            diag.update(self.spike_diagnostics(self.state.error, "error"))

        # Precision (attention) statistics
        diag["precision_mean"] = self.precision.mean().item()
        diag["precision_std"] = self.precision.std().item()
        diag["precision_min"] = self.precision.min().item()
        diag["precision_max"] = self.precision.max().item()

        # Temporal variance tracking stats
        diag["error_history_size"] = len(self._error_history)
        diag["timestep_counter"] = self._timestep_counter
        if len(self._error_history) >= 2:
            # Compute current temporal variance estimate
            error_history_tensor = torch.stack(self._error_history, dim=0)
            temporal_var = error_history_tensor.var(dim=0).mean().item()
            diag["temporal_error_variance"] = temporal_var

        # Weight diagnostics
        diag.update(self.weight_diagnostics(self.W_pred, "pred"))

        # Eligibility trace diagnostics (if available)
        if self.state.eligibility is not None:
            diag.update(self.trace_diagnostics(self.state.eligibility, "elig"))

        return diag


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
            default_learning_strategy=None,  # Managed by inner cortex
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

        # Register output port (port-based routing support)
        self.register_output_port("default", self.config.n_representation)

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

        # Set port output (port-based routing support)
        self.clear_port_outputs()
        self.set_port_output("default", output)

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
