"""Predictive Coding Layer - Hierarchical Prediction Error Minimization (Free Energy).

This module implements predictive coding, a biologically-plausible theory of
cortical function where perception and learning emerge from minimizing
prediction errors rather than backpropagation.

**The Core Idea**:
==================
1. Each layer generates **predictions** about the layer below
2. Only **prediction ERRORS** propagate upward (sparse, efficient)
3. Learning **minimizes prediction error** locally (no backprop needed!)

**Why This Matters**:
=====================
This is fundamentally different from traditional deep learning:

- **Traditional**: Errors backpropagate through layers (biologically implausible)
- **Predictive Coding**: Errors computed LOCALLY, learning is LOCAL

**Key Insight**: The brain may not need backprop because it uses a different
objective function - minimizing free energy / prediction error at each hierarchical level.

**Architecture**:
=================

.. code-block:: none

    Higher Cortical Areas
           │
           │ Predictions (top-down)
           ▼
    ┌──────────────────────────────────────────────────────┐
    │              PREDICTIVE CODING LAYER                  │
    │                                                       │
    │   ┌─────────────┐      ┌─────────────────────┐       │
    │   │ Prediction  │      │   Error Neurons     │       │
    │   │  Neurons    │──────│   (E+ and E-)       │───────┼──► Error to higher
    │   │  (top-down) │      │                     │       │    layers
    │   └─────────────┘      └─────────────────────┘       │
    │          │                      ▲                    │
    │          │ Predicted            │ Actual input       │
    │          ▼                      │                    │
    │   ┌─────────────────────────────┼────────────────┐   │
    │   │              Comparison                      │   │
    │   │         Error = Actual - Predicted           │   │
    │   └──────────────────────────────────────────────┘   │
    │                                                       │
    └──────────────────────────────────────────────────────┘
           ▲
           │ Bottom-up input
    Lower Areas / Sensory Input

**Key Features**:
=================
1. **LOCAL LEARNING**: Each layer learns to predict its inputs (no global error)
2. **ERROR SPARSITY**: Only mismatches generate activity (efficient coding)
3. **PRECISION WEIGHTING**: Confidence modulates error propagation
4. **NATURAL HIERARCHY**: Abstractions emerge from prediction dynamics
5. **ATTENTION VIA PRECISION**: Top-down precision modulation = attention

Biological basis:
- Superficial layers (L2/3): Error neurons, forward projections
- Deep layers (L5/6): Prediction neurons, feedback projections
- NMDA currents: Slow integration for prediction
- AMPA currents: Fast error signaling

References:
- Rao & Ballard (1999): Predictive coding in visual cortex
- Friston (2005): Free energy principle
- Bastos et al. (2012): Canonical microcircuits for predictive coding
- Keller & Mrsic-Flogel (2018): Predictive processing in visual cortex

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.mixins.diagnostics_mixin import DiagnosticsMixin


class ErrorType(Enum):
    """Types of prediction errors."""
    POSITIVE = "positive"  # Actual > Predicted (under-prediction)
    NEGATIVE = "negative"  # Actual < Predicted (over-prediction)
    SIGNED = "signed"      # Single population with +/- values


@dataclass
class PredictiveCodingConfig:
    """Configuration for a predictive coding layer.

    Attributes:
        n_input: Size of input (from lower layer or sensory)
        n_representation: Size of internal representation (prediction neurons)
        n_output: Size of output to higher layers (typically = n_input for residuals)

        # Prediction dynamics
        prediction_tau_ms: Time constant for prediction integration (slow, NMDA-like)
        error_tau_ms: Time constant for error neurons (fast, AMPA-like)

        # Learning parameters
        learning_rate: Base learning rate for prediction weight updates
        precision_learning_rate: Learning rate for precision updates

        # Precision (attention/confidence) parameters
        initial_precision: Starting precision (inverse variance)
        precision_min: Minimum precision (prevents division by zero)
        precision_max: Maximum precision (prevents over-confidence)

        # Architecture choices
        error_type: How errors are represented (separate +/- or signed)
        use_spiking: Whether to use spiking neurons (vs rate-based)
        sparse_coding: Apply sparsity constraint on representations
        sparsity_target: Target activation fraction if sparse_coding=True

        dt_ms: Simulation timestep
        device: Computation device
    """
    n_input: int = 256
    n_representation: int = 128
    n_output: int = 256  # Usually same as n_input for residuals

    # Dynamics
    prediction_tau_ms: float = 50.0   # Slow (NMDA-like) for stable predictions
    error_tau_ms: float = 5.0          # Fast (AMPA-like) for quick error signaling

    # Learning
    learning_rate: float = 0.01
    precision_learning_rate: float = 0.001

    # Precision (attention)
    initial_precision: float = 1.0
    precision_min: float = 0.1
    precision_max: float = 10.0

    # Temporal variance tracking for precision learning
    # Precision is updated based on variance of errors over recent history
    error_history_size: int = 50  # Number of timesteps to track for variance
    precision_update_interval: int = 10  # Update precision every N timesteps

    # Architecture
    error_type: ErrorType = ErrorType.SIGNED
    use_spiking: bool = True
    sparse_coding: bool = True
    sparsity_target: float = 0.1

    dt_ms: float = 1.0
    device: str = "cpu"


@dataclass
class PredictiveCodingState:
    """State of a predictive coding layer."""
    # Representations
    prediction: Optional[torch.Tensor] = None      # Current prediction of input
    representation: Optional[torch.Tensor] = None  # Internal representation
    error: Optional[torch.Tensor] = None           # Prediction error

    # For spiking implementation
    prediction_membrane: Optional[torch.Tensor] = None
    error_membrane: Optional[torch.Tensor] = None

    # Precision tracking
    precision: Optional[torch.Tensor] = None

    # Learning eligibility
    eligibility: Optional[torch.Tensor] = None


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
        self.device = torch.device(config.device)

        # =================================================================
        # PREDICTION PATHWAY (top-down)
        # =================================================================
        # Learns to predict input from representation
        # W_pred: representation → predicted_input

        from thalia.components.synapses.weight_init import WeightInitializer
        from thalia.regulation.learning_constants import WEIGHT_INIT_SCALE_PREDICTIVE

        self.W_pred = nn.Parameter(
            WeightInitializer.gaussian(
                config.n_input, config.n_representation,
                mean=0.0, std=WEIGHT_INIT_SCALE_PREDICTIVE,
                device=self.device
            )
        )

        # Prediction bias (learned prior)
        self.b_pred = nn.Parameter(
            torch.zeros(config.n_input, device=self.device)
        )

        # =================================================================
        # ERROR COMPUTATION (local, no backprop needed)
        # =================================================================
        # Error neurons receive:
        #   + excitation from actual input
        #   - inhibition from prediction
        # Net activity = error signal

        if config.use_spiking:
            # Spiking error neurons (fast dynamics)
            error_config = ConductanceLIFConfig(
                g_L=1.0 / config.error_tau_ms,  # Convert tau to conductance
                tau_E=5.0,  # Fast excitatory
                tau_I=10.0,  # Fast inhibitory
                v_threshold=0.5,  # Lower threshold for sensitive error detection
                dt_ms=config.dt_ms,
            )
            self.error_neurons = ConductanceLIF(config.n_input, error_config)

            # Spiking prediction neurons (slow dynamics)
            pred_config = ConductanceLIFConfig(
                g_L=1.0 / config.prediction_tau_ms,  # Convert tau to conductance
                tau_E=10.0,  # Slower excitatory for temporal integration
                tau_I=15.0,
                v_threshold=1.0,
                dt_ms=config.dt_ms,
            )
            self.prediction_neurons = ConductanceLIF(config.n_input, pred_config)
        else:
            self.error_neurons = None
            self.prediction_neurons = None

        # =================================================================
        # PRECISION (attention/confidence weighting)
        # =================================================================
        # High precision = pay attention to errors
        # Low precision = ignore errors (unreliable input)

        self.log_precision = nn.Parameter(
            torch.full((config.n_input,),
                      fill_value=torch.log(torch.tensor(config.initial_precision)).item(),
                      device=self.device)
        )

        # =================================================================
        # REPRESENTATION → PREDICTION ENCODING
        # =================================================================
        # Optional: learn a compact representation
        self.W_encode = nn.Parameter(
            WeightInitializer.gaussian(
                config.n_representation, config.n_input,
                mean=0.0, std=WEIGHT_INIT_SCALE_PREDICTIVE,
                device=self.device
            )
        )

        # =================================================================
        # OUTPUT PROJECTION (if n_output != n_input)
        # =================================================================
        if config.n_output != config.n_input:
            self.W_output = nn.Parameter(
                WeightInitializer.gaussian(
                    config.n_output, config.n_input,
                    mean=0.0, std=WEIGHT_INIT_SCALE_PREDICTIVE,
                    device=self.device
                )
            )
        else:
            self.W_output = None

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

        # Decay factors
        self.register_buffer(
            "prediction_decay",
            torch.tensor(torch.exp(torch.tensor(-config.dt_ms / config.prediction_tau_ms)).item())
        )
        self.register_buffer(
            "error_decay",
            torch.tensor(torch.exp(torch.tensor(-config.dt_ms / config.error_tau_ms)).item())
        )

    def reset_state(self) -> None:
        """Reset layer state for new sequence."""
        # ADR-005: Single-instance architecture (1D tensors, no batch dimension)
        self.state = PredictiveCodingState(
            prediction=torch.zeros(self.config.n_input, device=self.device),
            representation=torch.zeros(self.config.n_representation, device=self.device),
            error=torch.zeros(self.config.n_input, device=self.device),
            precision=self.precision,
            eligibility=torch.zeros(
                self.config.n_input, self.config.n_representation,
                device=self.device
            ),
        )

        # Clear temporal variance tracking
        self._error_history.clear()
        self._timestep_counter = 0

        if self.config.use_spiking:
            self.error_neurons.reset_state()
            self.prediction_neurons.reset_state()

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        state_dict = {
            "W_pred": self.W_pred.data if hasattr(self, 'W_pred') else None,
            "W_encode": self.W_encode.data if hasattr(self, 'W_encode') else None,
            "log_precision": self.log_precision.data if hasattr(self, 'log_precision') else None,
            "prediction": self.state.prediction.clone() if (hasattr(self.state, 'prediction') and self.state.prediction is not None) else None,
            "error": self.state.error.clone() if (hasattr(self.state, 'error') and self.state.error is not None) else None,
        }
        return state_dict

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        if state_dict.get("W_pred") is not None and hasattr(self, 'W_pred'):
            self.W_pred.data.copy_(state_dict["W_pred"].to(self.device))
        if state_dict.get("W_encode") is not None and hasattr(self, 'W_encode'):
            self.W_encode.data.copy_(state_dict["W_encode"].to(self.device))
        if state_dict.get("log_precision") is not None and hasattr(self, 'log_precision'):
            self.log_precision.data.copy_(state_dict["log_precision"].to(self.device))
        if state_dict.get("prediction") is not None and hasattr(self.state, 'prediction'):
            self.state.prediction = state_dict["prediction"].to(self.device)
        if state_dict.get("error") is not None and hasattr(self.state, 'error'):
            self.state.error = state_dict["error"].to(self.device)

    @property
    def precision(self) -> torch.Tensor:
        """Get precision (clamped to valid range)."""
        return torch.clamp(
            torch.exp(self.log_precision),
            self.config.precision_min,
            self.config.precision_max
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

        if self.config.use_spiking:
            # Convert to conductance and get spikes (ConductanceLIF expects g_exc)
            pred_g_exc = F.relu(prediction)  # Clamp to positive conductance
            pred_spikes, pred_membrane = self.prediction_neurons(pred_g_exc, g_inh_input=None)
            return pred_membrane  # Use membrane potential as analog prediction
        else:
            return prediction

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

        if self.config.use_spiking:
            # Error neurons: convert signed error to separate E/I conductances
            # Positive error → excitation, negative error → inhibition
            if self.config.error_type == ErrorType.SIGNED:
                # Split signed error into excitatory (positive) and inhibitory (negative)
                error_g_exc = F.relu(raw_error)   # Positive errors
                error_g_inh = F.relu(-raw_error)  # Negative errors (flipped)
                error_spikes, error_membrane = self.error_neurons(error_g_exc, error_g_inh)
                error = error_membrane
            else:
                # Two populations: E+ for positive, E- for negative
                # Not implemented for simplicity, but would double neurons
                raise NotImplementedError("Separate E+/E- populations not yet implemented")
        else:
            error = raw_error

        # Apply precision weighting (high precision = amplify error)
        precision_weighted_error = error * self.precision

        return precision_weighted_error

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
            prediction = self.predict(self.state.representation)

        # Smooth prediction over time (temporal integration)
        self.state.prediction = (
            self.prediction_decay * self.state.prediction +
            (1 - self.prediction_decay) * prediction
        )

        # Compute prediction error
        error = self.compute_error(actual_input, self.state.prediction)

        # Store error
        self.state.error = error

        # Compute representation update from error
        representation_update = self.encode(error)

        # Update eligibility trace for learning
        self._update_eligibility(actual_input, self.state.representation, error)

        # Project error to output dimension if needed
        if self.W_output is not None:
            output_error = F.linear(error, self.W_output)
        else:
            output_error = error

        return output_error, self.state.prediction, representation_update

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
        eligibility_update = torch.einsum('i,j->ij', error, representation)

        # Exponential moving average of eligibility
        eligibility_decay = 0.95
        self.state.eligibility = (
            eligibility_decay * self.state.eligibility +
            (1 - eligibility_decay) * eligibility_update
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
        with torch.no_grad():
            weight_update = self.config.learning_rate * weighted_eligibility
            self.W_pred.data += weight_update

            # Weight normalization (homeostatic)
            w_norm = torch.norm(self.W_pred, dim=1, keepdim=True)
            self.W_pred.data = self.W_pred.data / (w_norm + 1e-8) * torch.sqrt(
                torch.tensor(self.config.n_representation, dtype=torch.float)
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
        if (self._timestep_counter % self.config.precision_update_interval != 0 or
            len(self._error_history) < 10):  # Need at least 10 samples for meaningful variance
            return

        # Stack history into tensor: [history_size, n_input]
        error_history_tensor = torch.stack(self._error_history, dim=0)

        # Compute variance over time dimension (dim=0)
        error_var = error_history_tensor.var(dim=0, correction=1)  # [n_input]

        # Update log precision (in log space for stability)
        # precision = 1 / variance, so log_precision = -log(variance)
        target_log_precision = -torch.log(error_var + 1e-6)

        with torch.no_grad():
            self.log_precision.data = (
                (1 - self.config.precision_learning_rate) * self.log_precision.data +
                self.config.precision_learning_rate * target_log_precision
            )

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

        error_term = (self.precision * self.state.error ** 2).sum()
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


class HierarchicalPredictiveCoding(nn.Module):
    """
    Stack of predictive coding layers forming a hierarchy.

    Each layer predicts the errors from the layer below, creating
    an increasingly abstract representation as you go up.

    Information flow:
    =================

    Layer 3: Most abstract (predicts error from layer 2)
        ▲ errors          │ predictions
        │                 ▼
    Layer 2: Intermediate (predicts error from layer 1)
        ▲ errors          │ predictions
        │                 ▼
    Layer 1: Concrete (predicts sensory input)
        ▲                 │
        │ sensory input   ▼ prediction

    Learning cascades: each layer minimizes its prediction error locally.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        representation_ratios: list[float] | None = None,
        config_overrides: dict | None = None,
    ):
        """
        Initialize hierarchical predictive coding.

        Args:
            layer_sizes: Size of each layer [input, layer1, layer2, ...]
            representation_ratios: Compression ratio for each layer (default 0.5)
            config_overrides: Override default config parameters
        """
        super().__init__()

        n_layers = len(layer_sizes) - 1

        if representation_ratios is None:
            representation_ratios = [0.5] * n_layers

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            n_input = layer_sizes[i]
            n_next = layer_sizes[i + 1]
            n_repr = int(n_next * representation_ratios[i])

            config = PredictiveCodingConfig(
                n_input=n_input,
                n_representation=n_repr,
                n_output=n_next,  # Output matches next layer's input
                **(config_overrides or {})
            )

            self.layers.append(PredictiveCodingLayer(config))

    def reset_state(self) -> None:
        """Reset all layers."""
        for layer in self.layers:
            layer.reset_state()

    def forward(
        self,
        sensory_input: torch.Tensor,
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Process input through hierarchy.

        Args:
            sensory_input: Raw sensory input [batch, n_input]

        Returns:
            errors: List of errors at each level
            representations: List of representations at each level
        """
        errors = []
        representations = []

        # Bottom-up pass: compute errors at each level
        current_input = sensory_input

        for layer in self.layers:
            error, prediction, repr_update = layer(current_input)
            errors.append(error)

            # Next layer receives error as input
            current_input = error

            # Store representation
            representations.append(layer.state.representation)

        return errors, representations

    def learn(self, reward_signal: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Learn at all layers."""
        metrics = {}
        for i, layer in enumerate(self.layers):
            layer_metrics = layer.learn(reward_signal)
            for k, v in layer_metrics.items():
                metrics[f"layer{i}_{k}"] = v
        return metrics

    def get_total_free_energy(self) -> torch.Tensor:
        """Sum of free energy across all layers."""
        return sum(layer.get_free_energy() for layer in self.layers)
