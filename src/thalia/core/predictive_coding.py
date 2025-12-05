"""
Predictive Coding Layer - Hierarchical Prediction Error Minimization.

This module implements predictive coding, a theory of cortical function where:
1. Each layer generates predictions about the layer below
2. Only prediction ERRORS propagate upward (sparse, efficient)
3. Learning minimizes prediction error (no backprop needed!)

This is a fundamentally different approach to credit assignment:
- Traditional: Errors backpropagate through layers (biologically implausible)
- Predictive: Errors are computed LOCALLY, learning is LOCAL

Key insight: The brain may not need backprop because it uses a different
objective - minimizing free energy / prediction error at each level.

Architecture:
=============

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

Key Features:
=============
1. LOCAL LEARNING: Each layer learns to predict its inputs
2. ERROR SPARSITY: Only mismatches generate activity
3. PRECISION WEIGHTING: Confidence modulates error propagation
4. NATURAL HIERARCHY: Abstractions emerge from prediction
5. ATTENTION via PRECISION: Top-down precision = attention

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

from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF, ConductanceLIFConfig


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


class PredictiveCodingLayer(nn.Module):
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
        
        self.W_pred = nn.Parameter(
            torch.randn(config.n_input, config.n_representation, device=self.device) * 0.1
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
            error_lif_config = LIFConfig(
                tau_mem=config.error_tau_ms,
                v_threshold=0.5,  # Lower threshold for sensitive error detection
                dt=config.dt_ms,
            )
            self.error_neurons = LIFNeuron(config.n_input, error_lif_config)
            
            # Spiking prediction neurons (slow dynamics)
            pred_lif_config = LIFConfig(
                tau_mem=config.prediction_tau_ms,
                v_threshold=1.0,
                dt=config.dt_ms,
            )
            self.prediction_neurons = LIFNeuron(config.n_input, pred_lif_config)
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
            torch.randn(config.n_representation, config.n_input, device=self.device) * 0.1
        )
        
        # =================================================================
        # OUTPUT PROJECTION (if n_output != n_input)
        # =================================================================
        if config.n_output != config.n_input:
            self.W_output = nn.Parameter(
                torch.randn(config.n_output, config.n_input, device=self.device) * 0.1
            )
        else:
            self.W_output = None
        
        # =================================================================
        # STATE
        # =================================================================
        self.state = PredictiveCodingState()
        
        # Decay factors
        self.register_buffer(
            "prediction_decay",
            torch.tensor(torch.exp(torch.tensor(-config.dt_ms / config.prediction_tau_ms)).item())
        )
        self.register_buffer(
            "error_decay", 
            torch.tensor(torch.exp(torch.tensor(-config.dt_ms / config.error_tau_ms)).item())
        )
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset layer state for new sequence."""
        self.state = PredictiveCodingState(
            prediction=torch.zeros(batch_size, self.config.n_input, device=self.device),
            representation=torch.zeros(batch_size, self.config.n_representation, device=self.device),
            error=torch.zeros(batch_size, self.config.n_input, device=self.device),
            precision=self.precision,
            eligibility=torch.zeros(
                batch_size, self.config.n_input, self.config.n_representation, 
                device=self.device
            ),
        )
        
        if self.config.use_spiking:
            self.error_neurons.reset_state(batch_size)
            self.prediction_neurons.reset_state(batch_size)
    
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
            # Convert to current and get spikes
            pred_spikes, pred_membrane = self.prediction_neurons(prediction)
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
        
        Args:
            actual: Actual input [batch, n_input]
            predicted: Predicted input [batch, n_input]
            
        Returns:
            error: Precision-weighted prediction error [batch, n_input]
        """
        # Raw error
        raw_error = actual - predicted
        
        if self.config.use_spiking:
            # Error neurons receive positive error as excitation
            # and negative error as inhibition
            # Split into E+ and E- populations or use signed current
            if self.config.error_type == ErrorType.SIGNED:
                # Single population, signed error as current
                error_spikes, error_membrane = self.error_neurons(raw_error)
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
        
        Args:
            actual_input: Bottom-up input (sensory or from lower layer)
            representation: Current representation from higher layer
            top_down_prediction: Optional direct prediction from above
            
        Returns:
            error: Prediction error (goes UP to higher layers)
            prediction: Current prediction (for diagnostics)
            representation_update: Suggested update to representation
        """
        batch_size = actual_input.shape[0]
        
        # Initialize state if needed
        if self.state.prediction is None:
            self.reset_state(batch_size)
        
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
        # Outer product: [batch, n_input] x [batch, n_representation]
        # → [batch, n_input, n_representation]
        batch_eligibility = torch.einsum('bi,bj->bij', error, representation)
        
        # Exponential moving average of eligibility
        eligibility_decay = 0.95
        self.state.eligibility = (
            eligibility_decay * self.state.eligibility +
            (1 - eligibility_decay) * batch_eligibility
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
        Update precision (inverse variance) based on recent errors.
        
        Precision should be:
        - High when errors are consistently small (reliable predictions)
        - Low when errors are variable (unreliable input)
        
        This implements a simple variance estimation:
            precision = 1 / (variance(error) + epsilon)
        """
        if self.state.error is None:
            return
        
        # Estimate variance from recent errors
        error_var = self.state.error.var(dim=0)  # Per-input variance
        
        # Update log precision (in log space for stability)
        target_log_precision = -torch.log(error_var + 1e-6)
        
        with torch.no_grad():
            self.log_precision.data = (
                (1 - self.config.precision_learning_rate) * self.log_precision.data +
                self.config.precision_learning_rate * target_log_precision
            )
            
            # Clamp
            min_log = torch.log(torch.tensor(self.config.precision_min))
            max_log = torch.log(torch.tensor(self.config.precision_max))
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
        """Get diagnostic information about the layer."""
        return {
            "prediction_mean": self.state.prediction.mean().item() if self.state.prediction is not None else 0,
            "error_mean": self.state.error.mean().item() if self.state.error is not None else 0,
            "error_abs_mean": self.state.error.abs().mean().item() if self.state.error is not None else 0,
            "precision_mean": self.precision.mean().item(),
            "precision_std": self.precision.std().item(),
            "free_energy": self.get_free_energy().item(),
            "weight_norm": torch.norm(self.W_pred).item(),
        }


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
    
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all layers."""
        for layer in self.layers:
            layer.reset_state(batch_size)
    
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
