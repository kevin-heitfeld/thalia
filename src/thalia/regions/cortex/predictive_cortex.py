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

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from thalia.regions.cortex.layered_cortex import LayeredCortex, LayeredCortexConfig
from thalia.core.predictive_coding import (
    PredictiveCodingLayer,
    PredictiveCodingConfig,
)
from thalia.core.scalable_attention import (
    ScalableSpikingAttention,
    ScalableAttentionConfig,
    AttentionType,
)
from thalia.core.neuron import LIFNeuron, LIFConfig


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
    
    # Attention integration
    use_attention: bool = True
    attention_type: AttentionType = AttentionType.WINNER_TAKE_ALL
    attention_top_k: int = 16
    n_attention_heads: int = 4


@dataclass
class PredictiveCortexState:
    """State for predictive cortex."""
    # From LayeredCortex
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None
    
    # Predictive coding
    prediction: Optional[torch.Tensor] = None
    error: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    free_energy: float = 0.0
    
    # Attention
    attention_weights: Optional[torch.Tensor] = None


class PredictiveCortex(nn.Module):
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
            output, state = cortex(input_spikes)
            
        # Learning (after episode)
        metrics = cortex.learn()
    """
    
    def __init__(self, config: PredictiveCortexConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # =====================================================================
        # BASE LAYERED CORTEX
        # =====================================================================
        self.cortex = LayeredCortex(config)
        
        # Get layer sizes from cortex
        self.l4_size = self.cortex.l4_size
        self.l23_size = self.cortex.l23_size
        self.l5_size = self.cortex.l5_size
        
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
        # ATTENTION MODULE
        # =====================================================================
        if config.use_attention:
            # Self-attention over L2/3 representations
            self.attention = ScalableSpikingAttention(
                ScalableAttentionConfig(
                    n_queries=self.l23_size,
                    n_keys=self.l23_size,
                    d_model=self.l23_size,
                    n_heads=config.n_attention_heads,
                    attention_type=config.attention_type,
                    top_k=config.attention_top_k,
                    device=config.device,
                )
            )
        else:
            self.attention = None
        
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
    
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all states for new sequence."""
        self.cortex.reset_state(batch_size)
        
        if self.prediction_layer is not None:
            self.prediction_layer.reset_state(batch_size)
        
        if self.attention is not None:
            self.attention.reset_state(batch_size)
        
        self.state = PredictiveCortexState()
        self._total_free_energy = 0.0
        self._timesteps = 0
    
    def forward(
        self,
        input_spikes: torch.Tensor,
        top_down_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PredictiveCortexState]:
        """
        Process input through predictive cortex.
        
        Args:
            input_spikes: Input spike pattern [batch, n_input]
            top_down_signal: Optional top-down modulation from higher areas
            
        Returns:
            output: Output spikes (L2/3 + L5) [batch, l23_size + l5_size]
            state: Current state with predictions, errors, etc.
        """
        batch_size = input_spikes.shape[0]
        
        # Initialize if needed
        if self.state.l4_spikes is None:
            self.reset_state(batch_size)
        
        # =====================================================================
        # STEP 1: Standard feedforward through cortex
        # =====================================================================
        cortex_output = self.cortex(input_spikes)
        
        # Extract layer outputs from cortex state
        l4_output = self.cortex.state.l4_out
        l23_output = self.cortex.state.l23_out
        l5_output = self.cortex.state.l5_out
        
        # Store in state
        self.state.l4_spikes = l4_output
        self.state.l23_spikes = l23_output
        self.state.l5_spikes = l5_output
        
        # =====================================================================
        # STEP 2: Predictive coding (L5 → L4 prediction, compute error)
        # =====================================================================
        if self.prediction_layer is not None and l5_output is not None:
            # L5 generates prediction of what L4 should receive
            error, prediction, _ = self.prediction_layer(
                actual_input=l4_output,
                representation=l5_output,
                top_down_prediction=top_down_signal,
            )
            
            self.state.prediction = prediction
            self.state.error = error
            self.state.precision = self.prediction_layer.precision
            self.state.free_energy = self.prediction_layer.get_free_energy().item()
            
            # Accumulate free energy for monitoring
            self._total_free_energy += self.state.free_energy
            self._timesteps += 1
        
        # =====================================================================
        # STEP 3: Attention over L2/3 (self-attention for context integration)
        # =====================================================================
        if self.attention is not None and l23_output is not None:
            # Add sequence dimension for attention
            l23_seq = l23_output.unsqueeze(1)  # [batch, 1, l23_size]
            
            # Self-attention
            attended, attn_weights = self.attention(l23_seq)
            
            # Remove sequence dimension
            attended = attended.squeeze(1)
            
            self.state.attention_weights = attn_weights
            
            # Modulate L2/3 output with attention
            l23_output = l23_output + attended
        
        # =====================================================================
        # STEP 4: Precision modulation (attention → prediction weights)
        # =====================================================================
        if self.precision_modulator is not None and self.state.attention_weights is not None:
            # Use attention to modulate which inputs to trust
            attn_avg = self.state.attention_weights.mean(dim=1)  # Average over heads
            attn_flat = attn_avg.reshape(batch_size, -1)
            
            # Project to precision modulation
            if attn_flat.shape[-1] == self.l23_size:
                precision_mod = self.precision_modulator(attn_flat)
                precision_mod = torch.sigmoid(precision_mod)  # 0-1 range
                
                # Apply to prediction layer precision (if exists)
                if self.prediction_layer is not None:
                    # Modulate precision based on attention
                    with torch.no_grad():
                        self.prediction_layer.log_precision.data = (
                            self.prediction_layer.log_precision.data +
                            0.01 * torch.log(precision_mod.mean(dim=0) + 1e-6)
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
        
        return output, self.state
    
    def learn(
        self,
        reward_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update weights based on accumulated prediction errors.
        
        This is where learning happens WITHOUT backpropagation!
        
        Args:
            reward_signal: Optional reward for three-factor learning
            
        Returns:
            metrics: Learning metrics for monitoring
        """
        metrics = {}
        
        # Learn prediction weights
        if self.prediction_layer is not None:
            pred_metrics = self.prediction_layer.learn(reward_signal)
            metrics.update({f"pred_{k}": v for k, v in pred_metrics.items()})
        
        # Learn attention weights
        if self.attention is not None and self.state.attention_weights is not None:
            attn_metrics = self.attention.learn_from_attention(
                self.state.attention_weights,
                reward_signal,
            )
            metrics.update({f"attn_{k}": v for k, v in attn_metrics.items()})
        
        # Add summary metrics
        if self._timesteps > 0:
            metrics["avg_free_energy"] = self._total_free_energy / self._timesteps
        
        return metrics
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        diag = {
            "l4_spikes": self.state.l4_spikes.sum().item() if self.state.l4_spikes is not None else 0,
            "l23_spikes": self.state.l23_spikes.sum().item() if self.state.l23_spikes is not None else 0,
            "l5_spikes": self.state.l5_spikes.sum().item() if self.state.l5_spikes is not None else 0,
        }
        
        if self.prediction_layer is not None:
            pred_diag = self.prediction_layer.get_diagnostics()
            diag.update({f"pred_{k}": v for k, v in pred_diag.items()})
        
        if self.attention is not None:
            complexity = self.attention.get_complexity_estimate()
            diag.update({f"attn_{k}": v for k, v in complexity.items()})
        
        return diag
    
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
                use_attention=(i > 0),  # Attention in higher areas
                device=base_config.device,
            )
            self.areas.append(PredictiveCortex(area_config))
        
        self.n_areas = n_areas
    
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all areas."""
        for area in self.areas:
            area.reset_state(batch_size)
    
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
    
    def learn(
        self,
        reward_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Learn at all areas."""
        all_metrics = {}
        for i, area in enumerate(self.areas):
            metrics = area.learn(reward_signal)
            for k, v in metrics.items():
                all_metrics[f"area{i}_{k}"] = v
        return all_metrics
    
    def get_total_free_energy(self) -> float:
        """Sum of free energy across all areas."""
        total = 0.0
        for area in self.areas:
            if area.prediction_layer is not None:
                total += area.prediction_layer.get_free_energy().item()
        return total
