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

from thalia.regions.base import BrainRegion, RegionConfig, LearningRule
from thalia.regions.cortex.layered_cortex import LayeredCortex, LayeredCortexConfig
from thalia.core.diagnostics_mixin import DiagnosticsMixin
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

    # Store original input for learning
    input_spikes: Optional[torch.Tensor] = None

    # Predictive coding
    prediction: Optional[torch.Tensor] = None
    error: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    free_energy: float = 0.0

    # Attention
    attention_weights: Optional[torch.Tensor] = None


class PredictiveCortex(DiagnosticsMixin, BrainRegion):
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
            output = cortex(input_spikes)

        # Learning (after episode)
        metrics = cortex.learn(input_spikes, output_spikes)
    """

    def __init__(self, config: PredictiveCortexConfig):
        """Initialize predictive cortex."""
        self.predictive_config = config
        
        # =====================================================================
        # BASE LAYERED CORTEX (creates the L4→L2/3→L5 microcircuit)
        # =====================================================================
        self.cortex = LayeredCortex(config)

        # Get layer sizes from cortex
        self.l4_size = self.cortex.l4_size
        self.l23_size = self.cortex.l23_size
        self.l5_size = self.cortex.l5_size
        
        # Compute output size (L2/3 + L5)
        self._output_size = self.l23_size + self.l5_size

        # Create parent config for BrainRegion
        parent_config = RegionConfig(
            n_input=config.n_input,
            n_output=self._output_size,
            dt_ms=config.dt_ms,
            device=config.device,
        )
        
        # Call parent init
        super().__init__(parent_config)

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
    
    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS (from BrainRegion)
    # =========================================================================
    
    def _get_learning_rule(self) -> LearningRule:
        """Predictive cortex uses predictive STDP (local errors + spike timing)."""
        return LearningRule.PREDICTIVE_STDP
    
    def _initialize_weights(self) -> torch.Tensor:
        """Weights are managed by internal LayeredCortex."""
        return self.cortex.weights
    
    def _create_neurons(self) -> Any:
        """Neurons are managed by internal LayeredCortex."""
        return self.cortex.neurons

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
        self._last_plasticity_delta = 0.0

    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float = 1.0,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        top_down: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process input through predictive cortex.

        Args:
            input_spikes: Input spike pattern [batch, n_input]
            dt: Time step in ms (for compatibility)
            encoding_mod: Encoding strength modulation (for compatibility)
            retrieval_mod: Retrieval strength modulation (for compatibility)
            top_down: Optional top-down modulation from higher areas [batch, l23_size]
                      NOTE: This is for L2/3 modulation, NOT L4 prediction!
            **kwargs: Additional arguments for compatibility

        Returns:
            output: Output spikes (L2/3 + L5) [batch, l23_size + l5_size]
        """
        batch_size = input_spikes.shape[0] if input_spikes.dim() > 1 else 1
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.shape[-1] == self.predictive_config.n_input, (
            f"PredictiveCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.predictive_config.n_input}."
        )
        if top_down is not None:
            assert top_down.shape[-1] == self.l23_size, (
                f"PredictiveCortex.forward: top_down has shape {top_down.shape} "
                f"but must match l23_size={self.l23_size}. "
                f"top_down is for L2/3 modulation, not L4 prediction."
            )

        # Initialize if needed
        if self.state.l4_spikes is None:
            self.reset_state(batch_size)

        # =====================================================================
        # STEP 1: Standard feedforward through cortex
        # =====================================================================
        # Pass through to the base LayeredCortex with modulation parameters
        cortex_output = self.cortex.forward(
            input_spikes,
            dt=dt,
            encoding_mod=encoding_mod,
            retrieval_mod=retrieval_mod,
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

        # =====================================================================
        # STEP 2: Predictive coding (L5 → L4 prediction, compute error)
        # =====================================================================
        if self.prediction_layer is not None and l5_output is not None:
            # L5 generates prediction of what L4 should receive
            # NOTE: top_down is for L2/3 modulation, NOT for L4 prediction
            # The prediction_layer handles L5→L4 predictions internally
            error, prediction, _ = self.prediction_layer(
                actual_input=l4_output,
                representation=l5_output,
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
            
        # =====================================================================
        # CONTINUOUS LEARNING (plasticity happens as part of forward dynamics)
        # =====================================================================
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
        """Get diagnostic information."""
        diag = {
            "l4_spikes": self.state.l4_spikes.sum().item() if self.state.l4_spikes is not None else 0,
            "l23_spikes": self.state.l23_spikes.sum().item() if self.state.l23_spikes is not None else 0,
            "l5_spikes": self.state.l5_spikes.sum().item() if self.state.l5_spikes is not None else 0,
            "last_plasticity_delta": getattr(self, "_last_plasticity_delta", 0.0),
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

    def get_total_free_energy(self) -> float:
        """Sum of free energy across all areas."""
        total = 0.0
        for area in self.areas:
            if hasattr(area, 'prediction_layer') and area.prediction_layer is not None:
                total += area.prediction_layer.get_free_energy().item()
        return total
