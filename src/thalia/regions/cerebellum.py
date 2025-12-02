"""
Cerebellum - Supervised Error-Corrective Learning

The cerebellum learns through supervised error signals provided by climbing
fibers from the inferior olive. This enables precise, fast learning of
input-output mappings without trial-and-error exploration.

Key Features:
=============
1. ERROR-CORRECTIVE LEARNING (Delta Rule):
   - Δw ∝ pre × (target - actual)
   - Direct teaching signal, not reward/punishment
   - Can learn arbitrary mappings in few trials

2. CLIMBING FIBER ERROR SIGNAL:
   - Inferior olive computes mismatch between target and actual
   - Climbing fiber activates Purkinje cell → LTD on active parallel fibers
   - Absence of climbing fiber → maintain/strengthen active synapses

3. PRECISE TIMING:
   - Cerebellum is master of timing and coordination
   - Can learn precise temporal patterns
   - Sub-millisecond precision in motor control

4. FAST LEARNING:
   - Unlike RL (needs many trials), cerebellum can learn in 1-10 trials
   - Supervised signal provides direct gradient information

Biological Basis:
=================
- Marr (1969) and Albus (1971): Cerebellar learning theory
- Parallel fibers (inputs) → Purkinje cells (outputs)
- Climbing fibers carry error/teaching signals
- LTD at parallel fiber-Purkinje synapses when climbing fiber active

When to Use:
============
- You have explicit target outputs (labels)
- You want to learn arbitrary input→output mappings
- You need fast learning (few trials)
- Precise timing is important
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    LearningRule,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class CerebellumConfig(RegionConfig):
    """Configuration specific to cerebellar regions."""

    learning_rate_ltp: float = 0.02   # Rate for strengthening correct
    learning_rate_ltd: float = 0.02   # Rate for weakening incorrect
    error_threshold: float = 0.01     # Minimum error to trigger learning
    temporal_window_ms: float = 10.0  # Window for coincidence detection
    soft_bounds: bool = True
    input_trace_tau_ms: float = 20.0  # Eligibility trace decay


class ClimbingFiberSystem:
    """Climbing fiber error signaling system.

    Climbing fiber activity means: "You got it WRONG"
    Absence means: "You got it RIGHT (or no feedback)"

    The error signal: target - actual
    - Positive: Should have fired but didn't → strengthen inputs
    - Negative: Fired but shouldn't have → weaken inputs
    """

    def __init__(self, n_output: int, device: str = "cpu"):
        self.n_output = n_output
        self.device = torch.device(device)
        self.error = torch.zeros(n_output, device=self.device)

    def compute_error(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute error signal (climbing fiber activity)."""
        if actual.dim() == 1:
            actual = actual.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        self.error = target - actual
        return self.error

    def reset(self) -> None:
        self.error = torch.zeros(self.n_output, device=self.device)


class Cerebellum(BrainRegion):
    """Cerebellar region with supervised error-corrective learning.

    Implements the cerebellar learning rule:
    - Error signal = target - actual (from climbing fibers)
    - Weight change = learning_rate × input × error
    - This is essentially the delta rule / perceptron learning
    """

    def __init__(self, config: RegionConfig):
        if not isinstance(config, CerebellumConfig):
            config = CerebellumConfig(
                n_input=config.n_input,
                n_output=config.n_output,
                neuron_type=config.neuron_type,
                learning_rate=config.learning_rate,
                w_max=config.w_max,
                w_min=config.w_min,
                target_firing_rate_hz=config.target_firing_rate_hz,
                dt_ms=config.dt_ms,
                device=config.device,
            )

        self.cerebellum_config: CerebellumConfig = config  # type: ignore
        super().__init__(config)

        self.climbing_fiber = ClimbingFiberSystem(
            n_output=config.n_output,
            device=config.device,
        )

        self.input_trace = torch.zeros(config.n_input, device=self.device)
        self.input_trace_tau = self.cerebellum_config.input_trace_tau_ms

    def _get_learning_rule(self) -> LearningRule:
        return LearningRule.ERROR_CORRECTIVE

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights with small uniform values plus noise."""
        weights = torch.ones(self.config.n_output, self.config.n_input)
        weights = weights * self.config.w_max * 0.1
        weights = weights + torch.randn_like(weights) * 0.02
        return weights.clamp(self.config.w_min, self.config.w_max).to(self.device)

    def _create_neurons(self) -> ConductanceLIF:
        """Create Purkinje-like neurons."""
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=3.0, tau_I=8.0,  # Faster dynamics for precise timing
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def forward(
        self,
        input_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input through cerebellar circuit."""
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        if self.neurons.membrane is None:
            self.neurons.reset_state(input_spikes.shape[0])

        # Update input trace for temporal credit assignment
        decay = 1.0 - self.config.dt_ms / self.input_trace_tau
        self.input_trace = self.input_trace * decay + input_spikes.squeeze()

        # Compute synaptic input
        g_exc = torch.matmul(input_spikes, self.weights.T)

        # Forward through neurons
        output_spikes, _ = self.neurons(g_exc, None)

        self.state.spikes = output_spikes
        self.state.t += 1

        return output_spikes

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_neuron: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply error-corrective learning (delta rule).

        The key insight: we update weights to ALL neurons, not just those that fired:
        - Target neuron: Gets LTP from active inputs
        - Wrong neurons that fired: Get LTD from active inputs
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)
        if output_spikes.dim() == 1:
            output_spikes = output_spikes.unsqueeze(0)

        # Build target tensor if target_neuron provided
        if target is None and target_neuron is not None:
            target = torch.zeros(1, self.config.n_output, device=self.device)
            target[0, target_neuron] = 1.0

        if target is None:
            return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}

        if target.dim() == 1:
            target = target.unsqueeze(0)

        # Compute error via climbing fiber system
        error = self.climbing_fiber.compute_error(output_spikes, target)

        if error.abs().max() < self.cerebellum_config.error_threshold:
            return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}

        # Delta rule: Δw = lr × error × input
        error_squeezed = error.squeeze()
        input_squeezed = self.input_trace

        # Handle n_output=1 case where squeeze makes 0-D tensor
        if error_squeezed.dim() == 0:
            error_squeezed = error_squeezed.unsqueeze(0)

        dw = torch.outer(error_squeezed, input_squeezed)
        dw = dw * self.config.learning_rate

        # Soft bounds
        if self.cerebellum_config.soft_bounds:
            headroom = (self.config.w_max - self.weights) / self.config.w_max
            footroom = (self.weights - self.config.w_min) / self.config.w_max

            ltp_mask = dw > 0
            dw[ltp_mask] = dw[ltp_mask] * headroom[ltp_mask].clamp(0, 1)

            ltd_mask = dw < 0
            dw[ltd_mask] = dw[ltd_mask] * footroom[ltd_mask].clamp(0, 1)

        old_weights = self.weights.clone()
        self.weights = (self.weights + dw).clamp(self.config.w_min, self.config.w_max)

        actual_dw = self.weights - old_weights
        ltp = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
        ltd = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0

        return {
            "error": error.abs().mean().item(),
            "ltp": ltp,
            "ltd": ltd,
            "net_change": ltp + ltd,
        }

    def learn_from_phase(
        self,
        input_trace: torch.Tensor,
        winner: int,
        target: int,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convenience method for phase-based supervised learning."""
        lr = learning_rate or self.config.learning_rate

        if winner == target:
            # Correct - mild reinforcement
            dw = torch.zeros_like(self.weights)
            dw[target, :] = lr * 0.1 * input_trace
        else:
            # Wrong - correct both neurons
            dw = torch.zeros_like(self.weights)
            dw[target, :] = lr * input_trace      # LTP for target
            dw[winner, :] = -lr * input_trace     # LTD for wrong winner

        if self.cerebellum_config.soft_bounds:
            headroom = (self.config.w_max - self.weights) / self.config.w_max
            footroom = (self.weights - self.config.w_min) / self.config.w_max
            dw = torch.where(dw > 0, dw * headroom.clamp(0, 1), dw * footroom.clamp(0, 1))

        old_weights = self.weights.clone()
        self.weights = (self.weights + dw).clamp(self.config.w_min, self.config.w_max)

        actual_dw = self.weights - old_weights
        ltp = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
        ltd = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0

        return {"correct": winner == target, "ltp": ltp, "ltd": ltd}

    def reset(self) -> None:
        super().reset()
        self.input_trace.zero_()
        self.climbing_fiber.reset()
        if self.neurons is not None:
            self.neurons.reset_state(1)
