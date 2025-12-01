"""
Striatum - Reinforcement Learning with Three-Factor Rule

The striatum (part of basal ganglia) learns through dopamine-modulated
plasticity, implementing the classic three-factor learning rule for
reinforcement learning.

Key Features:
=============
1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
   - Pre-post activity creates eligibility traces
   - Eligibility alone does NOT cause plasticity
   - Dopamine arriving later converts eligibility to weight change
   - DA burst → LTP, DA dip → LTD, No DA → no learning

2. DOPAMINE as REWARD PREDICTION ERROR:
   - Burst: "Better than expected" → reinforce recent actions
   - Dip: "Worse than expected" → weaken recent actions
   - Baseline: "As expected" → maintain current policy

3. LONG ELIGIBILITY TRACES:
   - Biological tau: 500-2000ms (Yagishita et al., 2014)
   - Allows credit assignment for delayed rewards
   - Synaptic tag persists until dopamine arrives

4. ACTION SELECTION:
   - Winner-take-all competition via lateral inhibition
   - Selected action's synapses become eligible
   - Dopamine retroactively credits/blames the action

Biological Basis:
=================
- Medium Spiny Neurons (MSNs) in striatum
- D1-MSNs (direct pathway): DA → LTP → "Go" signal
- D2-MSNs (indirect pathway): DA → LTD → "No-Go" signal
- Schultz et al. (1997): Dopamine as reward prediction error

When to Use:
============
- Reinforcement learning (reward/punishment, not labels)
- Action selection and habit learning
- Delayed reward credit assignment
- When you want to learn from trial and error
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch

from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    LearningRule,
    NeuromodulatorSystem,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class StriatumConfig(RegionConfig):
    """Configuration specific to striatal regions."""

    # Eligibility trace parameters (biological: 500-2000ms)
    eligibility_tau_ms: float = 1000.0

    # Dopamine parameters
    dopamine_burst: float = 1.0       # Level for positive RPE
    dopamine_dip: float = -0.5        # Level for negative RPE
    dopamine_tau_ms: float = 200.0    # Decay time constant

    # Learning rates
    three_factor_lr: float = 0.02

    # Action selection
    lateral_inhibition: bool = True
    inhibition_strength: float = 0.5

    # Weight constraints
    soft_bounds: bool = True


class DopamineSystem(NeuromodulatorSystem):
    """Dopamine neuromodulatory system for reward prediction error.

    Unlike ACh (which gates magnitude), dopamine determines DIRECTION:
    - Positive dopamine → eligibility becomes LTP
    - Negative dopamine → eligibility becomes LTD
    - Zero dopamine → eligibility decays unused
    """

    def __init__(
        self,
        tau_ms: float = 200.0,
        burst_magnitude: float = 1.0,
        dip_magnitude: float = -0.5,
        device: str = "cpu",
    ):
        super().__init__(tau_ms, device)
        self.burst_magnitude = burst_magnitude
        self.dip_magnitude = dip_magnitude
        self.baseline = 0.0  # Dopamine baseline is 0 (no learning signal)
        self.level = 0.0

    def compute(
        self,
        reward: Optional[float] = None,
        expected_reward: float = 0.0,
        correct: Optional[bool] = None,
        **kwargs: Any,
    ) -> float:
        """Compute dopamine level based on reward prediction error.

        Can be called with either:
        - reward and expected_reward → compute RPE
        - correct (bool) → simple correct/incorrect signal
        """
        if correct is not None:
            # Simple correct/incorrect mode
            self.level = self.burst_magnitude if correct else self.dip_magnitude
        elif reward is not None:
            # RPE mode
            rpe = reward - expected_reward
            if rpe > 0:
                self.level = self.burst_magnitude * min(1.0, rpe)
            elif rpe < 0:
                self.level = self.dip_magnitude * min(1.0, abs(rpe))
            else:
                self.level = 0.0
        else:
            # No signal
            self.level = 0.0

        return self.level

    def decay(self, dt_ms: float) -> None:
        """Decay dopamine toward baseline (0)."""
        decay_factor = 1.0 - dt_ms / self.tau_ms
        self.level = self.level * max(0.0, decay_factor)


class EligibilityTraces:
    """Synapse-specific eligibility traces for three-factor learning.

    Each synapse maintains its own eligibility based on recent pre-post
    coincidence. When dopamine arrives, only eligible synapses are modified.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_ms: float = 1000.0,
        device: str = "cpu",
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_ms = tau_ms
        self.device = torch.device(device)
        self.traces = torch.zeros(n_post, n_pre, device=self.device)

    def update(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        dt_ms: float,
    ) -> None:
        """Update eligibility based on pre-post coincidence."""
        pre = pre_activity.squeeze()
        post = post_activity.squeeze()

        # Decay existing traces
        decay = 1.0 - dt_ms / self.tau_ms
        self.traces = self.traces * decay

        # Add new eligibility for co-active synapses
        # Outer product: which pre-post pairs were co-active
        if post.sum() > 0 and pre.sum() > 0:
            self.traces = self.traces + torch.outer(post, pre)

    def get(self) -> torch.Tensor:
        return self.traces

    def reset(self) -> None:
        self.traces.zero_()


class Striatum(BrainRegion):
    """Striatal region with three-factor reinforcement learning.

    Implements dopamine-modulated learning:
    - Eligibility traces tag recently active synapses
    - Dopamine signal converts eligibility to plasticity
    - No learning without dopamine (unlike Hebbian)
    """

    def __init__(self, config: RegionConfig):
        if not isinstance(config, StriatumConfig):
            config = StriatumConfig(
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

        self.striatum_config: StriatumConfig = config  # type: ignore
        super().__init__(config)

        # Eligibility traces (synapse-specific)
        self.eligibility = EligibilityTraces(
            n_pre=config.n_input,
            n_post=config.n_output,
            tau_ms=self.striatum_config.eligibility_tau_ms,
            device=config.device,
        )

        # Dopamine system
        self.dopamine = DopamineSystem(
            tau_ms=self.striatum_config.dopamine_tau_ms,
            burst_magnitude=self.striatum_config.dopamine_burst,
            dip_magnitude=self.striatum_config.dopamine_dip,
            device=config.device,
        )

        # Recent spikes for lateral inhibition
        self.recent_spikes = torch.zeros(config.n_output, device=self.device)

        # Track last action for credit assignment
        self.last_action: Optional[int] = None

    def _get_learning_rule(self) -> LearningRule:
        return LearningRule.THREE_FACTOR

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize with small positive weights."""
        weights = torch.rand(self.config.n_output, self.config.n_input)
        weights = weights * self.config.w_max * 0.2
        return weights.clamp(self.config.w_min, self.config.w_max).to(self.device)

    def _create_neurons(self) -> ConductanceLIF:
        """Create MSN-like neurons."""
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=5.0, tau_I=15.0,  # Slower inhibition for action selection
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def forward(
        self,
        input_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input and select action."""
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)

        if self.neurons.membrane is None:
            self.neurons.reset_state(input_spikes.shape[0])

        # Compute excitatory input
        g_exc = torch.matmul(input_spikes, self.weights.T)

        # Lateral inhibition
        g_inh = self.recent_spikes.unsqueeze(0) * self.striatum_config.inhibition_strength if self.striatum_config.lateral_inhibition else None

        # Forward through neurons
        output_spikes, _ = self.neurons(g_exc, g_inh)

        # Update eligibility traces
        self.eligibility.update(input_spikes, output_spikes, self.config.dt_ms)

        # Track which action was taken
        if output_spikes.sum() > 0:
            self.last_action = output_spikes.squeeze().argmax().item()

        # Decay dopamine
        self.dopamine.decay(self.config.dt_ms)

        # Update recent spikes
        self.recent_spikes = self.recent_spikes * 0.9 + output_spikes.squeeze()

        self.state.spikes = output_spikes
        self.state.dopamine = self.dopamine.level
        self.state.t += 1

        return output_spikes

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: Optional[float] = None,
        correct: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply three-factor learning rule.

        Δw = learning_rate × eligibility × dopamine

        This is fundamentally different from Hebbian:
        - No dopamine → NO learning (not reduced learning)
        - Dopamine sign determines LTP vs LTD direction
        """
        # Compute dopamine signal
        if reward is not None or correct is not None:
            self.dopamine.compute(reward=reward, correct=correct)

        da_level = self.dopamine.level

        # No learning without dopamine signal
        if abs(da_level) < 0.01:
            return {"dopamine": da_level, "ltp": 0.0, "ltd": 0.0, "net_change": 0.0}

        # Three-factor rule: Δw = lr × eligibility × dopamine
        eligibility = self.eligibility.get()
        dw = self.striatum_config.three_factor_lr * eligibility * da_level

        # Soft bounds
        if self.striatum_config.soft_bounds:
            headroom = (self.config.w_max - self.weights) / self.config.w_max
            footroom = (self.weights - self.config.w_min) / self.config.w_max
            dw = torch.where(dw > 0, dw * headroom.clamp(0, 1), dw * footroom.clamp(0, 1))

        old_weights = self.weights.clone()
        self.weights = (self.weights + dw).clamp(self.config.w_min, self.config.w_max)

        actual_dw = self.weights - old_weights
        ltp = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
        ltd = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0

        return {
            "dopamine": da_level,
            "ltp": ltp,
            "ltd": ltd,
            "net_change": ltp + ltd,
            "eligibility_max": eligibility.max().item(),
        }

    def deliver_reward(self, reward: float, expected: float = 0.0) -> float:
        """Deliver reward signal and trigger learning.

        Convenience method for RL tasks.
        """
        return self.dopamine.compute(reward=reward, expected_reward=expected)

    def reset(self) -> None:
        super().reset()
        self.eligibility.reset()
        self.dopamine.reset()
        self.recent_spikes.zero_()
        self.last_action = None
        if self.neurons is not None:
            self.neurons.reset_state(1)
