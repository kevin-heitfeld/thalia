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
    
    # REWARD_MODULATED_STDP parameters (spike-based RL)
    learning_rule: LearningRule = LearningRule.THREE_FACTOR
    stdp_tau_ms: float = 20.0         # STDP time constant for traces
    stdp_lr: float = 0.005            # STDP learning rate
    heterosynaptic_ratio: float = 0.3  # LTD/LTP ratio
    
    # Exploration parameters (epsilon-greedy)
    exploration_epsilon: float = 0.3   # Initial exploration rate
    exploration_decay: float = 0.995   # Per-episode decay factor
    min_epsilon: float = 0.01          # Minimum exploration rate


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
        
        # STDP traces for REWARD_MODULATED_STDP (spike-based learning)
        self.input_trace = torch.zeros(config.n_input, device=self.device)
        self.output_trace = torch.zeros(config.n_output, device=self.device)
        self.stdp_eligibility = torch.zeros(
            config.n_output, config.n_input, device=self.device
        )
        
        # Exploration state
        self.current_epsilon = self.striatum_config.exploration_epsilon
        self.exploring = False  # Track if current action was exploratory

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
        explore: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input and select action.
        
        Args:
            input_spikes: Input spike tensor
            explore: If True, use epsilon-greedy exploration. Set to False
                    during evaluation to use greedy action selection.
        """
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
        
        # Epsilon-greedy exploration
        self.exploring = False
        if explore and torch.rand(1).item() < self.current_epsilon:
            # Random action: force a random neuron to spike
            self.exploring = True
            random_action = int(torch.randint(0, self.config.n_output, (1,)).item())
            output_spikes = torch.zeros_like(output_spikes)
            output_spikes[0, random_action] = 1.0

        # Update eligibility traces
        self.eligibility.update(input_spikes, output_spikes, self.config.dt_ms)

        # Track which action was taken
        if output_spikes.sum() > 0:
            self.last_action = int(output_spikes.squeeze().argmax().item())

        # Decay dopamine
        self.dopamine.decay(self.config.dt_ms)

        # Update recent spikes
        self.recent_spikes = self.recent_spikes * 0.9 + output_spikes.squeeze()

        self.state.spikes = output_spikes
        self.state.dopamine = self.dopamine.level
        self.state.t += 1

        return output_spikes
    
    def decay_exploration(self) -> None:
        """Decay epsilon for epsilon-greedy exploration.
        
        Call this at the end of each episode/trial.
        """
        self.current_epsilon = max(
            self.striatum_config.min_epsilon,
            self.current_epsilon * self.striatum_config.exploration_decay
        )

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: Optional[float] = None,
        correct: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply learning rule based on configuration.

        For THREE_FACTOR: Δw = learning_rate × eligibility × dopamine
        For REWARD_MODULATED_STDP: Δw = STDP_eligibility × dopamine (spike-based)
        """
        # Dispatch to appropriate learning rule
        if self.striatum_config.learning_rule == LearningRule.REWARD_MODULATED_STDP:
            return self._reward_modulated_stdp_learn(
                input_spikes, output_spikes, reward, correct
            )
        else:
            return self._three_factor_learn(
                input_spikes, output_spikes, reward, correct
            )
    
    def _three_factor_learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: Optional[float] = None,
        correct: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Classic three-factor learning with rate-coded eligibility.

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

    def _reward_modulated_stdp_learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        reward: Optional[float] = None,
        correct: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Reward-modulated STDP learning (spike-based three-factor rule).
        
        This combines:
        1. Real spiking dynamics with STDP eligibility traces
        2. Dopamine as the third factor that gates learning
        
        Key differences from THREE_FACTOR:
        - Eligibility is computed from spike TIMING, not just correlation
        - STDP creates direction-dependent eligibility (LTP vs LTD based on timing)
        - Dopamine modulates the magnitude, not direction
        
        Weight update: Δw = STDP_eligibility × dopamine × soft_bounds
        
        This is biologically realistic: Yagishita et al. (2014) showed that
        dopamine arriving within ~1-2 seconds of spike correlation converts
        eligibility traces into lasting weight changes.
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)
        if output_spikes.dim() == 1:
            output_spikes = output_spikes.unsqueeze(0)
            
        dt = self.config.dt_ms
        cfg = self.striatum_config
        
        # ======================================================================
        # Step 1: Update STDP eligibility traces from spike timing
        # ======================================================================
        trace_decay = 1.0 - dt / cfg.stdp_tau_ms
        self.input_trace = self.input_trace * trace_decay + input_spikes.squeeze()
        self.output_trace = self.output_trace * trace_decay + output_spikes.squeeze()
        
        # STDP rule:
        # - LTP: post spike with pre trace → strengthen
        # - LTD: pre spike with post trace → weaken
        ltp = torch.outer(output_spikes.squeeze(), self.input_trace)
        ltd = torch.outer(self.output_trace, input_spikes.squeeze())
        
        # Soft bounds: reduce learning as weights approach limits
        w_normalized = (self.weights - self.config.w_min) / (self.config.w_max - self.config.w_min)
        ltp_factor = 1.0 - w_normalized
        ltd_factor = w_normalized
        
        soft_ltp = ltp * ltp_factor
        soft_ltd = ltd * ltd_factor
        
        # Competitive anti-Hebbian: non-spiking neurons get weaker to active inputs
        non_spiking = 1.0 - output_spikes.squeeze()
        anti_hebbian = torch.outer(non_spiking, input_spikes.squeeze()) * w_normalized
        
        # Compute STDP weight change (direction from spike timing)
        stdp_dw = cfg.stdp_lr * (soft_ltp - cfg.heterosynaptic_ratio * soft_ltd - 0.1 * anti_hebbian)
        
        # Accumulate into eligibility trace (long timescale: 500-2000ms)
        eligibility_decay = 1.0 - dt / cfg.eligibility_tau_ms
        self.stdp_eligibility = self.stdp_eligibility * eligibility_decay + stdp_dw
        
        # ======================================================================
        # Step 2: Compute dopamine signal
        # ======================================================================
        if reward is not None or correct is not None:
            self.dopamine.compute(reward=reward, correct=correct)
        
        da_level = self.dopamine.level
        
        # ======================================================================
        # Step 3: Apply dopamine-gated learning
        # ======================================================================
        # Key difference from THREE_FACTOR: STDP eligibility already has direction
        # Dopamine gates the MAGNITUDE of learning, not direction
        # Positive DA → apply eligibility (strengthen what STDP says)
        # Negative DA → apply OPPOSITE of eligibility (weaken what STDP says)
        # Zero DA → no learning (eligibility decays unused)
        
        if abs(da_level) < 0.01:
            # No dopamine = no learning, but track metrics
            return {
                "dopamine": da_level,
                "ltp": soft_ltp.sum().item(),
                "ltd": soft_ltd.sum().item(),
                "net_change": 0.0,
                "eligibility_max": self.stdp_eligibility.abs().max().item(),
            }
        
        # Apply eligibility modulated by dopamine
        # Positive DA: apply eligibility as-is
        # Negative DA: apply opposite of eligibility (reverse STDP direction)
        dw = self.stdp_eligibility * da_level
        
        old_weights = self.weights.clone()
        self.weights = (self.weights + dw).clamp(self.config.w_min, self.config.w_max)
        
        # Note: Unlike Cortex, we DON'T normalize weights here.
        # For RL, we want weights to differentiate (some actions get stronger).
        # Normalization would destroy the learned action preferences.
        
        actual_dw = self.weights - old_weights
        ltp_total = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
        ltd_total = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0
        
        return {
            "dopamine": da_level,
            "ltp": ltp_total,
            "ltd": ltd_total,
            "net_change": ltp_total + ltd_total,
            "eligibility_max": self.stdp_eligibility.abs().max().item(),
        }

    def deliver_reward(self, reward: float, expected: float = 0.0, learn_from_exploration: bool = False) -> Dict[str, Any]:
        """Deliver reward signal and trigger learning.

        For REWARD_MODULATED_STDP, this applies the accumulated eligibility
        modulated by dopamine WITHOUT adding new STDP traces.
        
        Args:
            reward: Reward signal (positive = good, negative = bad)
            expected: Expected reward (for computing RPE)
            learn_from_exploration: If False and last action was exploratory,
                                   skip learning to avoid credit misassignment.
        
        Returns:
            Metrics dict with dopamine level and weight changes.
        """
        # Compute dopamine signal
        da_level = self.dopamine.compute(reward=reward, expected_reward=expected)
        
        # For REWARD_MODULATED_STDP, apply eligibility now
        if self.striatum_config.learning_rule == LearningRule.REWARD_MODULATED_STDP:
            # Skip learning from exploratory actions to avoid credit misassignment
            if self.exploring and not learn_from_exploration:
                return {
                    "dopamine": da_level,
                    "ltp": 0.0,
                    "ltd": 0.0,
                    "net_change": 0.0,
                    "eligibility_max": self.stdp_eligibility.abs().max().item(),
                    "skipped": True,
                    "reason": "exploratory_action",
                }
            
            if abs(da_level) < 0.01:
                return {
                    "dopamine": da_level,
                    "ltp": 0.0,
                    "ltd": 0.0,
                    "net_change": 0.0,
                    "eligibility_max": self.stdp_eligibility.abs().max().item(),
                }
            
            # Apply eligibility modulated by dopamine
            # CRITICAL: Only update the SELECTED action's weights
            # This ensures correct credit assignment
            if self.last_action is not None:
                # Mask eligibility to only the selected action
                action_mask = torch.zeros(self.config.n_output, device=self.device)
                action_mask[self.last_action] = 1.0
                masked_eligibility = self.stdp_eligibility * action_mask.unsqueeze(1)
                
                # Also apply anti-Hebbian to non-selected actions
                # If reward is positive: weaken non-selected actions' weights to active inputs
                non_action_mask = 1.0 - action_mask
                anti_hebbian_strength = 0.3 if da_level > 0 else 0.0
                anti_hebbian = non_action_mask.unsqueeze(1) * self.stdp_eligibility.abs() * anti_hebbian_strength
                
                dw = masked_eligibility * da_level - anti_hebbian * da_level
            else:
                dw = self.stdp_eligibility * da_level
            
            old_weights = self.weights.clone()
            self.weights = (self.weights + dw).clamp(self.config.w_min, self.config.w_max)
            
            actual_dw = self.weights - old_weights
            ltp_total = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
            ltd_total = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0
            
            return {
                "dopamine": da_level,
                "ltp": ltp_total,
                "ltd": ltd_total,
                "net_change": ltp_total + ltd_total,
                "eligibility_max": self.stdp_eligibility.abs().max().item(),
            }
        else:
            return {"dopamine": da_level}

    def reset(self) -> None:
        super().reset()
        self.eligibility.reset()
        self.dopamine.reset()
        self.recent_spikes.zero_()
        self.last_action = None
        # Reset STDP traces for REWARD_MODULATED_STDP
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.stdp_eligibility.zero_()
        self.exploring = False
        if self.neurons is not None:
            self.neurons.reset_state(1)
    
    def reset_exploration(self) -> None:
        """Reset exploration epsilon to initial value.
        
        Call this when starting a new training run.
        """
        self.current_epsilon = self.striatum_config.exploration_epsilon

    # =========================================================================
    # RATE-CODED API
    # =========================================================================

    def encode_rate(self, input_pattern: torch.Tensor) -> torch.Tensor:
        """Rate-coded forward pass through weights."""
        if input_pattern.dim() == 1:
            input_pattern = input_pattern.unsqueeze(0)
        output = torch.matmul(input_pattern, self.weights.t())
        return output.squeeze(0) if output.shape[0] == 1 else output

    def learn_rate(
        self,
        input_pattern: torch.Tensor,
        target_value: float,
        learning_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simple supervised learning for value function."""
        if input_pattern.dim() == 1:
            input_pattern = input_pattern.unsqueeze(0)

        lr = learning_rate if learning_rate is not None else self.striatum_config.three_factor_lr

        # Current prediction
        pred = torch.matmul(input_pattern, self.weights.t())
        error = target_value - pred.squeeze()

        # Gradient update
        dW = lr * error * input_pattern
        with torch.no_grad():
            self.weights.data += dW
            self.weights.data.clamp_(self.striatum_config.w_min, self.striatum_config.w_max)

        return {"error": float(error.item()), "lr": lr}
