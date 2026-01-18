"""
TD(λ) - Multi-Step Credit Assignment for Striatum

Implements TD(λ) learning to bridge longer temporal delays (5-10 seconds)
compared to basic eligibility traces (~1 second).

Key Innovation:
===============
Traditional eligibility traces: e(t) = decay × e(t-1) + 1
    → Single exponential decay, limited temporal bridge

TD(λ): Combines eligibility with multi-step returns
    → δ(t) = r(t) + γV(t+1) - V(t)  [TD error]
    → e(t) = γλe(t-1) + ∇V(t)       [Eligibility trace]
    → ΔV(t) = α × δ(t) × e(t)       [Weight update]

This creates a BRIDGE between:
- Short-term eligibility (local spike timing)
- Long-term returns (delayed rewards)

Biology:
========
- Dopamine neurons show TD-error-like responses (Schultz et al., 1997)
- Eligibility traces in striatum persist 1-2 seconds (Yagishita et al., 2014)
- TD(λ) models how these traces accumulate over multiple timesteps
- λ parameter controls temporal credit assignment window

Parameters:
===========
- λ (lambda): Trace decay rate (0-1)
    * λ=0: Only immediate rewards (TD(0), current implementation)
    * λ=0.9: Bridge ~10 steps (typical value)
    * λ=0.95: Bridge ~20 steps (longer horizon)
    * λ=1.0: Monte Carlo (full episode returns)

- γ (gamma): Discount factor for future rewards (0-1)
    * γ=0.99: Value rewards 100 steps ahead at 37% of immediate
    * γ=0.95: Faster discounting, shorter horizon
    * γ=1.0: No discounting (infinite horizon)

Usage Example:
==============
    # Create TD(λ) learner
    td_lambda = TDLambdaLearner(
        n_actions=10,
        lambda_=0.9,
        gamma=0.99,
        device="cuda"
    )

    # During episode
    for t in range(n_timesteps):
        action = select_action(state)
        td_lambda.update_eligibility(action, gradient=1.0)

    # At reward delivery
    td_error = reward - value_estimate
    weight_update = td_lambda.compute_update(td_error)
    weights += learning_rate * weight_update

References:
===========
- Sutton & Barto (2018): Reinforcement Learning, Chapter 12
- Schultz et al. (1997): Dopamine reward prediction error
- Yagishita et al. (2014): Eligibility traces in striatum
- Doya (2000): Complementary roles of BG and cerebellum

Author: Thalia Project
Date: December 10, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from thalia.constants.regions import (
    STRIATUM_GAMMA,
    STRIATUM_TD_LAMBDA,
    STRIATUM_TD_MIN_TRACE,
)


@dataclass
class TDLambdaConfig:
    """Configuration for TD(λ) learning."""

    # Trace decay rate (0-1)
    lambda_: float = STRIATUM_TD_LAMBDA  # Bridge ~10 timesteps at dt=1ms

    # Discount factor for future rewards (0-1)
    gamma: float = STRIATUM_GAMMA  # Standard RL value

    # Minimum trace value (below this, trace is zeroed for efficiency)
    min_trace: float = STRIATUM_TD_MIN_TRACE

    # Whether to use accumulating traces (True) or replacing traces (False)
    # Accumulating: e(t) = γλe(t-1) + ∇V(t)
    # Replacing: e(t) = max(γλe(t-1), ∇V(t))
    accumulating: bool = True

    # Device
    device: str = "cpu"


class TDLambdaTraces:
    """
    TD(λ) eligibility traces for multi-step credit assignment.

    Maintains eligibility traces that decay with factor (γλ) and accumulate
    based on recent activity. These traces determine how much each synapse
    should be credited/blamed for a reward that arrives later.

    Key difference from basic eligibility traces:
    - Basic: e(t) = (1 - dt/tau) × e(t-1) + activity
    - TD(λ): e(t) = γλ × e(t-1) + gradient

    The γλ product creates longer temporal credit assignment than simple decay.
    """

    def __init__(
        self,
        n_output: int,
        n_input: int,
        config: Optional[TDLambdaConfig] = None,
    ):
        """
        Initialize TD(λ) traces.

        Args:
            n_output: Number of output neurons (actions or MSNs)
            n_input: Number of input neurons
            config: TD(λ) configuration
        """
        self.config = config or TDLambdaConfig()
        self.n_output = n_output
        self.n_input = n_input
        self.device = torch.device(self.config.device)

        # Eligibility traces [n_output, n_input]
        self.traces = torch.zeros(n_output, n_input, device=self.device)

        # Running product γλ for efficient decay
        self.decay_factor = self.config.gamma * self.config.lambda_

    def update(self, grad: torch.Tensor) -> None:
        """Update eligibility traces with new gradient.
        Update eligibility traces with current gradient.

        Args:
            gradient: Gradient of value function ∇V(s,a) [n_output, n_input]
                     For basic TD: gradient = outer(1, pre_activity)
                     For action selection: gradient is masked to chosen action
        """
        # Decay existing traces
        self.traces = self.traces * self.decay_factor

        # Add new gradient (passed as parameter)
        if self.config.accumulating:
            # Accumulating traces: e(t) = γλe(t-1) + ∇V(t)
            self.traces = self.traces + grad
        else:
            # Replacing traces: e(t) = max(γλe(t-1), ∇V(t))
            self.traces = torch.maximum(self.traces, grad)

        # Zero out very small traces for efficiency
        self.traces = torch.where(
            self.traces.abs() < self.config.min_trace, torch.zeros_like(self.traces), self.traces
        )

    def get(self) -> torch.Tensor:
        """Get current eligibility traces."""
        return self.traces

    def to(self, device: torch.device) -> TDLambdaTraces:
        """Move traces to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = device
        self.traces = self.traces.to(device)
        return self

    def reset_state(self) -> None:
        """Reset traces to zero (between episodes)."""
        self.traces.zero_()

    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic information."""
        return {
            "trace_mean": self.traces.abs().mean().item(),
            "trace_max": self.traces.abs().max().item(),
            "trace_nonzero_frac": (self.traces.abs() > self.config.min_trace).float().mean().item(),
        }


class TDLambdaLearner:
    """
    Complete TD(λ) learning system for striatum.

    Manages:
    1. Eligibility traces (credit assignment over time)
    2. Value estimation (state-action values)
    3. TD error computation (reward prediction error)
    4. Weight update computation (learning signal)

    Integrates with existing striatum three-factor learning:
    - Eligibility traces → tag which synapses were active
    - TD(λ) traces → extend temporal credit assignment
    - Dopamine (TD error) → trigger learning
    - Weight update = eligibility × TD_error × learning_rate
    """

    def __init__(
        self,
        n_actions: int,
        n_input: int,
        config: Optional[TDLambdaConfig] = None,
    ):
        """
        Initialize TD(λ) learner.

        Args:
            n_actions: Number of actions
            n_input: Number of input features
            config: TD(λ) configuration
        """
        self.config = config or TDLambdaConfig()
        self.n_actions = n_actions
        self.n_input = n_input
        self.device = torch.device(self.config.device)

        # TD(λ) eligibility traces
        self.traces = TDLambdaTraces(
            n_output=n_actions,
            n_input=n_input,
            config=self.config,
        )

        # Last state value for TD error computation
        self.last_value: Optional[float] = None

        # Statistics
        self._total_updates = 0
        self._cumulative_td_error = 0.0
        self._cumulative_trace_magnitude = 0.0

    def to(self, device: torch.device) -> TDLambdaLearner:
        """Move learner to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = device
        self.traces.to(device)
        return self

    def update_eligibility(
        self,
        action: int,
        pre_activity: torch.Tensor,
    ) -> None:
        """
        Update eligibility traces for chosen action.

        Called during forward pass when action is selected.
        Marks this action's synapses as eligible for credit/blame.

        Args:
            action: Chosen action index
            pre_activity: Input activity pattern [n_input]
        """
        # Create gradient: outer product of action indicator and input
        # This marks all synapses leading to the chosen action
        gradient = torch.zeros(self.n_actions, self.n_input, device=self.device)
        gradient[action] = pre_activity

        # Update traces
        self.traces.update(gradient)

    def compute_td_error(
        self,
        reward: float,
        next_value: float,
        terminal: bool = False,
    ) -> float:
        """
        Compute TD error: δ = r + γV(s') - V(s)

        Args:
            reward: Immediate reward
            next_value: Value of next state V(s')
            terminal: Whether episode ended (if True, next_value ignored)

        Returns:
            td_error: Temporal difference error (RPE signal)
        """
        if self.last_value is None:
            # First timestep, no previous value
            td_error = reward
        else:
            if terminal:
                # Episode ended, no next state
                td_error = reward - self.last_value
            else:
                # Standard TD error: r + γV(s') - V(s)
                td_error = reward + self.config.gamma * next_value - self.last_value

        # Update statistics
        self._total_updates += 1
        self._cumulative_td_error += abs(td_error)

        return td_error

    def compute_update(self, td_error: float) -> torch.Tensor:
        """
        Compute weight update using TD(λ) rule.

        Δw = α × δ × e
        Where:
        - α: learning rate (applied externally)
        - δ: TD error (dopamine signal)
        - e: eligibility trace (which synapses to credit)

        Args:
            td_error: TD error (dopamine signal)

        Returns:
            weight_update: [n_actions, n_input]
        """
        # Weight update = TD error × eligibility traces
        weight_update = td_error * self.traces.get()

        # Update statistics
        self._cumulative_trace_magnitude += self.traces.get().abs().sum().item()

        return weight_update

    def set_last_value(self, value: float) -> None:
        """Store current value estimate for next TD error computation."""
        self.last_value = value

    def reset_episode(self) -> None:
        """Reset traces and value at episode start."""
        self.traces.reset_state()
        self.last_value = None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        trace_diag = self.traces.get_diagnostics()

        avg_td_error = self._cumulative_td_error / max(1, self._total_updates)
        avg_trace_magnitude = self._cumulative_trace_magnitude / max(1, self._total_updates)

        return {
            "total_updates": self._total_updates,
            "avg_td_error": avg_td_error,
            "avg_trace_magnitude": avg_trace_magnitude,
            "lambda": self.config.lambda_,
            "gamma": self.config.gamma,
            **trace_diag,
        }
