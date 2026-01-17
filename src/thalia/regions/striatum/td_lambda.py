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
    STRIATUM_TD_ACCUMULATING,
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
    accumulating: bool = STRIATUM_TD_ACCUMULATING

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

    def update(self, radient: torch.Tensor) -> None:
        """
        Update eligibility traces with current gradient.

        Args:
            gradient: Gradient of value function ∇V(s,a) [n_output, n_input]
                     For basic TD: gradient = outer(1, pre_activity)
                     For action selection: gradient is masked to chosen action
        """
        # Decay existing traces
        self.traces = self.traces * self.decay_factor

        # Add new gradient
        if self.config.accumulating:
            # Accumulating traces: e(t) = γλe(t-1) + ∇V(t)
            self.traces = self.traces + gradient
        else:
            # Replacing traces: e(t) = max(γλe(t-1), ∇V(t))
            self.traces = torch.maximum(self.traces, gradient)

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

    def grow_input(self, n_new: int) -> None:
        """Grow input dimension by adding columns to eligibility traces.

        Expands trace matrix to accommodate new input neurons from upstream
        regions (e.g., cortex growing its output). Existing traces are preserved
        to maintain credit assignment for learned state-action pairs.

        Args:
            n_new: Number of new input neurons to add

        Effects:
            - self.n_input increases by n_new
            - self.traces expands from [n_output, n_input] to [n_output, n_input+n_new]
            - Old traces preserved (existing credit assignment maintained)
            - New columns initialized to zero (no credit yet for new inputs)

        Example:
            >>> traces = TDLambdaTraces(n_output=16, n_input=128, device=device)
            >>> traces.traces[0, 10] = 0.8  # Some existing credit
            >>> traces.grow_input(32)  # Cortex grew by 32 neurons
            >>> assert traces.n_input == 160
            >>> assert traces.traces[0, 10] == 0.8  # Old trace preserved
            >>> assert traces.traces[0, 150] == 0.0  # New trace zero

        Note:
            Called automatically when upstream region grows:
            >>> cortex.grow_output(32)
            >>> striatum.grow_input(32)  # Calls td_lambda.grow_input(32)
        """
        old_traces = self.traces
        self.n_input = self.n_input + n_new

        # Create new traces with expanded input dimension
        self.traces = torch.zeros(self.n_output, self.n_input, device=self.device)

        # Copy old traces (preserve existing credit assignment)
        self.traces[:, : old_traces.shape[1]] = old_traces

        # New columns initialized to zero (no credit yet for new inputs)

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

    def grow_input(self, n_new: int) -> None:
        """Grow input dimension to accommodate upstream region growth.

        Delegates to the underlying TDLambdaTraces instance to expand
        eligibility trace matrices. Maintains existing credit assignment
        for learned state-action pairs.

        Args:
            n_new: Number of new input neurons to add

        Effects:
            - self.n_input increases by n_new
            - self.traces.n_input increases by n_new
            - Eligibility traces expand from [n_output, n_input] to [n_output, n_input+n_new]

        Example:
            >>> manager = TDLambdaManager(n_output=16, n_input=128, device=device)
            >>> # Upstream cortex grows:
            >>> cortex.grow_output(32)
            >>> # Striatum propagates growth:
            >>> manager.grow_input(32)
            >>> assert manager.n_input == 160
            >>> assert manager.traces.n_input == 160

        Note:
            This is typically called from the parent Striatum region's
            grow_input() method during developmental growth or curriculum
            progression.
        """
        self.n_input = self.n_input + n_new
        self.traces.grow_input(n_new)

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

    def compute_update(self, d_error: float) -> torch.Tensor:
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


def compute_n_step_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Compute n-step returns for trajectory.

    G_t^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})

    This is used for batch updates or episodic learning.

    Args:
        rewards: Reward sequence [T]
        values: Value estimates [T]
        gamma: Discount factor
        n: Number of steps to look ahead

    Returns:
        returns: N-step returns [T]
    """
    T = len(rewards)
    returns = torch.zeros(T, device=rewards.device)

    for t in range(T):
        # Compute n-step return from timestep t
        return_val = 0.0

        for k in range(min(n, T - t)):
            # Add discounted reward
            return_val += (gamma**k) * rewards[t + k]

        # Add bootstrap value if episode didn't end
        if t + n < T:
            return_val += (gamma**n) * values[t + n]

        returns[t] = return_val

    return returns


def compute_lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """
    Compute λ-returns for trajectory (for batch/episodic updates).

    G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^(n)

    This is an exponentially-weighted average of all n-step returns.

    Args:
        rewards: Reward sequence [T]
        values: Value estimates [T]
        gamma: Discount factor
        lambda_: Trace decay parameter

    Returns:
        returns: λ-returns [T]
    """
    T = len(rewards)

    # Compute λ-return recursively (backward pass)
    lambda_returns = torch.zeros(T, device=rewards.device)

    # Start from end of episode
    for t in reversed(range(T)):
        if t == T - 1:
            # Terminal state: only immediate reward
            lambda_returns[t] = rewards[t]
        else:
            # G_t^λ = r_t + γ((1-λ)V(s_{t+1}) + λG_{t+1}^λ)
            lambda_returns[t] = rewards[t] + gamma * (
                (1 - lambda_) * values[t + 1] + lambda_ * lambda_returns[t + 1]
            )

    return lambda_returns
