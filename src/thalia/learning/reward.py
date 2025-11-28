"""
Reward-Modulated STDP (R-STDP).

Three-factor learning rule where weight changes require:
1. Pre-synaptic spike
2. Post-synaptic spike  
3. Reward/dopamine signal

This allows credit assignment over time - the eligibility trace
"remembers" which synapses were recently active, and reward
retroactively strengthens those connections.

Reference:
    Izhikevich, E.M. (2007). Solving the distal reward problem through 
    linkage of STDP and dopamine signaling. Cerebral Cortex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class RSTDPConfig:
    """Configuration for reward-modulated STDP.
    
    Attributes:
        tau_plus: LTP time constant in ms
        tau_minus: LTD time constant in ms
        tau_eligibility: Eligibility trace time constant in ms
            How long a synapse "remembers" it was active for reward
        a_plus: LTP amplitude
        a_minus: LTD amplitude
        learning_rate: Base learning rate
        w_min: Minimum weight
        w_max: Maximum weight
        dt: Simulation timestep in ms
    """
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    tau_eligibility: float = 1000.0  # 1 second eligibility window
    a_plus: float = 0.01
    a_minus: float = 0.01
    learning_rate: float = 0.01
    w_min: float = 0.0
    w_max: float = 1.0
    dt: float = 1.0


class RewardModulatedSTDP(nn.Module):
    """Reward-modulated STDP with eligibility traces.
    
    Standard STDP computes instantaneous weight changes based on spike timing.
    R-STDP instead accumulates these into an eligibility trace, and actual
    weight changes only occur when a reward signal arrives.
    
    This enables learning from delayed rewards - the network can learn
    that an action taken 500ms ago led to current reward.
    
    Args:
        n_pre: Number of pre-synaptic neurons
        n_post: Number of post-synaptic neurons
        config: R-STDP configuration
        
    Example:
        >>> rstdp = RewardModulatedSTDP(n_pre=100, n_post=50)
        >>> 
        >>> for t in range(1000):
        ...     # Run network
        ...     pre_spikes = input_layer(...)
        ...     post_spikes = output_layer(...)
        ...     
        ...     # Update eligibility traces
        ...     rstdp.update_traces(pre_spikes, post_spikes)
        ...     
        ...     # When reward arrives, apply it
        ...     if reward_received:
        ...         dw = rstdp.apply_reward(reward=1.0)
        ...         synapse.weight.data += dw
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[RSTDPConfig] = None,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config or RSTDPConfig()
        
        # Decay constants
        self.register_buffer(
            "decay_plus",
            torch.exp(torch.tensor(-self.config.dt / self.config.tau_plus))
        )
        self.register_buffer(
            "decay_minus",
            torch.exp(torch.tensor(-self.config.dt / self.config.tau_minus))
        )
        self.register_buffer(
            "decay_eligibility",
            torch.exp(torch.tensor(-self.config.dt / self.config.tau_eligibility))
        )
        
        # State variables
        self.trace_pre: Optional[torch.Tensor] = None
        self.trace_post: Optional[torch.Tensor] = None
        self.eligibility: Optional[torch.Tensor] = None
        
    def reset(self, batch_size: int = 1) -> None:
        """Reset all traces."""
        device = self.decay_plus.device
        self.trace_pre = torch.zeros(batch_size, self.n_pre, device=device)
        self.trace_post = torch.zeros(batch_size, self.n_post, device=device)
        self.eligibility = torch.zeros(self.n_pre, self.n_post, device=device)
        
    def update_traces(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor
    ) -> None:
        """Update spike traces and eligibility.
        
        Call this every timestep with current spikes.
        
        Args:
            pre_spikes: Pre-synaptic spikes, shape (batch, n_pre)
            post_spikes: Post-synaptic spikes, shape (batch, n_post)
        """
        if self.trace_pre is None:
            self.reset(batch_size=pre_spikes.shape[0])
            
        # Decay traces
        self.trace_pre = self.trace_pre * self.decay_plus
        self.trace_post = self.trace_post * self.decay_minus
        
        # Update traces with new spikes
        self.trace_pre = self.trace_pre + pre_spikes
        self.trace_post = self.trace_post + post_spikes
        
        # Compute instantaneous STDP update
        # LTP: post spike when pre trace is high
        ltp = self.config.a_plus * torch.einsum(
            'bp,bo->po', self.trace_pre, post_spikes
        )
        # LTD: pre spike when post trace is high
        ltd = -self.config.a_minus * torch.einsum(
            'bp,bo->po', pre_spikes, self.trace_post
        )
        
        # Average over batch
        stdp_update = (ltp + ltd) / pre_spikes.shape[0]
        
        # Decay eligibility and add new STDP contribution
        self.eligibility = self.eligibility * self.decay_eligibility + stdp_update
        
    def apply_reward(self, reward: float) -> torch.Tensor:
        """Apply reward signal to get weight update.
        
        Positive reward strengthens eligible synapses (those with positive eligibility).
        Negative reward weakens them (punishment).
        
        Args:
            reward: Reward signal (positive = reward, negative = punishment)
            
        Returns:
            Weight update matrix, shape (n_pre, n_post)
        """
        if self.eligibility is None:
            return torch.zeros(self.n_pre, self.n_post)
            
        # Weight change = learning_rate * reward * eligibility
        dw = self.config.learning_rate * reward * self.eligibility
        
        return dw
    
    def apply_reward_and_update(
        self, 
        weights: torch.Tensor, 
        reward: float
    ) -> torch.Tensor:
        """Apply reward and return updated weights (clamped to bounds).
        
        Args:
            weights: Current weight matrix, shape (n_pre, n_post)
            reward: Reward signal
            
        Returns:
            Updated weights
        """
        dw = self.apply_reward(reward)
        new_weights = weights + dw
        return torch.clamp(new_weights, self.config.w_min, self.config.w_max)
    
    def get_eligibility(self) -> torch.Tensor:
        """Get current eligibility trace for visualization."""
        if self.eligibility is None:
            return torch.zeros(self.n_pre, self.n_post)
        return self.eligibility.clone()
