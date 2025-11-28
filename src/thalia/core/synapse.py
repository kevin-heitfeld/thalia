"""
Synapse models for spike transmission between neurons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SynapseConfig:
    """Configuration for synapse parameters.
    
    Attributes:
        delay: Synaptic delay in timesteps (default: 1)
        w_min: Minimum weight (default: 0.0)
        w_max: Maximum weight (default: 1.0)
    """
    delay: int = 1
    w_min: float = 0.0
    w_max: float = 1.0


class Synapse(nn.Module):
    """Basic synapse with weighted connections and optional delays.
    
    Implements connections between pre-synaptic and post-synaptic neuron groups.
    Supports sparse or dense connectivity patterns.
    
    Args:
        n_pre: Number of pre-synaptic neurons
        n_post: Number of post-synaptic neurons  
        config: Synapse configuration
        connectivity: Connection probability (1.0 = fully connected)
        
    Example:
        >>> synapse = Synapse(n_pre=100, n_post=50, connectivity=0.1)
        >>> pre_spikes = torch.zeros(1, 100)
        >>> pre_spikes[0, [5, 10, 15]] = 1  # Some neurons spiked
        >>> post_current = synapse(pre_spikes)  # Shape: (1, 50)
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[SynapseConfig] = None,
        connectivity: float = 1.0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config or SynapseConfig()
        self.connectivity = connectivity
        
        # Initialize weights
        weights = torch.rand(n_pre, n_post) * 0.5
        
        # Apply connectivity mask
        if connectivity < 1.0:
            mask = torch.rand(n_pre, n_post) < connectivity
            weights = weights * mask.float()
            self.register_buffer("mask", mask)
        else:
            self.register_buffer("mask", torch.ones(n_pre, n_post, dtype=torch.bool))
            
        self.weight = nn.Parameter(weights)
        
        # Delay buffer for spike transmission
        if self.config.delay > 1:
            self.register_buffer(
                "delay_buffer",
                torch.zeros(self.config.delay, n_pre)
            )
            self.delay_idx = 0
        else:
            self.delay_buffer = None
            
    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Transmit spikes through synapses.
        
        Args:
            pre_spikes: Pre-synaptic spikes, shape (batch, n_pre)
            
        Returns:
            Post-synaptic current, shape (batch, n_post)
        """
        # Handle delay if configured
        if self.delay_buffer is not None:
            # Store current spikes, retrieve delayed spikes
            delayed_spikes = self.delay_buffer[self.delay_idx].clone()
            self.delay_buffer[self.delay_idx] = pre_spikes[0]  # TODO: handle batch
            self.delay_idx = (self.delay_idx + 1) % self.config.delay
            effective_spikes = delayed_spikes.unsqueeze(0)
        else:
            effective_spikes = pre_spikes
            
        # Clamp weights to bounds
        clamped_weights = torch.clamp(
            self.weight, 
            self.config.w_min, 
            self.config.w_max
        )
        
        # Apply mask and compute post-synaptic current
        effective_weights = clamped_weights * self.mask.float()
        post_current = torch.matmul(effective_spikes, effective_weights)
        
        return post_current
    
    def __repr__(self) -> str:
        return (
            f"Synapse({self.n_pre} -> {self.n_post}, "
            f"connectivity={self.connectivity:.2f})"
        )
