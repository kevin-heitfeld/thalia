"""
Synapse models for spike transmission between neurons.

Supports:
- Excitatory and inhibitory synapse types (Dale's law)
- Synaptic delays
- Sparse connectivity patterns
- Weight bounds and normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class SynapseType(Enum):
    """Synapse types following Dale's law."""
    EXCITATORY = "excitatory"  # Positive weights only
    INHIBITORY = "inhibitory"  # Negative weights only
    MIXED = "mixed"            # Both (not biologically realistic, but useful)


@dataclass
class SynapseConfig:
    """Configuration for synapse parameters.

    Attributes:
        synapse_type: Type of synapse (excitatory/inhibitory/mixed)
        delay: Synaptic delay in timesteps (default: 1)
        w_init_mean: Mean of initial weight distribution
        w_init_std: Std of initial weight distribution
        w_min: Minimum weight magnitude (default: 0.0)
        w_max: Maximum weight magnitude (default: 1.0)
    """
    synapse_type: SynapseType = SynapseType.EXCITATORY
    delay: int = 1
    w_init_mean: float = 0.3
    w_init_std: float = 0.1
    w_min: float = 0.0
    w_max: float = 1.0


class Synapse(nn.Module):
    """Synapse with weighted connections, E/I types, and optional delays.

    Implements connections between pre-synaptic and post-synaptic neuron groups.
    Supports sparse or dense connectivity patterns and Dale's law enforcement.

    Args:
        n_pre: Number of pre-synaptic neurons
        n_post: Number of post-synaptic neurons
        config: Synapse configuration
        connectivity: Connection probability (1.0 = fully connected)

    Example:
        >>> # Excitatory synapse (default)
        >>> exc_syn = Synapse(n_pre=80, n_post=100)
        >>>
        >>> # Inhibitory synapse
        >>> inh_config = SynapseConfig(synapse_type=SynapseType.INHIBITORY)
        >>> inh_syn = Synapse(n_pre=20, n_post=100, config=inh_config)
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

        # Initialize weights (positive values, sign applied in forward)
        weights = torch.abs(torch.randn(n_pre, n_post) * self.config.w_init_std + self.config.w_init_mean)
        weights = torch.clamp(weights, self.config.w_min, self.config.w_max)

        # Apply connectivity mask
        if connectivity < 1.0:
            mask = torch.rand(n_pre, n_post) < connectivity
            weights = weights * mask.float()
            self.register_buffer("mask", mask)
        else:
            self.register_buffer("mask", torch.ones(n_pre, n_post, dtype=torch.bool))

        self.weight = nn.Parameter(weights)

        # Sign for Dale's law
        if self.config.synapse_type == SynapseType.EXCITATORY:
            self.register_buffer("sign", torch.tensor(1.0))
        elif self.config.synapse_type == SynapseType.INHIBITORY:
            self.register_buffer("sign", torch.tensor(-1.0))
        else:  # MIXED
            self.register_buffer("sign", torch.tensor(1.0))

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
            batch_size = pre_spikes.shape[0]
            if batch_size == 1:
                delayed_spikes = self.delay_buffer[self.delay_idx].clone()
                self.delay_buffer[self.delay_idx] = pre_spikes[0]
                self.delay_idx = (self.delay_idx + 1) % self.config.delay
                effective_spikes = delayed_spikes.unsqueeze(0)
            else:
                # For batched input, just use current spikes (delay not batched)
                effective_spikes = pre_spikes
        else:
            effective_spikes = pre_spikes

        # Get effective weights with Dale's law and bounds
        if self.config.synapse_type == SynapseType.MIXED:
            # Mixed allows positive and negative
            effective_weights = torch.clamp(
                self.weight,
                -self.config.w_max,
                self.config.w_max
            )
        else:
            # E/I: ensure positive magnitude, apply sign
            effective_weights = self.sign * torch.clamp(
                torch.abs(self.weight),
                self.config.w_min,
                self.config.w_max
            )

        # Apply connectivity mask
        effective_weights = effective_weights * self.mask.float()

        # Compute post-synaptic current: I = W^T @ spikes
        post_current = torch.matmul(effective_spikes, effective_weights)

        return post_current

    def get_effective_weights(self) -> torch.Tensor:
        """Get the effective weight matrix after applying bounds and sign."""
        if self.config.synapse_type == SynapseType.MIXED:
            w = torch.clamp(self.weight, -self.config.w_max, self.config.w_max)
        else:
            w = self.sign * torch.clamp(torch.abs(self.weight), self.config.w_min, self.config.w_max)
        return w * self.mask.float()

    def __repr__(self) -> str:
        type_str = self.config.synapse_type.value[0].upper()  # E, I, or M
        return (
            f"Synapse({self.n_pre}->{self.n_post}, "
            f"type={type_str}, conn={self.connectivity:.2f})"
        )
