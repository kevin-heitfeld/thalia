"""
Spike-Timing-Dependent Plasticity (STDP) implementation.

STDP is a biologically-plausible learning rule where synaptic strength
changes based on the relative timing of pre- and post-synaptic spikes.

- Pre before post (Δt > 0): Potentiation (strengthen connection)
- Post before pre (Δt < 0): Depression (weaken connection)

Weight change formula:
    Δw = A+ * exp(-Δt/τ+)  if Δt > 0
    Δw = -A- * exp(Δt/τ-)  if Δt < 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class STDPConfig:
    """Configuration for STDP learning rule.
    
    Attributes:
        tau_plus: Time constant for potentiation (ms)
        tau_minus: Time constant for depression (ms)
        a_plus: Learning rate for potentiation
        a_minus: Learning rate for depression
        w_min: Minimum weight
        w_max: Maximum weight
    """
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.01
    a_minus: float = 0.01
    w_min: float = 0.0
    w_max: float = 1.0


class STDP(nn.Module):
    """STDP learning rule for synapse weight updates.
    
    Uses eligibility traces to track recent spike activity and
    compute weight updates based on spike timing relationships.
    
    Args:
        n_pre: Number of pre-synaptic neurons
        n_post: Number of post-synaptic neurons
        config: STDP configuration parameters
        
    Example:
        >>> stdp = STDP(n_pre=100, n_post=50)
        >>> 
        >>> for t in range(1000):
        ...     # Your simulation loop
        ...     pre_spikes = ...
        ...     post_spikes = ...
        ...     
        ...     # Compute weight updates
        ...     dw = stdp(pre_spikes, post_spikes)
        ...     
        ...     # Apply to your synapse
        ...     synapse.weight.data += dw
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[STDPConfig] = None,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config or STDPConfig()
        
        # Decay factors
        self.register_buffer(
            "decay_plus",
            torch.tensor(1 - 1/self.config.tau_plus, dtype=torch.float32)
        )
        self.register_buffer(
            "decay_minus", 
            torch.tensor(1 - 1/self.config.tau_minus, dtype=torch.float32)
        )
        
        # Eligibility traces
        self.trace_pre: Optional[torch.Tensor] = None
        self.trace_post: Optional[torch.Tensor] = None
        
    def reset_traces(self, batch_size: int = 1) -> None:
        """Reset eligibility traces to zero."""
        device = self.decay_plus.device
        self.trace_pre = torch.zeros(batch_size, self.n_pre, device=device)
        self.trace_post = torch.zeros(batch_size, self.n_post, device=device)
        
    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute STDP weight update.
        
        Args:
            pre_spikes: Pre-synaptic spikes, shape (batch, n_pre)
            post_spikes: Post-synaptic spikes, shape (batch, n_post)
            
        Returns:
            Weight update matrix, shape (n_pre, n_post)
        """
        # Initialize traces if needed
        if self.trace_pre is None:
            self.reset_traces(batch_size=pre_spikes.shape[0])
            
        # Update traces with decay
        self.trace_pre = self.trace_pre * self.decay_plus + pre_spikes
        self.trace_post = self.trace_post * self.decay_minus + post_spikes
        
        # Compute weight updates
        # Potentiation: post spike with pre trace
        # (batch, n_pre).T @ (batch, n_post) -> (n_pre, n_post)
        dw_plus = self.config.a_plus * torch.einsum(
            'bi,bj->ij', self.trace_pre, post_spikes
        )
        
        # Depression: pre spike with post trace  
        dw_minus = self.config.a_minus * torch.einsum(
            'bi,bj->ij', pre_spikes, self.trace_post
        )
        
        # Net change (averaged over batch)
        dw = (dw_plus - dw_minus) / pre_spikes.shape[0]
        
        return dw
    
    def apply_to_synapse(
        self,
        synapse: nn.Module,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute and apply STDP update to a synapse.
        
        Args:
            synapse: Synapse module with 'weight' parameter
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes
            
        Returns:
            The weight update that was applied
        """
        dw = self(pre_spikes, post_spikes)
        
        # Apply update with bounds
        with torch.no_grad():
            synapse.weight.data = torch.clamp(
                synapse.weight.data + dw,
                self.config.w_min,
                self.config.w_max
            )
            
        return dw
    
    def __repr__(self) -> str:
        return (
            f"STDP({self.n_pre} -> {self.n_post}, "
            f"τ+={self.config.tau_plus}, τ-={self.config.tau_minus})"
        )
