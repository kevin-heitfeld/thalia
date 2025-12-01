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


@dataclass
class TripletSTDPConfig:
    """Configuration for Triplet STDP learning rule (Pfister & Gerstner, 2006).
    
    Triplet STDP extends pair-based STDP by considering triplets of spikes,
    which better matches experimental data on frequency-dependent plasticity.
    
    The weight change depends on:
    - Pair terms (A2+, A2-): Standard pair-based STDP
    - Triplet terms (A3+, A3-): Modulated by recent spike history
    
    Attributes:
        tau_plus: Fast time constant for pre-synaptic trace (ms)
        tau_minus: Fast time constant for post-synaptic trace (ms)
        tau_x: Slow time constant for pre-synaptic triplet trace (ms)
        tau_y: Slow time constant for post-synaptic triplet trace (ms)
        a2_plus: Pair-based potentiation amplitude
        a2_minus: Pair-based depression amplitude
        a3_plus: Triplet potentiation amplitude (post-post-pre contribution)
        a3_minus: Triplet depression amplitude (pre-pre-post contribution)
        w_min: Minimum weight
        w_max: Maximum weight
    """
    # Fast traces (like standard STDP)
    tau_plus: float = 16.8    # From Pfister & Gerstner fits to visual cortex data
    tau_minus: float = 33.7
    # Slow traces (for triplet terms)
    tau_x: float = 101.0      # Pre-synaptic slow trace
    tau_y: float = 125.0      # Post-synaptic slow trace
    # Pair amplitudes
    a2_plus: float = 0.005    # Baseline potentiation
    a2_minus: float = 0.007   # Baseline depression
    # Triplet amplitudes (these create frequency dependence)
    a3_plus: float = 0.006    # Extra potentiation from recent post-spikes
    a3_minus: float = 0.002   # Extra depression from recent pre-spikes
    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0


class TripletSTDP(nn.Module):
    """Triplet STDP learning rule (Pfister & Gerstner, 2006).
    
    This implements the "minimal triplet model" which uses four traces:
    - r1 (trace_pre_fast): Fast pre-synaptic trace, decays with tau_plus
    - r2 (trace_pre_slow): Slow pre-synaptic trace, decays with tau_x
    - o1 (trace_post_fast): Fast post-synaptic trace, decays with tau_minus
    - o2 (trace_post_slow): Slow post-synaptic trace, decays with tau_y
    
    Weight update rules:
    - On post-spike: Δw+ = r1 * (A2+ + A3+ * o2)
    - On pre-spike:  Δw- = o1 * (A2- + A3- * r2)
    
    The triplet terms (A3+ * o2, A3- * r2) create frequency dependence:
    - High-frequency post-spikes → larger o2 → more potentiation
    - High-frequency pre-spikes → larger r2 → more depression
    
    Reference:
        Pfister, J.P. & Gerstner, W. (2006). Triplets of Spikes in a Model 
        of Spike Timing-Dependent Plasticity. J. Neurosci. 26(38):9673-9682.
    
    Args:
        n_pre: Number of pre-synaptic neurons
        n_post: Number of post-synaptic neurons
        config: Triplet STDP configuration parameters
    """
    
    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[TripletSTDPConfig] = None,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config or TripletSTDPConfig()
        
        # Decay factors for fast traces
        self.register_buffer(
            "decay_plus",
            torch.tensor(1 - 1/self.config.tau_plus, dtype=torch.float32)
        )
        self.register_buffer(
            "decay_minus", 
            torch.tensor(1 - 1/self.config.tau_minus, dtype=torch.float32)
        )
        # Decay factors for slow traces (triplet terms)
        self.register_buffer(
            "decay_x",
            torch.tensor(1 - 1/self.config.tau_x, dtype=torch.float32)
        )
        self.register_buffer(
            "decay_y",
            torch.tensor(1 - 1/self.config.tau_y, dtype=torch.float32)
        )
        
        # Fast eligibility traces (like standard STDP)
        self.trace_pre_fast: Optional[torch.Tensor] = None   # r1
        self.trace_post_fast: Optional[torch.Tensor] = None  # o1
        # Slow eligibility traces (for triplet terms)
        self.trace_pre_slow: Optional[torch.Tensor] = None   # r2
        self.trace_post_slow: Optional[torch.Tensor] = None  # o2
        
    def reset_traces(self, batch_size: int = 1) -> None:
        """Reset all eligibility traces to zero."""
        device = self.decay_plus.device
        # Fast traces
        self.trace_pre_fast = torch.zeros(batch_size, self.n_pre, device=device)
        self.trace_post_fast = torch.zeros(batch_size, self.n_post, device=device)
        # Slow traces
        self.trace_pre_slow = torch.zeros(batch_size, self.n_pre, device=device)
        self.trace_post_slow = torch.zeros(batch_size, self.n_post, device=device)
        
    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Triplet STDP weight update.
        
        Args:
            pre_spikes: Pre-synaptic spikes, shape (batch, n_pre)
            post_spikes: Post-synaptic spikes, shape (batch, n_post)
            
        Returns:
            Weight update matrix, shape (n_pre, n_post)
        """
        # Initialize traces if needed
        if self.trace_pre_fast is None:
            self.reset_traces(batch_size=pre_spikes.shape[0])
            
        # Decay all traces
        self.trace_pre_fast = self.trace_pre_fast * self.decay_plus
        self.trace_post_fast = self.trace_post_fast * self.decay_minus
        self.trace_pre_slow = self.trace_pre_slow * self.decay_x
        self.trace_post_slow = self.trace_post_slow * self.decay_y
        
        # Compute weight updates BEFORE updating traces with current spikes
        # This ensures we use the trace values from PREVIOUS spikes
        
        # Potentiation: on post-spike, use pre trace
        # Δw+ = r1 * (A2+ + A3+ * o2)
        # The triplet term (A3+ * o2) means more potentiation if post recently fired
        triplet_post_factor = self.config.a2_plus + self.config.a3_plus * self.trace_post_slow
        dw_plus = torch.einsum(
            'bi,bj->ij', 
            self.trace_pre_fast,  # r1: recent pre-spikes
            post_spikes * triplet_post_factor  # weighted by recent post history
        )
        
        # Depression: on pre-spike, use post trace
        # Δw- = o1 * (A2- + A3- * r2)
        # The triplet term (A3- * r2) means more depression if pre recently fired
        triplet_pre_factor = self.config.a2_minus + self.config.a3_minus * self.trace_pre_slow
        dw_minus = torch.einsum(
            'bi,bj->ij',
            pre_spikes * triplet_pre_factor,  # weighted by recent pre history
            self.trace_post_fast  # o1: recent post-spikes
        )
        
        # Update traces with current spikes
        self.trace_pre_fast = self.trace_pre_fast + pre_spikes
        self.trace_post_fast = self.trace_post_fast + post_spikes
        self.trace_pre_slow = self.trace_pre_slow + pre_spikes
        self.trace_post_slow = self.trace_post_slow + post_spikes
        
        # Net change (averaged over batch)
        dw = (dw_plus - dw_minus) / pre_spikes.shape[0]
        
        return dw
    
    def apply_to_synapse(
        self,
        synapse: nn.Module,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute and apply Triplet STDP update to a synapse.
        
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
            f"TripletSTDP({self.n_pre} -> {self.n_post}, "
            f"τ+={self.config.tau_plus}, τ-={self.config.tau_minus}, "
            f"τx={self.config.tau_x}, τy={self.config.tau_y})"
        )
