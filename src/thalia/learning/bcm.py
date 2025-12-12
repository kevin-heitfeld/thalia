"""
BCM (Bienenstock-Cooper-Munro) Learning Rule.

The BCM rule provides a sliding threshold for synaptic modification that
automatically adjusts based on postsynaptic activity history. This prevents
runaway potentiation/depression and enables stable, competitive learning.

Key features:
1. SLIDING THRESHOLD (θ_M): Adapts based on recent postsynaptic activity
   - θ_M increases when neuron is highly active → harder to potentiate
   - θ_M decreases when neuron is quiet → easier to potentiate

2. BIDIRECTIONAL PLASTICITY:
   - Post > θ_M: LTP (strengthen active synapses)
   - Post < θ_M: LTD (weaken active synapses)
   - Post = 0: No change (requires postsynaptic activity)

3. SELECTIVITY: High-firing inputs that drive postsynaptic activity above
   threshold get strengthened, while low-correlation inputs get weakened.

Biological basis:
- NMDA receptor dynamics provide a natural threshold mechanism
- Metaplasticity adjusts the LTP/LTD balance based on activity history
- Calcium dynamics: high [Ca²⁺] → LTP, moderate [Ca²⁺] → LTD

The BCM function:
    φ(c, θ_M) = c(c - θ_M)

    where c is postsynaptic activity and θ_M is the sliding threshold.
    - c > θ_M: φ > 0 → LTP
    - 0 < c < θ_M: φ < 0 → LTD
    - c = 0: φ = 0 → no change

Threshold dynamics:
    τ_θ * dθ_M/dt = c² - θ_M

    or equivalently: θ_M = E[c²] (average squared activity)

References:
- Bienenstock, Cooper & Munro (1982): Theory for the development of neuron
  selectivity: orientation specificity in visual cortex
- Intrator & Cooper (1992): Objective function formulation of the BCM theory
- Bear (2003): Bidirectional synaptic plasticity: from theory to reality

Integration with STDP:
    BCM can modulate STDP by scaling the learning rate based on how far
    the postsynaptic activity is from the sliding threshold. This creates
    a natural metaplasticity that stabilizes learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from thalia.core.errors import ComponentError
from thalia.core.learning_constants import (
    LEARNING_RATE_BCM,
    TAU_BCM_THRESHOLD,
)


@dataclass
class BCMConfig:
    """Configuration for BCM learning rule.

    Attributes:
        tau_theta: Time constant for sliding threshold adaptation (ms)
            Larger values = slower threshold adaptation = more stable
            Typical: 1000-10000ms (seconds to minutes in biology)

        theta_init: Initial value of sliding threshold
            Should be set to roughly the expected average squared activity.

        theta_min: Minimum allowed threshold (prevents pathological states)
            If threshold goes too low, any activity causes LTP.

        theta_max: Maximum allowed threshold (prevents complete shutdown)
            If threshold goes too high, nothing can cause LTP.

        learning_rate: Base learning rate for weight updates

        p: Power for threshold computation (default: 2 for c²)
            Original BCM uses p=2, but p=1 (linear) is sometimes used.

        dt: Simulation timestep (ms)
    """
    tau_theta: float = TAU_BCM_THRESHOLD    # Slow adaptation (5 seconds)
    theta_init: float = 0.01                # Initial threshold
    theta_min: float = 1e-6                 # Minimum threshold
    theta_max: float = 1.0                  # Maximum threshold
    learning_rate: float = LEARNING_RATE_BCM  # Base learning rate
    p: float = 2.0                          # Power for threshold (c^p)
    dt: float = 1.0                         # Timestep

    @property
    def decay_theta(self) -> float:
        """Decay factor for threshold EMA per timestep."""
        return torch.exp(torch.tensor(-self.dt / self.tau_theta)).item()


class BCMRule(nn.Module):
    """BCM learning rule with sliding threshold.

    Computes weight updates based on the BCM plasticity function:
        Δw = η * pre * φ(post, θ_M)

    where φ(c, θ) = c(c - θ) is the BCM function.

    The sliding threshold θ_M tracks the running average of post²:
        θ_M(t) = decay * θ_M(t-1) + (1-decay) * post(t)²

    Args:
        n_post: Number of postsynaptic neurons
        config: BCM configuration parameters

    Example:
        >>> bcm = BCMRule(n_post=100)
        >>> bcm.reset_state(batch_size=1)
        >>>
        >>> for t in range(1000):
        ...     pre_spikes = ...   # (batch, n_pre)
        ...     post_spikes = ...  # (batch, n_post)
        ...
        ...     # Compute BCM modulation factor
        ...     bcm_factor = bcm.compute_phi(post_spikes)  # (batch, n_post)
        ...
        ...     # Use with STDP or Hebbian learning:
        ...     # dw = stdp_dw * bcm_factor  (per-synapse)
        ...
        ...     # Update BCM threshold
        ...     bcm.update_threshold(post_spikes)
    """

    def __init__(
        self,
        n_post: int,
        config: Optional[BCMConfig] = None,
    ):
        super().__init__()
        self.n_post = n_post
        self.config = config or BCMConfig()

        # Register constants
        self.register_buffer(
            "decay_theta",
            torch.tensor(self.config.decay_theta, dtype=torch.float32)
        )
        self.register_buffer(
            "one_minus_decay",
            torch.tensor(1.0 - self.config.decay_theta, dtype=torch.float32)
        )

        # Sliding threshold (per neuron)
        self.theta: Optional[torch.Tensor] = None

        # Activity accumulator for smoother threshold updates
        self.activity_trace: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Reset BCM state to batch_size=1."""
        device = self.decay_theta.device
        batch_size = 1

        # Threshold is per-neuron (not batched)
        self.theta = torch.full(
            (self.n_post,),
            self.config.theta_init,
            device=device,  # type: ignore[arg-type]
            dtype=torch.float32
        )

        # Activity trace for smoother updates
        self.activity_trace = torch.zeros(
            (self.n_post,),
            device=device,  # type: ignore[arg-type]
            dtype=torch.float32
        )

    def compute_phi(
        self,
        post_activity: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute BCM plasticity function φ(c, θ).

        The BCM function determines the direction and magnitude of plasticity:
            φ(c, θ) = c(c - θ)

        - c > θ: φ > 0 (LTP)
        - 0 < c < θ: φ < 0 (LTD)
        - c = 0: φ = 0 (no change)

        Args:
            post_activity: Postsynaptic activity (spikes or rates),
                          shape: (n_post,) per-neuron 1D tensor (ADR-005: No Batch Dimension)
            normalize: If True, normalize by theta for more stable learning

        Returns:
            BCM factor φ, shape (n_post,) per-neuron modulation
            Positive = LTP direction, Negative = LTD direction
        """
        # Initialize theta if needed (per-neuron 1D tensor)
        if self.theta is None:
            self.reset_state()

        # BCM function: φ(c, θ) = c(c - θ)
        # With normalize: φ(c, θ) = c(c - θ) / θ for scale invariance
        phi = post_activity * (post_activity - self.theta)

        if normalize:
            # Normalize by theta for scale-invariant learning
            # Add small epsilon to prevent division by zero
            phi = phi / (self.theta + 1e-8)

        return phi

    def update_threshold(self, post_activity: torch.Tensor) -> None:
        """Update sliding threshold based on postsynaptic activity.

        The threshold tracks the running average of activity^p:
            θ_M(t) = decay * θ_M(t-1) + (1-decay) * c^p

        where p=2 for classic BCM (tracks squared activity).

        Args:
            post_activity: Postsynaptic activity (spikes or rates),
                          shape: (n_post,) per-neuron 1D tensor (ADR-005: No Batch Dimension)
        """
        # Initialize theta if needed (per-neuron 1D tensor)
        if self.theta is None:
            self.reset_state()

        # Compute activity^p per neuron
        c_p = post_activity.pow(self.config.p)  # (n_post,)

        # Update threshold with EMA (per-neuron)
        self.theta = (
            self.decay_theta * self.theta +
            self.one_minus_decay * c_p
        )

        # Clamp to valid range
        self.theta = torch.clamp(
            self.theta,
            self.config.theta_min,
            self.config.theta_max
        )

    def forward(
        self,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        update_theta: bool = True,
    ) -> torch.Tensor:
        """Compute BCM weight update.

        Args:
            pre_activity: Presynaptic activity, shape (batch, n_pre)
            post_activity: Postsynaptic activity, shape (batch, n_post)
            update_theta: Whether to update the sliding threshold

        Returns:
            Weight update matrix, shape (n_pre, n_post)
            Apply as: weights += learning_rate * dw
        """
        # Compute BCM modulation
        phi = self.compute_phi(post_activity, normalize=True)

        # Hebbian outer product modulated by BCM
        # dw[i,j] = pre[i] * phi[j] = pre[i] * post[j] * (post[j] - theta[j])
        # Averaged over batch
        dw = torch.einsum('bi,bj->ij', pre_activity, phi) / pre_activity.shape[0]

        # Scale by learning rate
        dw = self.config.learning_rate * dw

        # Update threshold
        if update_theta:
            self.update_threshold(post_activity)

        return dw

    def apply_to_synapse(
        self,
        synapse: nn.Module,
        pre_activity: torch.Tensor,
        post_activity: torch.Tensor,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ) -> torch.Tensor:
        """Compute and apply BCM update to a synapse.

        Args:
            synapse: Synapse module with 'weight' parameter
            pre_activity: Presynaptic spikes/rates
            post_activity: Postsynaptic spikes/rates
            w_min: Minimum weight bound
            w_max: Maximum weight bound

        Returns:
            Weight update that was applied
        """
        dw = self(pre_activity, post_activity)

        with torch.no_grad():
            synapse.weight.data = torch.clamp(
                synapse.weight.data + dw,
                w_min,
                w_max
            )

        return dw

    def get_threshold(self) -> torch.Tensor:
        """Get current sliding threshold."""
        if self.theta is None:
            raise ComponentError(
                "BCM",
                "BCM state not initialized. Call reset_state() first."
            )
        return self.theta.clone()

    def get_state(self) -> dict[str, Optional[torch.Tensor]]:
        """Get current BCM state for analysis/saving."""
        return {
            "theta": self.theta.clone() if self.theta is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"BCMRule(n_post={self.n_post}, "
            f"τ_θ={self.config.tau_theta:.0f}ms, "
            f"θ_init={self.config.theta_init:.4f})"
        )
