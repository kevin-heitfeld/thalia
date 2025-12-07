"""
Eligibility traces for three-factor learning.

Eligibility traces enable credit assignment in reinforcement learning
by maintaining a transient memory of recent synaptic activity that
can be consolidated into long-term changes when dopamine arrives.

Note: Dopamine computation has been centralized at the Brain level
(Brain acts as VTA), so this module only contains EligibilityTraces.
"""

from __future__ import annotations

import torch


class EligibilityTraces:
    """Synapse-specific eligibility traces for three-factor learning.

    Each synapse maintains its own eligibility based on recent pre-post
    coincidence. When dopamine arrives, only eligible synapses are modified.
    
    Biological basis: Calcium transients and signaling cascades persist
    for 500-2000ms after synaptic activity (Yagishita et al., 2014),
    creating a window for dopamine to gate plasticity.
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
        # Decay existing traces (always happens)
        decay = 1.0 - dt_ms / self.tau_ms
        self.traces = self.traces * decay

        # Handle batched input: skip eligibility update for batch_size > 1
        if pre_activity.dim() > 1 and pre_activity.shape[0] > 1:
            return
        if post_activity.dim() > 1 and post_activity.shape[0] > 1:
            return

        pre = pre_activity.squeeze()
        post = post_activity.squeeze()

        # Add new eligibility for co-active synapses
        if post.sum() > 0 and pre.sum() > 0:
            self.traces = self.traces + torch.outer(post, pre)

    def get(self) -> torch.Tensor:
        return self.traces

    def reset_state(self) -> None:
        self.traces.zero_()
