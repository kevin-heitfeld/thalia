"""
Dopamine system and eligibility traces for three-factor learning.

These classes implement the core neuromodulatory mechanisms for
reinforcement learning in the striatum.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch

from thalia.regions.base import NeuromodulatorSystem


class DopamineSystem(NeuromodulatorSystem):
    """Dopamine neuromodulatory system for reward prediction error.

    Unlike ACh (which gates magnitude), dopamine determines DIRECTION:
    - Positive dopamine → eligibility becomes LTP
    - Negative dopamine → eligibility becomes LTD
    - Zero dopamine → eligibility decays unused

    Features ADAPTIVE RPE NORMALIZATION (when normalize_rpe=True):
    - Maintains running average of |RPE| to adapt to reward statistics
    - Outputs normalized RPE directly in range [-rpe_clip, +rpe_clip]
    - This REPLACES the need for separate burst/dip magnitudes and sensitivities
    - Biologically grounded: DA neurons adapt to reward context (Schultz 2016)

    With normalization, dopamine level is simply the normalized RPE,
    which then gets multiplied by learning_rate in the learning rules.
    This reduces 6 scaling parameters to 3 meta-parameters + 1 learning rate.
    """

    def __init__(
        self,
        tau_ms: float = 200.0,
        burst_magnitude: float = 1.0,
        dip_magnitude: float = -1.0,
        device: str = "cpu",
        # RPE normalization parameters
        normalize_rpe: bool = True,
        rpe_avg_tau: float = 0.9,
        rpe_clip: float = 2.0,
    ):
        super().__init__(tau_ms, device)
        self.burst_magnitude = burst_magnitude
        self.dip_magnitude = dip_magnitude
        self.baseline = 0.0
        self.level = 0.0

        # RPE normalization state
        self.normalize_rpe = normalize_rpe
        self.rpe_avg_tau = rpe_avg_tau
        self.rpe_clip = rpe_clip
        self.avg_abs_rpe = 0.5
        self.rpe_history_count = 0

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

        With normalize_rpe=True:
        - Output is normalized RPE in range [-rpe_clip, +rpe_clip]
        - This directly becomes the learning signal (no separate scaling)
        - Symmetric by construction: same mechanism for + and - RPE

        With normalize_rpe=False:
        - Uses traditional burst/dip magnitudes for scaling
        """
        if correct is not None:
            raw_rpe = 1.0 if correct else -1.0
        elif reward is not None:
            raw_rpe = reward - expected_reward
        else:
            self.level = 0.0
            return self.level

        if self.normalize_rpe:
            # NORMALIZED MODE: Output normalized RPE directly
            abs_rpe = abs(raw_rpe)
            self.rpe_history_count += 1

            # Adaptive smoothing: slower early on for stability
            if self.rpe_history_count < 10:
                alpha = 1.0 / self.rpe_history_count
            else:
                alpha = 1.0 - self.rpe_avg_tau

            self.avg_abs_rpe = self.rpe_avg_tau * self.avg_abs_rpe + alpha * abs_rpe

            # Normalize RPE by running average (with epsilon for stability)
            epsilon = 0.1
            normalized_rpe = raw_rpe / (self.avg_abs_rpe + epsilon)

            # Clip to prevent extreme updates
            self.level = max(-self.rpe_clip, min(self.rpe_clip, normalized_rpe))
        else:
            # LEGACY MODE: Use burst/dip magnitudes
            if raw_rpe > 0:
                self.level = self.burst_magnitude * min(1.0, raw_rpe)
            elif raw_rpe < 0:
                self.level = self.dip_magnitude * min(1.0, abs(raw_rpe))
            else:
                self.level = 0.0

        return self.level

    def get_diagnostics(self) -> Dict[str, float]:
        """Return diagnostic information about RPE normalization state."""
        return {
            "avg_abs_rpe": self.avg_abs_rpe,
            "rpe_history_count": float(self.rpe_history_count),
            "current_level": self.level,
        }

    def decay(self, dt_ms: float) -> None:
        """Decay dopamine toward baseline (0)."""
        decay_factor = 1.0 - dt_ms / self.tau_ms
        self.level = self.level * max(0.0, decay_factor)


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

    def reset(self) -> None:
        self.traces.zero_()
