"""
E/I Balance Regulation: Maintaining Excitation/Inhibition Balance.

Real neural circuits maintain a careful balance between excitation and
inhibition (typically ~4:1 in cortex). This balance is crucial because:
- Too much excitation → seizure-like runaway activity
- Too much inhibition → network silence, no computation

This module provides automatic E/I balance regulation that:
1. Monitors the ratio of excitatory to inhibitory activity
2. Adapts inhibitory gain to maintain target balance
3. Operates on a slow timescale (seconds) to not interfere with fast dynamics

Biological Basis:
=================
- GABAergic interneurons regulate cortical excitability
- Inhibitory plasticity adjusts to maintain E/I balance
- Disrupted E/I balance is implicated in epilepsy, autism, schizophrenia

Key References:
- Vogels et al. (2011): Inhibitory plasticity balances excitation and inhibition
- Turrigiano (2011): Homeostatic synaptic plasticity
- Denève & Machens (2016): Efficient balanced networks

Usage:
======
    from thalia.learning.ei_balance import EIBalanceRegulator, EIBalanceConfig

    regulator = EIBalanceRegulator(EIBalanceConfig(target_ratio=4.0))

    # During forward pass:
    exc_spikes, inh_spikes = cortex(input)
    regulator.update(exc_spikes, inh_spikes)

    # Apply to inhibitory weights:
    inh_weights = inh_weights * regulator.get_inh_scaling()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from thalia.components.coding import compute_firing_rate
from thalia.diagnostics import auto_diagnostics


@dataclass
class EIBalanceConfig:
    """Configuration for E/I balance regulation.

    Attributes:
        target_ratio: Target E/I activity ratio (default: 4.0)
            Cortex typically has ~4:1 E:I ratio.
            Higher values = more excitation allowed.

        tau_balance: Time constant for E/I ratio tracking (ms)
            How quickly the regulator adapts to changes.
            Typical: 1000-10000ms (slow to avoid oscillations).

        adaptation_rate: How aggressively to correct imbalance (per update)
            Higher values = faster correction but risk of oscillation.

        ratio_min: Minimum allowed E/I ratio
            Below this, the network is over-inhibited.

        ratio_max: Maximum allowed E/I ratio
            Above this, seizure risk increases.

        inh_scale_min: Minimum inhibitory scaling factor
            Prevents inhibition from being completely suppressed.

        inh_scale_max: Maximum inhibitory scaling factor
            Prevents runaway inhibition boost.

        dt: Simulation timestep (ms)
    """

    target_ratio: float = 4.0
    tau_balance: float = 5000.0  # 5 second time constant
    adaptation_rate: float = 0.001  # Slow adaptation
    ratio_min: float = 1.0
    ratio_max: float = 10.0
    inh_scale_min: float = 0.1
    inh_scale_max: float = 10.0
    dt: float = 1.0

    @property
    def decay(self) -> float:
        """Decay factor for exponential moving average."""
        return float(torch.exp(torch.tensor(-self.dt / self.tau_balance)).item())


class EIBalanceRegulator(nn.Module):
    """Maintains excitation/inhibition balance through inhibitory gain modulation.

    This regulator tracks the ratio of excitatory to inhibitory activity
    and adjusts a scaling factor for inhibitory weights/currents to
    maintain the target E/I ratio.

    Key insight: Rather than trying to control excitation (which carries
    information), we modulate inhibition (which provides gain control).

    The regulator operates on a slow timescale to:
    1. Not interfere with fast synaptic dynamics
    2. Provide stable, predictable behavior
    3. Allow transient E/I fluctuations (which may be functional)
    """

    def __init__(
        self,
        config: Optional[EIBalanceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config or EIBalanceConfig()
        self._device = device

        # Running averages of activity (as Python floats for stability)
        self._exc_avg: float = 0.1
        self._inh_avg: float = 0.025  # Start at target ratio

        # Current inhibitory scaling factor
        self._inh_scale: float = 1.0

        # History for diagnostics
        self._ratio_history: List[float] = []
        self._scale_history: List[float] = []

    @property
    def device(self) -> torch.device:
        """Get the device (inferred from first tensor if not set)."""
        return self._device or torch.device("cpu")

    def reset_state(self):
        """Reset running averages and scaling."""
        self._exc_avg = 0.1
        self._inh_avg = 0.025
        self._inh_scale = 1.0
        self._ratio_history.clear()
        self._scale_history.clear()

    def compute_ratio(
        self,
        exc_spikes: torch.Tensor,
        inh_spikes: torch.Tensor,
        eps: float = 1e-8,
    ) -> float:
        """Compute instantaneous E/I ratio.

        Args:
            exc_spikes: Excitatory spike tensor (any shape, will be summed)
            inh_spikes: Inhibitory spike tensor (any shape, will be summed)
            eps: Small constant for numerical stability

        Returns:
            E/I ratio (excitatory activity / inhibitory activity)
        """
        exc_rate = compute_firing_rate(exc_spikes)
        inh_rate = compute_firing_rate(inh_spikes)

        return exc_rate / (inh_rate + eps)

    def update(
        self,
        exc_spikes: torch.Tensor,
        inh_spikes: torch.Tensor,
    ) -> float:
        """Update E/I tracking and adjust inhibitory scaling.

        Should be called every timestep with the current excitatory
        and inhibitory spike tensors.

        Args:
            exc_spikes: Excitatory spike tensor
            inh_spikes: Inhibitory spike tensor

        Returns:
            Current inhibitory scaling factor
        """
        cfg = self.config

        # Compute instantaneous rates
        exc_rate = compute_firing_rate(exc_spikes)
        inh_rate = compute_firing_rate(inh_spikes)

        # Update running averages with exponential decay
        decay = cfg.decay
        self._exc_avg = decay * self._exc_avg + (1 - decay) * exc_rate
        self._inh_avg = decay * self._inh_avg + (1 - decay) * inh_rate

        # Compute current ratio from averages (more stable than instantaneous)
        current_ratio = self._exc_avg / (self._inh_avg + 1e-8)

        # Compute error from target
        # Positive error = too much excitation = need more inhibition
        ratio_error = current_ratio - cfg.target_ratio

        # Update inhibitory scaling
        # If E/I too high, increase inhibition (scale > 1)
        # If E/I too low, decrease inhibition (scale < 1)
        scale_adjustment = cfg.adaptation_rate * ratio_error
        self._inh_scale = self._inh_scale * (1 + scale_adjustment)

        # Clamp to safe range
        self._inh_scale = max(cfg.inh_scale_min, min(cfg.inh_scale_max, self._inh_scale))

        # Store history for diagnostics
        self._ratio_history.append(current_ratio)
        self._scale_history.append(self._inh_scale)

        # Keep history bounded
        if len(self._ratio_history) > 10000:
            self._ratio_history = self._ratio_history[-5000:]
            self._scale_history = self._scale_history[-5000:]

        return self._inh_scale

    def get_inh_scaling(self) -> float:
        """Get current inhibitory scaling factor.

        Multiply inhibitory weights or currents by this factor.

        Returns:
            Scaling factor (>1 = boost inhibition, <1 = reduce inhibition)
        """
        return self._inh_scale

    def get_current_ratio(self) -> float:
        """Get current (smoothed) E/I ratio.

        Returns:
            Current E/I ratio from running averages
        """
        return self._exc_avg / (self._inh_avg + 1e-8)

    def get_ratio_error(self) -> float:
        """Get error from target ratio.

        Returns:
            current_ratio - target_ratio (positive = too excitable)
        """
        return self.get_current_ratio() - self.config.target_ratio

    def is_balanced(self, tolerance: float = 0.5) -> bool:
        """Check if E/I ratio is within tolerance of target.

        Args:
            tolerance: Allowed deviation from target ratio

        Returns:
            True if |current_ratio - target_ratio| < tolerance
        """
        return abs(self.get_ratio_error()) < tolerance

    def get_health_status(self) -> str:
        """Get human-readable health status.

        Returns:
            Status string: "balanced", "over-excited", or "over-inhibited"
        """
        ratio = self.get_current_ratio()
        cfg = self.config

        if ratio > cfg.target_ratio * 1.5:
            return "over-excited"
        elif ratio < cfg.target_ratio * 0.5:
            return "over-inhibited"
        else:
            return "balanced"

    @auto_diagnostics(
        scalars=["_exc_avg", "_inh_avg", "_inh_scale"],
    )
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information.

        Note: Auto-collects _exc_avg, _inh_avg, _inh_scale. Custom metrics added manually.

        Returns:
            Dictionary with E/I metrics
        """
        return {
            "current_ratio": self.get_current_ratio(),
            "target_ratio": self.config.target_ratio,
            "ratio_error": self.get_ratio_error(),
            "status": self.get_health_status(),
            "history_length": len(self._ratio_history),
        }

    def forward(
        self,
        exc_spikes: torch.Tensor,
        inh_spikes: torch.Tensor,
        inh_current: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """Forward pass: update tracking and optionally scale inhibitory current.

        Args:
            exc_spikes: Excitatory spike tensor
            inh_spikes: Inhibitory spike tensor
            inh_current: Optional inhibitory current to scale

        Returns:
            Tuple of (inh_scale, scaled_inh_current or None)
        """
        scale = self.update(exc_spikes, inh_spikes)

        if inh_current is not None:
            return scale, inh_current * scale
        else:
            return scale, None


class LayerEIBalance(nn.Module):
    """E/I balance tracking for a single layer with separate E/I populations.

    Many cortical layers have distinct excitatory (pyramidal) and inhibitory
    (interneuron) populations. This class tracks E/I balance for such layers.
    """

    def __init__(
        self,
        n_exc: int,
        n_inh: int,
        config: Optional[EIBalanceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.config = config or EIBalanceConfig()
        self._device = device

        # Core regulator
        self.regulator = EIBalanceRegulator(self.config, device)

        # Per-neuron excitability modulation (optional refinement)
        # Excitatory neurons that fire too much become less excitable
        self.register_buffer(
            "exc_modulation",
            torch.ones(n_exc, device=device),
        )
        self.register_buffer(
            "inh_modulation",
            torch.ones(n_inh, device=device),
        )

    def update(
        self,
        exc_spikes: torch.Tensor,
        inh_spikes: torch.Tensor,
    ) -> float:
        """Update E/I balance tracking.

        Args:
            exc_spikes: Spikes from excitatory population [batch, n_exc]
            inh_spikes: Spikes from inhibitory population [batch, n_inh]

        Returns:
            Inhibitory scaling factor
        """
        return self.regulator.update(exc_spikes, inh_spikes)

    def scale_inhibition(
        self,
        inh_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Scale inhibitory weights to maintain E/I balance.

        Args:
            inh_weights: Inhibitory weight matrix

        Returns:
            Scaled weight matrix
        """
        return inh_weights * self.regulator.get_inh_scaling()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        return dict(self.regulator.get_diagnostics())
