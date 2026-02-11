"""Intrinsic Plasticity - Activity-Dependent Threshold Adaptation for Homeostasis.

**Scope**: Neuron excitability adaptation (firing rate homeostasis)
**Focus**: Threshold modulation to maintain stable firing rates

Intrinsic plasticity adjusts neuron EXCITABILITY based on activity history.
Unlike synaptic plasticity (which modifies connections between neurons),
intrinsic plasticity modifies the neuron's INTERNAL PROPERTIES.

Related Homeostasis Modules:
=============================
This module is ONE of several homeostatic mechanisms in Thalia:

- **This module** (intrinsic_plasticity.py): Neuron threshold/excitability adaptation
- **synaptic_homeostasis.py**: Synaptic weight normalization and scaling
- **metabolic.py**: Energy-based constraints and sparsity pressure
- **neuromodulation/homeostasis.py**: Global neuromodulator baseline regulation
- **regions/*/homeostasis_component.py**: Region-specific integration

**When to Use This Module**:
- Need to maintain stable firing rates across neuron population
- Want activity-dependent threshold adaptation
- Require non-synaptic memory (persistent excitability changes)

**What It Does**:
=================
1. **THRESHOLD ADAPTATION**: Neurons that fire too much raise their threshold;
   neurons that fire too little lower their threshold.

2. **HOMEOSTATIC FUNCTION**: Maintains stable firing rates across the population,
   even as synaptic weights change dramatically through learning.

3. **MEMORY FUNCTION**: Persistent threshold changes encode long-term
   excitability preferences (a form of non-synaptic memory).

**Biological Basis**:
=====================
- **Ion channel remodeling**: Voltage-gated channel expression changes with activity
- **CREB-dependent gene expression**: Modifies intrinsic excitability over hours/days
- **After-hyperpolarization (AHP)**: Adapts to recent firing history
- **Timescale**: Minutes to hours (much SLOWER than synaptic plasticity)

**Mathematical Model**:
=======================
The threshold θ_i for neuron i adapts according to:

.. code-block:: none

    τ_θ * dθ_i/dt = target_rate - rate_i(t)

Discretized update rule:

.. code-block:: none

    θ_i(t+1) = θ_i(t) + η * (rate_i(t) - target_rate)

**Where**:
- τ_θ: Adaptation time constant (slow, ~10-100 seconds)
- target_rate: Desired firing rate (homeostatic setpoint)
- rate_i: Running average of neuron i's firing rate
- η: Learning rate for threshold adaptation

**Key References**:
===================
- Turrigiano (2011): "Too many cooks? Intrinsic and synaptic homeostatic mechanisms"
- Triesch (2005): "Synergies between intrinsic and synaptic plasticity mechanisms"
- Desai et al. (1999): "Plasticity in the intrinsic excitability of cortical neurons"

Usage:
======
    from thalia.learning.homeostasis.intrinsic_plasticity import (
        IntrinsicPlasticityConfig,
        IntrinsicPlasticity,
    )

    # Create intrinsic plasticity module
    ip = IntrinsicPlasticity(
        n_neurons=100,
        config=IntrinsicPlasticityConfig(target_rate=0.1),
    )

    # After each spike generation:
    spikes = lif(input_current)
    new_thresholds = ip.update(spikes, current_thresholds)

    # Or use wrapper for automatic adaptation:
    adaptive_lif = AdaptiveThresholdLIF(n_neurons=100, config=lif_config, ip_config=ip_config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from thalia.utils import compute_firing_rate


@dataclass
class IntrinsicPlasticityConfig:
    """Configuration for intrinsic plasticity.

    Attributes:
        target_rate: Target firing rate (fraction of timesteps with spike)
            Default 0.1 means neurons should fire ~10% of timesteps.
            This is quite active; cortical neurons often fire 0.01-0.05.

        tau_rate: Time constant for rate estimation (ms)
            How quickly the running average of firing rate updates.
            Smaller = faster tracking, more noise.

        tau_threshold: Time constant for threshold adaptation (ms)
            How quickly thresholds change. Should be SLOW (seconds to minutes).
            Faster than synaptic homeostasis, slower than learning.

        learning_rate: Rate at which threshold adapts per timestep
            Computed from tau_threshold if not specified.

        v_thresh_min: Minimum allowed threshold
            Prevents neurons from becoming too excitable.

        v_thresh_max: Maximum allowed threshold
            Prevents neurons from becoming completely silent.

        bidirectional: Whether to both increase AND decrease thresholds
            If False, only increases threshold (prevents runaway only).
    """

    target_rate: float = 0.1
    tau_rate: float = 1000.0  # 1 second for rate averaging
    tau_threshold: float = 10000.0  # 10 seconds for threshold adaptation
    learning_rate: Optional[float] = None  # Computed from tau if None
    v_thresh_min: float = 0.5
    v_thresh_max: float = 2.0
    bidirectional: bool = True


class IntrinsicPlasticity(nn.Module):
    """Intrinsic plasticity module for threshold adaptation.

    Tracks each neuron's firing rate and adjusts thresholds to maintain
    the target rate. Can be used standalone or integrated into LIF neurons.
    """

    def __init__(
        self,
        n_neurons: int,
        config: Optional[IntrinsicPlasticityConfig] = None,
        initial_threshold: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or IntrinsicPlasticityConfig()
        self._device = device or torch.device("cpu")

        # Cached temporal factors (updated via update_temporal_parameters)
        self._rate_decay: Optional[float] = None
        self._effective_lr: Optional[float] = None

        # Per-neuron running average of firing rate
        self.register_buffer(
            "rate_avg",
            torch.full((n_neurons,), self.config.target_rate, device=self._device),
        )
        self.rate_avg: torch.Tensor  # Type annotation for mypy

        # Per-neuron thresholds (can be modified by adaptation)
        self.register_buffer(
            "thresholds",
            torch.full((n_neurons,), initial_threshold, device=self._device),
        )
        self.thresholds: torch.Tensor  # Type annotation for mypy

        # Track total adaptation for diagnostics
        self._total_adaptation: float = 0.0
        self._update_count: int = 0

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self.rate_avg.device

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update cached decay and learning rate when dt changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._rate_decay = float(torch.exp(torch.tensor(-dt_ms / self.config.tau_rate)).item())

        if self.config.learning_rate is not None:
            self._effective_lr = self.config.learning_rate
        else:
            # Derive from tau: lr ≈ dt_ms / tau_threshold
            self._effective_lr = dt_ms / self.config.tau_threshold

    def update(
        self,
        spikes: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update rate tracking and adapt thresholds.

        Args:
            spikes: Spike tensor [batch, n_neurons] or [n_neurons]
            thresholds: Current thresholds to modify (uses internal if None)

        Returns:
            Updated thresholds [n_neurons]
        """
        cfg = self.config

        # Use internal thresholds if not provided
        if thresholds is None:
            thresholds = self.thresholds

        # Compute instantaneous rate (mean over batch if batched)
        if spikes.dim() > 1:
            instant_rate = spikes.float().mean(dim=0)
        else:
            instant_rate = spikes.float()

        # Ensure temporal factors are initialized
        if self._rate_decay is None or self._effective_lr is None:
            self.update_temporal_parameters(dt_ms=1.0)

        # Update running average of rate
        assert self._rate_decay is not None and self._effective_lr is not None
        self.rate_avg = self._rate_decay * self.rate_avg + (1 - self._rate_decay) * instant_rate

        # Compute error: positive if firing too much
        rate_error = self.rate_avg - cfg.target_rate

        # Compute threshold adjustment
        # High rate → increase threshold (less excitable)
        # Low rate → decrease threshold (more excitable, if bidirectional)
        if cfg.bidirectional:
            adjustment = self._effective_lr * rate_error
        else:
            # Only increase threshold (prevent runaway)
            adjustment = self._effective_lr * torch.relu(rate_error)

        # Update thresholds
        new_thresholds = thresholds + adjustment

        # Clamp to valid range
        new_thresholds = torch.clamp(
            new_thresholds,
            cfg.v_thresh_min,
            cfg.v_thresh_max,
        )

        # Store in internal buffer
        self.thresholds = new_thresholds

        # Track diagnostics
        self._total_adaptation += adjustment.abs().mean().item()
        self._update_count += 1

        return new_thresholds

    def get_threshold_modulation(self) -> torch.Tensor:
        """Get current threshold modulation relative to initial.

        Returns:
            Modulation factor [n_neurons] (>1 = less excitable)
        """
        # Assume initial threshold was 1.0
        return self.thresholds

    def get_rate_errors(self) -> torch.Tensor:
        """Get current rate errors for each neuron.

        Returns:
            rate_avg - target_rate [n_neurons]
        """
        return self.rate_avg - self.config.target_rate

    def forward(
        self,
        spikes: torch.Tensor,
        thresholds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: same as update."""
        return self.update(spikes, thresholds)


class PopulationIntrinsicPlasticity(nn.Module):
    """Population-level intrinsic plasticity.

    Instead of per-neuron adaptation, this tracks population-level
    activity and provides a global excitability modulation.

    Useful when you want simpler dynamics or when individual neuron
    thresholds are not accessible.
    """

    def __init__(
        self,
        config: Optional[IntrinsicPlasticityConfig] = None,
    ):
        super().__init__()
        self.config = config or IntrinsicPlasticityConfig()

        # Cached temporal factors (updated via update_temporal_parameters)
        self._rate_decay: Optional[float] = None
        self._effective_lr: Optional[float] = None

        # Population-level rate average
        self._rate_avg: float = self.config.target_rate

        # Global excitability modulation (multiplied with input)
        self._excitability: float = 1.0

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update cached decay and learning rate when dt changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._rate_decay = float(torch.exp(torch.tensor(-dt_ms / self.config.tau_rate)).item())

        if self.config.learning_rate is not None:
            self._effective_lr = self.config.learning_rate
        else:
            # Derive from tau: lr ≈ dt_ms / tau_threshold
            self._effective_lr = dt_ms / self.config.tau_threshold

    def update(self, spikes: torch.Tensor) -> float:
        """Update with current spikes and return excitability modulation.

        Args:
            spikes: Spike tensor (any shape)

        Returns:
            Excitability modulation factor (multiply inputs by this)
        """
        cfg = self.config

        # Compute population rate
        pop_rate = compute_firing_rate(spikes)

        # Ensure temporal factors are initialized
        if self._rate_decay is None or self._effective_lr is None:
            self.update_temporal_parameters(dt_ms=1.0)

        # Update running average
        assert self._rate_decay is not None and self._effective_lr is not None
        self._rate_avg = self._rate_decay * self._rate_avg + (1 - self._rate_decay) * pop_rate

        # Compute error
        rate_error = self._rate_avg - cfg.target_rate

        # Adjust excitability
        # High rate → lower excitability
        # Low rate → higher excitability
        if cfg.bidirectional:
            adjustment = -self._effective_lr * rate_error
        else:
            adjustment = -self._effective_lr * max(0, rate_error)

        self._excitability = self._excitability * (1 + adjustment)

        # Clamp excitability to reasonable range
        self._excitability = max(0.1, min(10.0, self._excitability))

        return self._excitability

    def get_excitability(self) -> float:
        """Get current excitability modulation."""
        return self._excitability

    def get_rate_error(self) -> float:
        """Get current rate error."""
        return self._rate_avg - self.config.target_rate

    def modulate_input(self, input_current: torch.Tensor) -> torch.Tensor:
        """Apply excitability modulation to input current.

        Args:
            input_current: Input current tensor

        Returns:
            Modulated input current
        """
        return input_current * self._excitability
