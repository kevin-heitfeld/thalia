"""
Metabolic Constraints: Energy-Based Regularization for Neural Networks.

**Scope**: Energy-based activity regulation (ATP costs, sparsity pressure)
**Focus**: Soft constraints that encourage efficient, sparse representations

Real neurons are metabolically expensive — each spike costs ATP. The brain
uses ~20% of the body's energy despite being ~2% of its mass. This creates
a natural pressure toward efficient, sparse representations.

Related Homeostasis Modules:
=============================
This module is ONE of several homeostatic mechanisms in Thalia:

- **This module** (metabolic.py): Energy budgets and metabolic costs
- **synaptic_homeostasis.py**: Synaptic weight normalization and scaling
- **intrinsic_plasticity.py**: Neuron threshold adaptation (firing rate homeostasis)
- **neuromodulation/homeostasis.py**: Global neuromodulator baseline regulation
- **regions/*/homeostasis_component.py**: Region-specific integration

**When to Use This Module**:
- Need to limit total activity/energy consumption
- Want sparsity pressure (fewer spikes = lower cost)
- Require global or per-region energy budgets
- Model fatigue or resource depletion effects

This module implements soft metabolic constraints that:
1. PENALIZE EXCESSIVE ACTIVITY: High spike rates incur energy costs
2. ENCOURAGE SPARSITY: Fewer spikes = lower metabolic cost
3. PROVIDE INTRINSIC REWARD SIGNAL: Efficiency is rewarding
4. ENABLE ENERGY BUDGETING: Global or per-region budgets

Biological Basis:
=================
- Each action potential costs ~10^8 ATP molecules
- Synaptic transmission adds further costs
- Inhibition is cheaper than excitation (GABA vs glutamate)
- Sleep may serve metabolic restoration functions

Mathematical Model:
==================
Energy cost per timestep:
    E(t) = Σ_i spike_i(t) * cost_per_spike + baseline_cost

Penalty for exceeding budget:
    penalty(t) = max(0, E(t) - budget) * penalty_scale

This penalty can be:
- Added to loss function (training)
- Subtracted from intrinsic reward (RL)
- Used to modulate global gain (homeostasis)

Usage:
======
    from thalia.learning.homeostasis.metabolic import MetabolicConstraint, MetabolicConfig

    metabolic = MetabolicConstraint(MetabolicConfig(energy_budget=1.0))

    # Each timestep:
    cost = metabolic.compute_cost(total_spikes)
    penalty = metabolic.compute_penalty(total_spikes)

    # Optionally modulate gain when over budget:
    gain = metabolic.get_gain_modulation(total_spikes)
    input_current = input_current * gain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class MetabolicConfig:
    """Configuration for metabolic constraints.

    Attributes:
        energy_per_spike: Energy cost per spike (arbitrary units)
            Higher values = stronger pressure for sparsity.

        baseline_cost: Fixed energy cost per timestep (maintenance)
            Represents the cost of just existing (ion pumps, etc).

        energy_budget: Energy budget per timestep
            Activities exceeding this incur a penalty.

        penalty_scale: How strongly to penalize excess energy use
            Higher = stronger penalty = sparser activity.

        gain_modulation_enabled: Whether to modulate gain based on energy
            If True, high energy use reduces gain (automatic sparsity).

        gain_modulation_rate: How quickly gain changes with energy excess
            Higher = faster adaptation.

        gain_min: Minimum allowed gain modulation
            Prevents complete shutdown.

        gain_max: Maximum allowed gain modulation
            Prevents runaway excitation.

        tau_energy: Time constant for energy averaging (ms)
            Smooths energy estimation over time.
    """

    energy_per_spike: float = 0.001
    baseline_cost: float = 0.0
    energy_budget: float = 1.0
    penalty_scale: float = 0.1
    gain_modulation_enabled: bool = True
    gain_modulation_rate: float = 0.01
    gain_min: float = 0.1
    gain_max: float = 2.0
    tau_energy: float = 100.0


class MetabolicConstraint(nn.Module):
    """Metabolic constraint module for energy-based regularization.

    Tracks energy expenditure and provides penalties/modulations
    to encourage efficient, sparse neural activity.
    """

    def __init__(
        self,
        config: Optional[MetabolicConfig] = None,
    ):
        super().__init__()
        self.config = config or MetabolicConfig()

        # Cached decay factor (updated via update_temporal_parameters)
        self._energy_decay: Optional[float] = None

        # Running average of energy expenditure
        self._energy_avg: float = 0.0

        # Current gain modulation
        self._gain: float = 1.0

        # Cumulative statistics
        self._total_energy: float = 0.0
        self._total_spikes: int = 0
        self._total_penalties: float = 0.0
        self._timesteps: int = 0

        # History for diagnostics
        self._energy_history: List[float] = []

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update cached decay factor when dt changes.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._energy_decay = float(torch.exp(torch.tensor(-dt_ms / self.config.tau_energy)).item())

    def compute_cost(
        self,
        spikes: torch.Tensor,
    ) -> float:
        """Compute energy cost for current spikes.

        Args:
            spikes: Spike tensor (any shape)

        Returns:
            Energy cost for this timestep
        """
        cfg = self.config

        n_spikes = spikes.float().sum().item()
        cost = n_spikes * cfg.energy_per_spike + cfg.baseline_cost

        return float(cost)

    def compute_penalty(
        self,
        spikes: torch.Tensor,
    ) -> float:
        """Compute penalty for exceeding energy budget.

        Args:
            spikes: Spike tensor

        Returns:
            Penalty value (0 if within budget)
        """
        cfg = self.config

        cost = self.compute_cost(spikes)
        excess = max(0, cost - cfg.energy_budget)
        penalty = excess * cfg.penalty_scale

        return penalty

    def update(
        self,
        spikes: torch.Tensor,
    ) -> float:
        """Update energy tracking and return current cost.

        Args:
            spikes: Spike tensor

        Returns:
            Current energy cost
        """
        cfg = self.config

        # Compute current cost
        cost = self.compute_cost(spikes)
        n_spikes = int(spikes.float().sum().item())

        # Ensure decay factor is initialized
        if self._energy_decay is None:
            self.update_temporal_parameters(dt_ms=1.0)

        # Update running average
        assert self._energy_decay is not None
        self._energy_avg = self._energy_decay * self._energy_avg + (1 - self._energy_decay) * cost

        # Update cumulative stats
        self._total_energy += cost
        self._total_spikes += n_spikes
        self._timesteps += 1

        # Compute and track penalty
        excess = max(0, cost - cfg.energy_budget)
        penalty = excess * cfg.penalty_scale
        self._total_penalties += penalty

        # Update gain modulation if enabled
        if cfg.gain_modulation_enabled:
            self._update_gain(cost)

        # Store history
        self._energy_history.append(cost)
        if len(self._energy_history) > 10000:
            self._energy_history = self._energy_history[-5000:]

        return cost

    def _update_gain(self, current_cost: float):
        """Update gain modulation based on energy expenditure."""
        cfg = self.config

        # Error from budget (positive = over budget)
        error = self._energy_avg - cfg.energy_budget

        # Reduce gain if over budget, increase if under
        adjustment = -cfg.gain_modulation_rate * error
        self._gain = self._gain * (1 + adjustment)

        # Clamp to valid range
        self._gain = max(cfg.gain_min, min(cfg.gain_max, self._gain))

    def get_gain_modulation(self) -> float:
        """Get current gain modulation factor.

        Multiply input currents by this to reduce activity when over budget.

        Returns:
            Gain modulation factor
        """
        return self._gain

    def modulate_input(self, input_current: torch.Tensor) -> torch.Tensor:
        """Apply gain modulation to input current.

        Args:
            input_current: Input current tensor

        Returns:
            Modulated input current
        """
        return input_current * self._gain

    def get_intrinsic_reward(
        self,
        spikes: torch.Tensor,
        efficiency_bonus: float = 0.0,
    ) -> float:
        """Compute intrinsic reward based on metabolic efficiency.

        Negative reward for high energy use, bonus for efficiency.

        Args:
            spikes: Spike tensor
            efficiency_bonus: Bonus reward for being under budget

        Returns:
            Intrinsic reward (negative = metabolic cost)
        """
        cost = self.compute_cost(spikes)

        if cost <= self.config.energy_budget:
            # Under budget: small positive reward
            return efficiency_bonus
        else:
            # Over budget: negative reward proportional to excess
            return -self.compute_penalty(spikes)

    def is_over_budget(self) -> bool:
        """Check if currently exceeding energy budget."""
        return self._energy_avg > self.config.energy_budget

    def get_efficiency(self) -> float:
        """Get current metabolic efficiency (budget / actual).

        Returns:
            Efficiency ratio (>1 = under budget, <1 = over budget)
        """
        if self._energy_avg < 1e-8:
            return float("inf")
        return self.config.energy_budget / self._energy_avg

    def get_average_spikes_per_timestep(self) -> float:
        """Get average spikes per timestep."""
        if self._timesteps == 0:
            return 0.0
        return self._total_spikes / self._timesteps

    def forward(
        self,
        spikes: torch.Tensor,
    ) -> float:
        """Forward pass: update and return cost."""
        return self.update(spikes)


class RegionalMetabolicBudget:
    """Metabolic budget allocation across brain regions.

    Different brain regions may have different energy budgets based on
    their size, function, or importance. This class manages per-region
    budgets and tracks overall brain energy.
    """

    def __init__(
        self,
        region_budgets: Dict[str, float],
        global_budget: Optional[float] = None,
        config: Optional[MetabolicConfig] = None,
    ):
        """Initialize regional metabolic budget.

        Args:
            region_budgets: Dict mapping region name to energy budget
            global_budget: Optional global budget (overrides sum of regions)
            config: Base metabolic config
        """
        self.config = config or MetabolicConfig()
        self.region_budgets = region_budgets.copy()

        if global_budget is not None:
            self.global_budget = global_budget
        else:
            self.global_budget = sum(region_budgets.values())

        # Per-region trackers
        self.region_costs: Dict[str, float] = {r: 0.0 for r in region_budgets}
        self.region_penalties: Dict[str, float] = {r: 0.0 for r in region_budgets}

    def update_region(
        self,
        region: str,
        spikes: torch.Tensor,
    ) -> float:
        """Update energy tracking for a specific region.

        Args:
            region: Region name
            spikes: Spike tensor for this region

        Returns:
            Energy cost for this region
        """
        cfg = self.config

        n_spikes = spikes.float().sum().item()
        cost = n_spikes * cfg.energy_per_spike

        self.region_costs[region] = cost

        # Compute region-specific penalty
        budget = self.region_budgets.get(region, cfg.energy_budget)
        excess = max(0, cost - budget)
        self.region_penalties[region] = excess * cfg.penalty_scale

        return float(cost)

    def get_total_cost(self) -> float:
        """Get total energy cost across all regions."""
        return sum(self.region_costs.values())

    def get_total_penalty(self) -> float:
        """Get total penalty across all regions."""
        return sum(self.region_penalties.values())

    def is_globally_over_budget(self) -> bool:
        """Check if total cost exceeds global budget."""
        return self.get_total_cost() > self.global_budget

    def get_region_efficiency(self, region: str) -> float:
        """Get efficiency for a specific region."""
        cost = self.region_costs.get(region, 0)
        budget = self.region_budgets.get(region, 1)
        if cost < 1e-8:
            return float("inf")
        return budget / cost
