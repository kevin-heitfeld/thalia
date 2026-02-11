"""Unified Homeostasis - Constraint-Based Stability for Spiking Neural Networks.

**Scope**: Synaptic weight normalization and scaling
**Focus**: Mathematical constraints that GUARANTEE stability (not corrections)

This module replaces multiple biological homeostatic mechanisms with a
simpler, mathematically-guaranteed approach using CONSTRAINTS instead
of CORRECTIONS.

Related Homeostasis Modules:
=============================
This module is ONE of several homeostatic mechanisms in Thalia:

- **This module** (synaptic_homeostasis.py): Synaptic weight constraints
- **intrinsic_plasticity.py**: Neuron threshold adaptation (firing rate homeostasis)
- **metabolic.py**: Energy-based constraints and sparsity pressure
- **neuromodulation/homeostasis.py**: Global neuromodulator baseline regulation
- **regions/*/homeostasis_component.py**: Region-specific integration

**When to Use This Module**:
- Need to prevent weight explosion/collapse
- Want competitive normalization (zero-sum learning)
- Require guaranteed stability (not heuristic corrections)

**Philosophy**:
===============
The brain has 10+ overlapping homeostatic mechanisms (BCM, synaptic scaling,
intrinsic plasticity, heterosynaptic competition, etc.) because biology is
messy and redundant - evolution stacked these over millions of years.

**But we're not constrained by evolution** - we can impose mathematical guarantees!

**Key Insight: CONSTRAINTS > CORRECTIONS**:
===========================================
- **Correction approach**: "If weights get too high, slow down learning"
  → Might not work, requires tuning, can still fail
- **Constraint approach**: "Weights MUST sum to X"
  → Mathematically guaranteed, no tuning needed

**Effects Captured**:
=====================
1. **STABILITY**: Weights and activity cannot explode or collapse
   → Hard bounds, weight normalization

2. **DIFFERENTIATION**: Neurons learn different features (no redundancy)
   → Competitive normalization (if one grows, others shrink)

3. **COMPETITION**: Learning resources are finite (zero-sum game)
   → Per-neuron or per-action budget constraints

4. **MEMORY**: Learning persists appropriately over time
   → Normalization preserves relative differences between weights

5. **ADAPTABILITY**: Can still learn new things
   → Constraints bound learning but don't prevent it

**Biological Mechanisms Replaced**:
===================================
- BCM sliding threshold (metaplasticity)
- Synaptic scaling (global weight adjustment)
- Intrinsic plasticity (threshold adaptation)
- Heterosynaptic LTD (competitive weakening)
- Various soft bounds and rate limiters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from thalia.utils import clamp_weights


@dataclass
class UnifiedHomeostasisConfig:
    """Configuration for unified homeostatic regulation."""

    device: str = "cpu"  # Device to run on: 'cpu', 'cuda', 'cuda:0', etc.
    seed: Optional[int] = None  # Random seed for reproducibility. None = no seeding.

    # Weight constraints
    weight_budget: float = 1.0  # Target sum of weights per row (neuron)
    w_min: float = 0.0  # Absolute minimum weight
    w_max: float = 1.0  # Absolute maximum weight

    # Activity constraints
    activity_target: float = 0.1  # Target fraction of neurons active

    # Normalization settings
    soft_normalization: bool = True  # Use soft (multiplicative) vs hard normalization
    normalization_rate: float = 0.1  # How fast to approach target (soft only)

    # Competition settings
    competition_strength: float = 0.1  # Strength of winner-take-all effect


class UnifiedHomeostasis(nn.Module):
    """Unified homeostatic regulation using constraints.

    Replaces BCM, synaptic scaling, intrinsic plasticity, etc. with
    simple normalization operations that mathematically guarantee
    stability and competition.

    Key operations:
    1. Weight normalization: Each neuron's total input is bounded
    2. Activity normalization: Population activity is regulated
    3. Competitive adjustment: Strong weights suppress weak ones

    All operations are differentiable and can be applied every timestep.
    """

    def __init__(self, config: Optional[UnifiedHomeostasisConfig] = None):
        super().__init__()
        self.config = config or UnifiedHomeostasisConfig()

    def normalize_weights(
        self,
        weights: torch.Tensor,
        dim: int = 1,
    ) -> torch.Tensor:
        """Normalize weights to enforce budget constraint.

        Each row (dim=1) or column (dim=0) is scaled so its sum equals
        the weight budget. This:
        - Prevents runaway potentiation (sum is bounded)
        - Enables competition (if one grows, others shrink)
        - Implicitly implements BCM (relative weights matter, not absolute)

        Args:
            weights: Weight matrix [n_output, n_input]
            dim: Dimension to normalize (1=rows/neurons, 0=columns/inputs)

        Returns:
            Normalized weights with same shape
        """
        cfg = self.config

        if cfg.soft_normalization:
            # Soft normalization: gradually move toward target
            current_sum = weights.sum(dim=dim, keepdim=True).clamp(min=1e-8)
            target_scale = cfg.weight_budget / current_sum

            # Blend toward target
            scale = 1.0 + cfg.normalization_rate * (target_scale - 1.0)
            weights = weights * scale
        else:
            # Hard normalization: exactly enforce constraint
            current_sum = weights.sum(dim=dim, keepdim=True).clamp(min=1e-8)
            weights = weights / current_sum * cfg.weight_budget

        # Always enforce hard bounds
        weights = clamp_weights(weights, cfg.w_min, cfg.w_max, inplace=False)

        return weights

    def normalize_weights_paired(
        self,
        weights_a: torch.Tensor,
        weights_b: torch.Tensor,
        budget_per_pair: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize paired weight matrices to share a budget.

        Perfect for D1/D2 opponent pathways: the total weight to each
        action is constrained, forcing D1 and D2 to compete.

        If D1 for action X grows, D2 for action X must shrink (and vice versa).
        This GUARANTEES that neither pathway can dominate completely.

        Args:
            weights_a: First weight matrix (e.g., D1) [n_output, n_input]
            weights_b: Second weight matrix (e.g., D2) [n_output, n_input]
            budget_per_pair: Total budget for A + B per row

        Returns:
            Tuple of normalized (weights_a, weights_b)
        """
        cfg = self.config

        # Sum across both pathways
        sum_a = weights_a.sum(dim=1, keepdim=True).clamp(min=1e-8)
        sum_b = weights_b.sum(dim=1, keepdim=True).clamp(min=1e-8)
        total = sum_a + sum_b

        if cfg.soft_normalization:
            # Soft: gradually move toward budget
            target_scale = budget_per_pair / total
            scale = 1.0 + cfg.normalization_rate * (target_scale - 1.0)

            weights_a = weights_a * scale
            weights_b = weights_b * scale
        else:
            # Hard: exactly enforce budget
            scale = budget_per_pair / total
            weights_a = weights_a * scale
            weights_b = weights_b * scale

        # Enforce bounds
        weights_a = clamp_weights(weights_a, cfg.w_min, cfg.w_max, inplace=False)
        weights_b = clamp_weights(weights_b, cfg.w_min, cfg.w_max, inplace=False)

        return weights_a, weights_b

    def normalize_activity(
        self,
        activity: torch.Tensor,
        target: Optional[float] = None,
    ) -> torch.Tensor:
        """Normalize population activity toward target.

        Implements the effect of intrinsic plasticity: if neurons are
        too active, scale down; if too quiet, scale up.

        Args:
            activity: Activity tensor (spikes or rates)
            target: Target mean activity (uses config default if None)

        Returns:
            Scaled activity tensor
        """
        cfg = self.config
        target = target or cfg.activity_target

        # Compute current mean activity
        mean_activity = activity.mean().clamp(min=1e-8)

        # Compute scaling factor
        scale = target / mean_activity

        # Bound the scaling to prevent extreme adjustments
        scale = scale.clamp(0.5, 2.0)

        # Apply soft scaling
        if cfg.soft_normalization:
            scale = 1.0 + cfg.normalization_rate * (scale - 1.0)

        return activity * scale

    def compute_excitability_modulation(
        self,
        activity_history: torch.Tensor,
        tau: float = 100.0,
    ) -> torch.Tensor:
        """Compute per-neuron excitability modulation.

        Neurons that fire too much become less excitable.
        Neurons that fire too little become more excitable.

        This replaces intrinsic plasticity with a simpler feedback loop.

        Args:
            activity_history: Running average of each neuron's activity
            tau: Time constant for modulation (higher = slower)

        Returns:
            Excitability modulation factor per neuron (multiply g_exc by this)
        """
        cfg = self.config

        # Error from target
        error = activity_history - cfg.activity_target

        # Modulation: high activity → lower excitability
        modulation = 1.0 - error / tau

        # Bound to reasonable range
        modulation = modulation.clamp(0.5, 2.0)

        return modulation

    def apply_competition(
        self,
        weights: torch.Tensor,
        winners: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply competitive weight adjustment.

        Winner-take-all effect: weights to "winning" neurons are boosted,
        weights to "losing" neurons are suppressed.

        This replaces heterosynaptic LTD and lateral inhibition effects.

        Args:
            weights: Weight matrix [n_output, n_input]
            winners: Binary mask of "winning" outputs (uses max if None)

        Returns:
            Competitively adjusted weights
        """
        cfg = self.config

        # Identify winners (neurons with highest total weight)
        if winners is None:
            weight_sums = weights.sum(dim=1)
            threshold = weight_sums.mean()
            winners = (weight_sums > threshold).float()

        # Boost winners, suppress losers
        # Winners: multiply by (1 + strength)
        # Losers: multiply by (1 - strength)
        losers = 1.0 - winners

        scale = winners.unsqueeze(1) * (1.0 + cfg.competition_strength) + losers.unsqueeze(1) * (
            1.0 - cfg.competition_strength
        )

        weights = weights * scale

        # Re-normalize to maintain budget
        return self.normalize_weights(weights)


class StriatumHomeostasis(UnifiedHomeostasis):
    """Specialized homeostasis for D1/D2 opponent pathways.

    Extends UnifiedHomeostasis with striatum-specific constraints:
    - D1 and D2 share a budget per action
    - Competition between GO and NOGO pathways
    - Dopamine-modulated normalization
    - Activity tracking and excitability modulation (replaces IntrinsicPlasticity)
    """

    def __init__(
        self,
        n_actions: int,
        neurons_per_action: int = 1,
        config: Optional[UnifiedHomeostasisConfig] = None,
        target_activity: float = 0.05,  # Target firing rate (fraction of timesteps)
        excitability_tau: float = 100.0,  # Time constant for excitability modulation
        d2_neurons_per_action: Optional[int] = None,  # If different from D1
    ):
        super().__init__(config)
        self.n_actions = n_actions
        self.neurons_per_action = neurons_per_action  # D1 pathway
        self.d1_neurons_per_action = neurons_per_action
        self.d2_neurons_per_action = (
            d2_neurons_per_action if d2_neurons_per_action is not None else neurons_per_action
        )

        self.d1_size = n_actions * self.d1_neurons_per_action
        self.d2_size = n_actions * self.d2_neurons_per_action
        self.target_activity = target_activity
        self.excitability_tau = excitability_tau

        # Get device from config
        device = (config or UnifiedHomeostasisConfig()).device

        # Per-action budgets (can vary if some actions should be favored)
        self.register_buffer(
            "action_budgets",
            torch.ones(n_actions, device=device)
            * (config or UnifiedHomeostasisConfig()).weight_budget,
        )

        # Activity tracking for excitability modulation
        # Running average of firing rate per neuron (D1 and D2 separately)
        self.register_buffer("d1_activity_avg", torch.zeros(self.d1_size, device=device))
        self.register_buffer("d2_activity_avg", torch.zeros(self.d2_size, device=device))

        # Excitability modulation factors (multiply g_E by this)
        # > 1.0 means more excitable, < 1.0 means less excitable
        self.register_buffer("d1_excitability", torch.ones(self.d1_size, device=device))
        self.register_buffer("d2_excitability", torch.ones(self.d2_size, device=device))

    def update_activity(
        self,
        d1_spikes: torch.Tensor,
        d2_spikes: torch.Tensor,
        decay: float = 0.99,
    ) -> None:
        """Update running average of D1/D2 activity.

        Called every timestep to track firing rates.

        Args:
            d1_spikes: D1 spike tensor [n_neurons] or [n_neurons]
            d2_spikes: D2 spike tensor [n_neurons] or [n_neurons]
            decay: Exponential decay for running average
        """
        # Squeeze to 1D if needed
        d1 = d1_spikes.squeeze().float()
        d2 = d2_spikes.squeeze().float()

        # Update running averages
        self.d1_activity_avg: torch.Tensor = decay * self.d1_activity_avg + (1 - decay) * d1
        self.d2_activity_avg: torch.Tensor = decay * self.d2_activity_avg + (1 - decay) * d2

    def compute_excitability(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute excitability modulation based on activity history.

        Neurons firing above target → lower excitability (harder to fire)
        Neurons firing below target → higher excitability (easier to fire)

        This replaces IntrinsicPlasticity with a constraint-based approach.

        Returns:
            Tuple of (d1_excitability, d2_excitability) modulation factors
        """
        # Error from target rate
        d1_error = self.d1_activity_avg - self.target_activity
        d2_error = self.d2_activity_avg - self.target_activity

        # Modulation: high activity → lower excitability
        # excitability = 1 - error/tau, clamped to [0.5, 2.0]
        self.d1_excitability = (1.0 - d1_error / self.excitability_tau).clamp(0.5, 2.0)
        self.d2_excitability = (1.0 - d2_error / self.excitability_tau).clamp(0.5, 2.0)

        return self.d1_excitability, self.d2_excitability

    def get_excitability(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current excitability modulation factors.

        Returns:
            Tuple of (d1_excitability, d2_excitability)
        """
        return self.d1_excitability.clone(), self.d2_excitability.clone()

    def reset_activity(self) -> None:
        """Reset activity tracking (e.g., at start of new episode)."""
        self.d1_activity_avg.zero_()
        self.d2_activity_avg.zero_()
        self.d1_excitability.fill_(1.0)
        self.d2_excitability.fill_(1.0)

    def normalize_d1_d2(
        self,
        d1_weights: torch.Tensor,
        d2_weights: torch.Tensor,
        per_action: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize D1 and D2 weights with shared budget.

        Key insight: D1 and D2 for each action should compete for
        a fixed total weight budget. This GUARANTEES:
        - Neither pathway can completely dominate
        - If D1 grows, D2 must shrink (and vice versa)
        - Stable equilibrium is enforced mathematically

        Args:
            d1_weights: D1 pathway weights [n_neurons, n_input]
            d2_weights: D2 pathway weights [n_neurons, n_input]
            per_action: If True, normalize per action; else per neuron

        Returns:
            Tuple of (normalized_d1, normalized_d2)
        """
        if per_action:
            # Normalize per action (groups of neurons)
            d1_out = d1_weights.clone()
            d2_out = d2_weights.clone()

            for action in range(self.n_actions):
                start = action * self.neurons_per_action
                end = start + self.neurons_per_action

                d1_action = d1_weights[start:end]
                d2_action = d2_weights[start:end]

                # Sum across all neurons for this action
                d1_sum = d1_action.sum()
                d2_sum = d2_action.sum()
                total = (d1_sum + d2_sum).clamp(min=1e-8)

                # Scale to budget
                budget = self.action_budgets[action].item()
                scale = budget / total

                d1_out[start:end] = d1_action * scale
                d2_out[start:end] = d2_action * scale

            # Enforce bounds
            d1_out = clamp_weights(d1_out, self.config.w_min, self.config.w_max, inplace=False)
            d2_out = clamp_weights(d2_out, self.config.w_min, self.config.w_max, inplace=False)

            return d1_out, d2_out
        else:
            # Simple paired normalization per row
            return self.normalize_weights_paired(d1_weights, d2_weights)

    def modulate_by_dopamine(
        self,
        weights: torch.Tensor,
        dopamine: float,
        is_d2: bool = False,
    ) -> torch.Tensor:
        """Modulate normalization strength by dopamine level.

        During high dopamine (reward), we might want to allow more
        deviation from the budget to enable faster learning.
        During low dopamine, enforce the budget more strictly.

        Args:
            weights: Weights to normalize
            dopamine: Current dopamine level (-1 to 1)
            is_d2: Whether this is the D2 (indirect) pathway

        Returns:
            Modulated weights
        """
        # High dopamine = looser constraints (more learning flexibility)
        # Low dopamine = tighter constraints (more stability)
        flexibility = 0.5 + 0.5 * abs(dopamine)  # 0.5 to 1.0

        # Adjust normalization rate
        original_rate = self.config.normalization_rate
        self.config.normalization_rate = original_rate * (2.0 - flexibility)

        # Apply normalization
        weights = self.normalize_weights(weights)

        # Restore rate
        self.config.normalization_rate = original_rate

        return weights

    def grow(self, n_new_d1: int, n_new_d2: int) -> None:
        """Grow the striatum homeostasis to support more D1 and D2 neurons.

        Args:
            n_new_d1: Number of new D1 neurons to add
            n_new_d2: Number of new D2 neurons to add
        """
        # Update sizes
        self.d1_size += n_new_d1
        self.d2_size += n_new_d2

        # Expand D1 activity tracking buffers
        new_d1_activity = torch.zeros(n_new_d1, device=self.d1_activity_avg.device)
        self.d1_activity_avg = torch.cat([self.d1_activity_avg, new_d1_activity])

        # Expand D1 excitability buffers - start at neutral (1.0)
        new_d1_excitability = torch.ones(n_new_d1, device=self.d1_excitability.device)
        self.d1_excitability = torch.cat([self.d1_excitability, new_d1_excitability])

        # Expand D2 activity tracking buffers
        new_d2_activity = torch.zeros(n_new_d2, device=self.d2_activity_avg.device)
        self.d2_activity_avg = torch.cat([self.d2_activity_avg, new_d2_activity])

        # Expand D2 excitability buffers - start at neutral (1.0)
        new_d2_excitability = torch.ones(n_new_d2, device=self.d2_excitability.device)
        self.d2_excitability = torch.cat([self.d2_excitability, new_d2_excitability])


# =========================================================================
# BASE CLASS GROW (for non-striatum regions)
# =========================================================================
class UnifiedHomeostasisGrowable:
    """Mixin to provide grow() for non-striatum UnifiedHomeostasis instances."""

    def grow(self, n_new_neurons: int) -> None:
        """Grow the homeostasis component to support more neurons.

        Expands activity tracking and excitability buffers for new neurons.

        Args:
            n_new_neurons: Number of new neurons to add
        """
        # Access instance attributes (d1_size/d2_size are set in __init__)
        # Grow both pathways equally
        if hasattr(self, "d1_size"):
            self.d1_size += n_new_neurons
        if hasattr(self, "d2_size"):
            self.d2_size += n_new_neurons

        # Expand activity tracking buffers [n_neurons]
        new_d1_activity = torch.zeros(n_new_neurons, device=self.d1_activity_avg.device)
        self.d1_activity_avg: torch.Tensor = torch.cat([self.d1_activity_avg, new_d1_activity])

        new_d2_activity = torch.zeros(n_new_neurons, device=self.d2_activity_avg.device)
        self.d2_activity_avg: torch.Tensor = torch.cat([self.d2_activity_avg, new_d2_activity])

        # Expand excitability buffers [n_neurons] - start at neutral (1.0)
        new_d1_excitability = torch.ones(n_new_neurons, device=self.d1_excitability.device)
        self.d1_excitability: torch.Tensor = torch.cat([self.d1_excitability, new_d1_excitability])

        new_d2_excitability = torch.ones(n_new_neurons, device=self.d2_excitability.device)
        self.d2_excitability: torch.Tensor = torch.cat([self.d2_excitability, new_d2_excitability])
