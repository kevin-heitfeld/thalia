"""
Homeostasis Manager for Striatum

Encapsulates all homeostatic regulation for D1/D2 opponent pathways:
1. Unified Homeostasis - Budget-constrained weight normalization
2. Baseline Pressure - Active drift towards balanced D1/D2
3. Activity Tracking - Intrinsic excitability modulation

This module extracts homeostasis logic from the Striatum god object,
providing a clean interface for stability and balance maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from thalia.core.base_manager import BaseManager, ManagerContext
from thalia.learning.unified_homeostasis import (
    StriatumHomeostasis,
    UnifiedHomeostasisConfig,
)


@dataclass
class HomeostasisManagerConfig:
    """Configuration for striatal homeostasis management.

    Combines unified homeostasis and baseline pressure settings.
    """

    # Unified homeostasis settings
    weight_budget: float = 1.0
    target_firing_rate: float = 0.05
    excitability_tau: float = 100.0
    normalization_rate: float = 0.1

    # Baseline pressure settings
    baseline_pressure_enabled: bool = True
    baseline_pressure_rate: float = 0.01
    baseline_target_net: float = 0.0  # Target D1-D2 balance (0 = balanced)

    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0

    # Device
    device: torch.device = torch.device("cpu")


class HomeostasisManager(BaseManager[HomeostasisManagerConfig]):
    """Manages homeostatic regulation for striatal D1/D2 pathways.

    Provides two complementary mechanisms:

    1. Unified Homeostasis (from StriatumHomeostasis):
       - Budget-constrained weight normalization
       - D1+D2 per action must sum to fixed budget
       - Guarantees neither pathway can dominate
       - Activity-based excitability modulation

    2. Baseline Pressure:
       - Active drift towards balanced D1/D2 (NET = D1-D2 → target)
       - Prevents runaway biases where one action becomes dominant
       - Simulates synaptic scaling and homeostatic plasticity

    Together these ensure:
    - Stability: Weights and activity cannot explode/collapse
    - Balance: D1/D2 maintain healthy competition
    - Adaptability: Can still learn new patterns within constraints
    """

    def __init__(
        self,
        config: HomeostasisManagerConfig,
        context: ManagerContext,
    ):
        """Initialize homeostasis manager.

        Args:
            config: Homeostasis configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)

        # Extract dimensions from context
        self.n_actions = context.n_output if context.n_output else 1
        self.neurons_per_action = context.metadata.get("neurons_per_action", 1) if context.metadata else 1
        self.n_neurons = self.n_actions * self.neurons_per_action

        # Create unified homeostasis controller
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=self.config.weight_budget,
            w_min=self.config.w_min,
            w_max=self.config.w_max,
            normalization_rate=self.config.normalization_rate,
            device=self.context.device,
        )

        self.unified_homeostasis = StriatumHomeostasis(
            n_actions=self.n_actions,
            neurons_per_action=self.neurons_per_action,
            config=homeostasis_config,
            target_rate=self.config.target_firing_rate,
            excitability_tau=self.config.excitability_tau,
        )

    def normalize_weights(
        self,
        d1_weights: nn.Parameter,
        d2_weights: nn.Parameter,
        per_action: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply unified homeostasis normalization to D1/D2 weights.

        Ensures D1+D2 per action sums to budget, preventing runaway
        potentiation or depression.

        Args:
            d1_weights: D1 pathway weights [n_output, n_input]
            d2_weights: D2 pathway weights [n_output, n_input]
            per_action: If True, normalize per action; else per neuron

        Returns:
            Tuple of (normalized_d1, normalized_d2) tensors
        """
        return self.unified_homeostasis.normalize_d1_d2(
            d1_weights.data,
            d2_weights.data,
            per_action=per_action,
        )

    def apply_baseline_pressure(
        self,
        d1_weights: nn.Parameter,
        d2_weights: nn.Parameter,
    ) -> Dict[str, Any]:
        """Apply baseline pressure to drift D1/D2 towards target balance.

        Unlike budget normalization (which preserves D1:D2 ratios), baseline
        pressure actively drifts NET (D1-D2) towards a target value for each
        action. This prevents runaway biases where one action becomes dominant.

        Biological basis:
        - Synaptic scaling adjusts weights towards a setpoint
        - Homeostatic plasticity maintains balanced excitation/inhibition
        - Without active use, synapses drift towards baseline

        The mechanism:
        1. Calculate current NET (D1-D2) for each action
        2. Compute error from target (default: 0 = balanced)
        3. Adjust D1 down and D2 up (or vice versa) to reduce error

        Args:
            d1_weights: D1 pathway weights [n_output, n_input]
            d2_weights: D2 pathway weights [n_output, n_input]

        Returns:
            Dict with diagnostic information about the adjustment
        """
        if not self.config.baseline_pressure_enabled:
            return {"baseline_pressure_applied": False}

        rate = self.config.baseline_pressure_rate
        target = self.config.baseline_target_net

        # Track adjustments for diagnostics
        net_before = []
        net_after = []

        for action in range(self.n_actions):
            start = action * self.neurons_per_action
            end = start + self.neurons_per_action

            # Current mean weights for this action
            d1_action = d1_weights.data[start:end]
            d2_action = d2_weights.data[start:end]

            d1_mean = d1_action.mean()
            d2_mean = d2_action.mean()
            current_net = d1_mean - d2_mean
            net_before.append(current_net.item())

            # Error from target
            error = current_net - target

            # Adjustment: reduce D1 and increase D2 (or vice versa)
            # to move NET towards target
            # Split the correction between both pathways
            adjustment = rate * error * 0.5

            # Apply adjustments (proportionally to current values to avoid negatives)
            d1_weights.data[start:end] = (d1_action - adjustment).clamp(
                self.config.w_min, self.config.w_max
            )
            d2_weights.data[start:end] = (d2_action + adjustment).clamp(
                self.config.w_min, self.config.w_max
            )

            # Record new NET
            new_net = d1_weights.data[start:end].mean() - d2_weights.data[start:end].mean()
            net_after.append(new_net.item())

        return {
            "baseline_pressure_applied": True,
            "net_before": net_before,
            "net_after": net_after,
            "pressure_rate": rate,
            "target_net": target,
        }

    def update_activity(
        self,
        d1_spikes: torch.Tensor,
        d2_spikes: torch.Tensor,
        decay: float = 0.99,
    ) -> None:
        """Update running average of D1/D2 activity for excitability modulation.

        Should be called every timestep during forward pass.

        Args:
            d1_spikes: D1 spike tensor [n_neurons]
            d2_spikes: D2 spike tensor [n_neurons]
            decay: Exponential decay for running average
        """
        self.unified_homeostasis.update_activity(d1_spikes, d2_spikes, decay)

    def compute_excitability(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute excitability modulation based on activity history.

        Neurons firing above target → lower excitability (harder to fire)
        Neurons firing below target → higher excitability (easier to fire)

        This replaces IntrinsicPlasticity with a constraint-based approach.

        Returns:
            Tuple of (d1_excitability, d2_excitability) modulation factors
        """
        return self.unified_homeostasis.compute_excitability()

    def get_excitability(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current excitability modulation factors.

        Returns:
            Tuple of (d1_excitability, d2_excitability)
        """
        return self.unified_homeostasis.get_excitability()

    def reset_activity(self) -> None:
        """Reset activity tracking (e.g., at start of new episode)."""
        self.unified_homeostasis.reset_activity()

    def grow(self, n_new_neurons: int) -> None:
        """Grow homeostasis state for new neurons.

        Called when Striatum adds new neurons (e.g., new actions).

        Args:
            n_new_neurons: Number of new neurons being added
        """
        old_size = self.n_neurons
        new_size = old_size + n_new_neurons
        n_new_actions = n_new_neurons // self.neurons_per_action

        # Expand activity averages with zeros
        new_d1_avg = torch.zeros(n_new_neurons, device=self.context.device)
        new_d2_avg = torch.zeros(n_new_neurons, device=self.context.device)

        self.unified_homeostasis.d1_activity_avg = torch.cat([
            self.unified_homeostasis.d1_activity_avg,
            new_d1_avg,
        ])
        self.unified_homeostasis.d2_activity_avg = torch.cat([
            self.unified_homeostasis.d2_activity_avg,
            new_d2_avg,
        ])

        # Expand excitability with ones (neutral)
        new_d1_exc = torch.ones(n_new_neurons, device=self.context.device)
        new_d2_exc = torch.ones(n_new_neurons, device=self.context.device)

        self.unified_homeostasis.d1_excitability = torch.cat([
            self.unified_homeostasis.d1_excitability,
            new_d1_exc,
        ])
        self.unified_homeostasis.d2_excitability = torch.cat([
            self.unified_homeostasis.d2_excitability,
            new_d2_exc,
        ])

        # Update counts
        self.n_actions += n_new_actions
        self.n_neurons = new_size
        self.unified_homeostasis.n_actions = self.n_actions
        self.unified_homeostasis.n_neurons = new_size

        # Expand action budgets
        new_budgets = torch.ones(n_new_actions, device=self.context.device) * self.config.weight_budget
        self.unified_homeostasis.action_budgets = torch.cat([
            self.unified_homeostasis.action_budgets,
            new_budgets,
        ])

    def reset_state(self) -> None:
        """Reset homeostasis state (trial boundaries)."""
        self.reset_activity()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic metrics for monitoring.

        Returns:
            Dict with homeostasis-related metrics
        """
        d1_exc, d2_exc = self.get_excitability()

        return {
            "d1_excitability_mean": d1_exc.mean().item(),
            "d1_excitability_std": d1_exc.std().item(),
            "d2_excitability_mean": d2_exc.mean().item(),
            "d2_excitability_std": d2_exc.std().item(),
            "d1_activity_mean": self.unified_homeostasis.d1_activity_avg.mean().item(),
            "d2_activity_mean": self.unified_homeostasis.d2_activity_avg.mean().item(),
        }

    def to(self, device: torch.device) -> "HomeostasisManager":
        """Move all tensors to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.context.device = device
        # Unified homeostasis will handle its own tensors
        self.unified_homeostasis.to(device)
        return self

    def get_state(self) -> Dict[str, Any]:
        """Get homeostasis state for checkpointing.

        Returns:
            Dictionary with current state
        """
        return {
            "config": {
                "weight_budget": self.config.weight_budget,
                "target_firing_rate": self.config.target_firing_rate,
                "excitability_tau": self.config.excitability_tau,
                "baseline_pressure_enabled": self.config.baseline_pressure_enabled,
                "baseline_pressure_rate": self.config.baseline_pressure_rate,
                "baseline_target_net": self.config.baseline_target_net,
            },
            "unified_homeostasis": self.unified_homeostasis.get_state(),
            "n_actions": self.n_actions,
            "n_neurons": self.n_neurons,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore homeostasis state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        # Restore unified homeostasis
        self.unified_homeostasis.load_state(state["unified_homeostasis"])

        # Update counts
        self.n_actions = state["n_actions"]
        self.n_neurons = state["n_neurons"]
