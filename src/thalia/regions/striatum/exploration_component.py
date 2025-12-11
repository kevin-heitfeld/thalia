"""
Striatum Exploration Component

Manages exploration strategies for action selection including UCB and adaptive tonic dopamine.
Standardized component following the region_components pattern.
"""

from __future__ import annotations

from typing import Dict, Any, List, TYPE_CHECKING

import torch

from thalia.core.region_components import ExplorationComponent
from thalia.core.base_manager import ManagerContext

if TYPE_CHECKING:
    from thalia.regions.striatum.config import StriatumConfig


class StriatumExplorationComponent(ExplorationComponent):
    """Manages exploration strategies for striatal action selection.

    Handles:
    1. UCB (Upper Confidence Bound) tracking for information-seeking behavior
    2. Adaptive exploration via tonic dopamine adjustment
    3. Performance history tracking
    """

    def __init__(
        self,
        config: StriatumConfig,
        context: ManagerContext,
        initial_tonic_dopamine: float = 0.1,
    ):
        """Initialize striatum exploration component.

        Args:
            config: Striatum configuration
            context: Manager context (device, dimensions, etc.)
            initial_tonic_dopamine: Starting tonic DA level
        """
        super().__init__(config, context)

        # Extract n_actions from context
        self.n_actions = context.n_output if context.n_output else 1

        # UCB tracking
        self._action_counts = torch.zeros(self.n_actions, device=self.context.device)
        self._total_trials = 0

        # Adaptive exploration tracking
        self._recent_rewards: List[float] = []
        self._recent_accuracy = 0.0
        self.tonic_dopamine = initial_tonic_dopamine

    def compute_exploration_bonus(
        self,
        q_values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute UCB exploration bonus for actions.

        Args:
            q_values: Current Q-values for actions
            **kwargs: Additional parameters

        Returns:
            Exploration bonus per action
        """
        if not self.config.ucb_exploration:
            return torch.zeros_like(q_values)

        # UCB formula: sqrt(ln(total_trials) / action_count)
        if self._total_trials == 0:
            return torch.ones_like(q_values) * float('inf')  # Try all actions initially

        ucb_bonus = self.config.ucb_coefficient * torch.sqrt(
            torch.log(torch.tensor(self._total_trials, device=q_values.device)) /
            (self._action_counts + 1e-8)
        )

        return ucb_bonus

    def update_action_counts(self, action: int) -> None:
        """Update UCB action counts after a trial completes.

        Args:
            action: The action that was selected (0 to n_actions-1)
        """
        self._action_counts[action] += 1
        self._total_trials += 1

    def update_performance(self, reward: float, correct: bool) -> None:
        """Update performance history for adaptive exploration.

        Args:
            reward: Reward received
            correct: Whether action was correct
        """
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > self.config.performance_window:
            self._recent_rewards.pop(0)

        # Update running accuracy
        window = min(len(self._recent_rewards), self.config.performance_window)
        if window > 0:
            correct_count = sum(1 for r in self._recent_rewards[-window:] if r > 0)
            self._recent_accuracy = correct_count / window

        # Adjust tonic dopamine based on performance
        if self.config.adaptive_exploration:
            # Poor performance → higher tonic DA → more exploration
            # Good performance → lower tonic DA → more exploitation
            if self._recent_accuracy < 0.5:
                self.tonic_dopamine = self.config.max_tonic_dopamine
            elif self._recent_accuracy > 0.8:
                self.tonic_dopamine = self.config.min_tonic_dopamine
            else:
                # Linear interpolation
                self.tonic_dopamine = (
                    self.config.min_tonic_dopamine +
                    (self.config.max_tonic_dopamine - self.config.min_tonic_dopamine) *
                    (0.8 - self._recent_accuracy) / 0.3
                )

    def reset_state(self) -> None:
        """Reset exploration component state."""
        self._action_counts.zero_()
        self._total_trials = 0
        self._recent_rewards.clear()
        self._recent_accuracy = 0.0

    def get_exploration_diagnostics(self) -> Dict[str, Any]:
        """Get exploration-specific diagnostics."""
        diag = super().get_exploration_diagnostics()
        diag.update({
            "action_counts": self._action_counts.tolist(),
            "total_trials": self._total_trials,
            "recent_accuracy": self._recent_accuracy,
            "tonic_dopamine": self.tonic_dopamine,
            "ucb_enabled": self.config.ucb_exploration,
            "adaptive_enabled": self.config.adaptive_exploration,
        })
        return diag


# Backwards compatibility alias
ExplorationManager = StriatumExplorationComponent
