"""
Striatum Exploration Component

Manages exploration strategies for action selection including UCB and adaptive tonic dopamine.
Standardized component following the region_components pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from thalia.core.region_components import ExplorationComponent
from thalia.managers.base_manager import ManagerContext
from thalia.regions.striatum.exploration import ExplorationConfig


@dataclass
class ExplorationState:
    """State for exploration component.

    Attributes:
        action_counts: Number of times each action has been selected [n_actions]
        total_trials: Total number of trials (for UCB computation)
        recent_rewards: Sliding window of recent rewards for adaptive exploration
        recent_accuracy: Recent accuracy metric
        tonic_dopamine: Current tonic dopamine level (motivation/exploration)
    """

    action_counts: torch.Tensor  # [n_actions] int tensor
    total_trials: int
    recent_rewards: List[float]
    recent_accuracy: float
    tonic_dopamine: float


class StriatumExplorationComponent(ExplorationComponent):
    """Manages exploration strategies for striatal action selection.

    This component implements exploration mechanisms that balance exploitation
    of known good actions with information-seeking about uncertain actions.

    Responsibilities:
    =================
    1. **UCB Tracking**: Upper Confidence Bound for information seeking
    2. **Adaptive Exploration**: Adjusts exploration based on performance
    3. **Tonic Dopamine Management**: Motivational state modulation
    4. **Performance History**: Tracks recent rewards and accuracy

    Exploration Strategies:
    =======================
    - **UCB (Upper Confidence Bound)**:
      bonus = coefficient * sqrt(log(total_trials) / action_count)
      Selects uncertain actions more often (information value)

    - **Adaptive Tonic Dopamine**:
      High performance → reduce exploration (exploit)
      Low performance → increase exploration (explore alternatives)
      Biologically plausible: sustained DA levels modulate motivation

    - **Tonic Modulation of Phasic**:
      Phasic DA (reward prediction error) scaled by tonic level
      High tonic → stronger learning from rewards
      Low tonic → reduced impact of individual rewards

    Biological Motivation:
    =====================
    In biological systems, exploration is driven by:
    - **Tonic dopamine**: Sustained baseline DA from VTA/SNc
    - **Norepinephrine**: Uncertainty-driven arousal (LC)
    - **Prefrontal control**: Goal-directed exploration policies

    This component models tonic DA and implicit exploration bonuses.

    Usage:
    ======
        exploration = StriatumExplorationComponent(config, context)

        # Add exploration bonus to Q-values
        exploration_bonus = exploration.compute_exploration_bonus(q_values)
        action_values = q_values + exploration_bonus

        # Update after trial
        exploration.update_exploration(action, reward, accuracy)

        # Get current tonic dopamine level
        tonic_da = exploration.get_tonic_dopamine()

    See Also:
    =========
    - `thalia.regions.striatum.action_selection` for selection methods
    - `docs/design/exploration_strategies.md` (if exists)
    """

    def __init__(
        self,
        config: ExplorationConfig,
        context: ManagerContext,
        initial_tonic_dopamine: float = 0.1,
    ):
        """Initialize striatum exploration component.

        Args:
            config: Exploration configuration
            context: Manager context (device, dimensions, etc.)
            initial_tonic_dopamine: Starting tonic DA level
        """
        super().__init__(config, context)  # type: ignore[arg-type]

        # Type narrowing for mypy: config is ExplorationConfig
        self.config: ExplorationConfig = config  # type: ignore[assignment]

        # Extract n_actions from context
        self.n_actions = context.n_output if context.n_output else 1

        # UCB tracking
        self._action_counts = torch.zeros(self.n_actions, device=self.context.device)
        self._total_trials = 0

        # Adaptive exploration tracking
        self._recent_rewards: List[float] = []
        self._recent_accuracy = 0.0
        self.tonic_dopamine = initial_tonic_dopamine

    def compute_exploration_bonus(self, q_values: torch.Tensor, **kwargs) -> torch.Tensor:
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
            return torch.ones_like(q_values) * float("inf")  # Try all actions initially

        ucb_bonus = self.config.ucb_coefficient * torch.sqrt(
            torch.log(torch.tensor(self._total_trials, device=q_values.device))
            / (self._action_counts + 1e-8)
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
                    self.config.min_tonic_dopamine
                    + (self.config.max_tonic_dopamine - self.config.min_tonic_dopamine)
                    * (0.8 - self._recent_accuracy)
                    / 0.3
                )

    def reset_state(self) -> None:
        """Reset exploration component state."""
        self._action_counts.zero_()
        self._total_trials = 0
        self._recent_rewards.clear()
        self._recent_accuracy = 0.0

    def grow(self, n_actions: int) -> None:
        """Grow the exploration component to support more actions.

        Args:
            n_actions: New total number of actions
        """
        if n_actions <= self.n_actions:
            return  # No growth needed

        # Expand action counts with zeros for new actions
        new_counts = torch.zeros(n_actions, device=self.context.device)
        new_counts[: self.n_actions] = self._action_counts
        self._action_counts = new_counts

        # Update n_actions
        self.n_actions = n_actions

    def compute_ucb_bonus(self) -> torch.Tensor:
        """Compute UCB (Upper Confidence Bound) exploration bonus.

        UCB formula: c * sqrt(ln(total_trials) / action_count)

        Returns:
            UCB bonus per action [n_actions]
        """
        if not self.config.ucb_exploration or self._total_trials == 0:
            return torch.zeros(self.n_actions, device=self.context.device)

        log_t = torch.log(
            torch.tensor(self._total_trials + 1, dtype=torch.float32, device=self.context.device)
        )
        ucb_bonus = self.config.ucb_coefficient * torch.sqrt(log_t / (self._action_counts + 1.0))
        return ucb_bonus

    def get_state(self) -> ExplorationState:
        """Get exploration component state for checkpointing.

        Returns:
            ExplorationState containing all state needed to restore exploration
        """
        return ExplorationState(
            action_counts=self._action_counts.detach().clone(),
            total_trials=self._total_trials,
            recent_rewards=self._recent_rewards.copy(),
            recent_accuracy=self._recent_accuracy,
            tonic_dopamine=self.tonic_dopamine,
        )

    def load_state(self, state: ExplorationState) -> None:
        """Restore exploration component state from checkpoint.

        Args:
            state: ExplorationState from get_state()
        """
        self._action_counts = state.action_counts.to(self.context.device)
        self._total_trials = state.total_trials
        self._recent_rewards = state.recent_rewards.copy()
        self._recent_accuracy = state.recent_accuracy
        self.tonic_dopamine = state.tonic_dopamine

    def get_exploration_diagnostics(self) -> Dict[str, Any]:
        """Get exploration-specific diagnostics."""
        diag = super().get_exploration_diagnostics()
        diag.update(
            {
                "action_counts": self._action_counts.tolist(),
                "total_trials": self._total_trials,
                "recent_accuracy": self._recent_accuracy,
                "tonic_dopamine": self.tonic_dopamine,
                "ucb_enabled": self.config.ucb_exploration,
                "adaptive_enabled": self.config.adaptive_exploration,
            }
        )
        return diag
