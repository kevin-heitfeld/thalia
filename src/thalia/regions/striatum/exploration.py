"""
Exploration Management for Striatum

This module handles exploration strategies including UCB (Upper Confidence Bound)
tracking and adaptive exploration based on recent performance.

Biological basis:
- UCB exploration: Information-seeking behavior (try less-explored options)
- Adaptive exploration: Adjust tonic DA based on performance
  * Poor performance → higher tonic DA → more exploration (locus coeruleus stress response)
  * Good performance → lower tonic DA → more exploitation
- Tonic DA levels influence exploration probability and action selection

Extracted from Striatum god object as part of architecture refactoring (Tier 2.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

from thalia.core.base_manager import BaseManager, ManagerContext


@dataclass
class ExplorationConfig:
    """Configuration for exploration strategies.

    Attributes:
        ucb_exploration: Enable UCB (Upper Confidence Bound) exploration
        ucb_coefficient: UCB exploration coefficient (higher = more exploration)
        adaptive_exploration: Enable adaptive tonic DA adjustment based on performance
        performance_window: Number of trials to track for adaptive exploration
        min_tonic_dopamine: Minimum tonic DA level (high performance)
        max_tonic_dopamine: Maximum tonic DA level (poor performance)
        tonic_modulates_exploration: Whether tonic DA boosts exploration probability
        tonic_exploration_scale: How much tonic DA boosts exploration (multiplier)
    """
    ucb_exploration: bool = True
    ucb_coefficient: float = 1.0
    adaptive_exploration: bool = True
    performance_window: int = 20
    min_tonic_dopamine: float = 0.0
    max_tonic_dopamine: float = 0.3
    tonic_modulates_exploration: bool = True
    tonic_exploration_scale: float = 0.5


class ExplorationManager(BaseManager[ExplorationConfig]):
    """Manages exploration strategies for striatal action selection.

    Handles:
    1. UCB (Upper Confidence Bound) tracking for information-seeking behavior
    2. Adaptive exploration via tonic dopamine adjustment
    3. Performance history tracking

    Biological inspiration:
    - Locus coeruleus releases norepinephrine during uncertainty/stress
    - Tonic dopamine levels modulate exploration vs exploitation
    - UCB mirrors information-seeking behavior (try less-explored options)
    """

    def __init__(
        self,
        config: ExplorationConfig,
        context: ManagerContext,
        initial_tonic_dopamine: float = 0.1,
    ):
        """Initialize exploration manager.

        Args:
            config: Exploration configuration
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

    def update_action_counts(self, action: int) -> None:
        """Update UCB action counts after a trial completes.

        Should be called ONCE per trial after action selection is finalized.

        Args:
            action: The action that was selected (0 to n_actions-1)
        """
        self._action_counts[action] += 1
        self._total_trials += 1

    def compute_ucb_bonus(self) -> torch.Tensor:
        """Compute UCB (Upper Confidence Bound) exploration bonus for each action.

        The UCB bonus encourages trying less-explored actions. The formula is:

            bonus[a] = c * sqrt(log(t) / n_a)

        where:
        - c = exploration coefficient (config.ucb_coefficient)
        - t = total trials
        - n_a = number of times action a was taken

        Less-tried actions get higher bonus, encouraging information-seeking.

        Returns:
            Tensor of shape [n_actions] with UCB bonus per action.
            Returns zeros if UCB is disabled or total_trials == 0.
        """
        ucb_bonus = torch.zeros(self.n_actions, device=self.context.device)

        if not self.config.ucb_exploration or self._total_trials == 0:
            return ucb_bonus

        c = self.config.ucb_coefficient
        log_t = math.log(self._total_trials + 1)

        for a in range(self.n_actions):
            n_a = max(1, int(self._action_counts[a].item()))
            ucb_bonus[a] = c * math.sqrt(log_t / n_a)

        return ucb_bonus

    def adjust_tonic_dopamine(self, reward: float) -> dict:
        """Adjust tonic dopamine based on recent performance.

        Poor performance → increase tonic DA → more exploration
        Good performance → decrease tonic DA → more exploitation

        Biological basis:
        - Locus coeruleus releases norepinephrine during uncertainty/stress
        - Tonic DA levels influence exploration probability
        - Performance tracking uses exponential moving average for stability

        Args:
            reward: Raw reward signal from current trial (>0 = correct, <=0 = wrong)

        Returns:
            Dict with diagnostic information:
            - old_tonic: Previous tonic DA level
            - new_tonic: Updated tonic DA level
            - recent_accuracy: Current accuracy estimate
            - window_size: Number of trials in history
        """
        if not self.config.adaptive_exploration:
            return {
                "old_tonic": self.tonic_dopamine,
                "new_tonic": self.tonic_dopamine,
                "recent_accuracy": self._recent_accuracy,
                "window_size": 0,
                "adaptive_exploration_enabled": False,
            }

        # Track this trial's outcome (reward > 0 = correct)
        was_correct = 1.0 if reward > 0 else 0.0
        self._recent_rewards.append(was_correct)

        # Keep only the most recent trials
        window = self.config.performance_window
        if len(self._recent_rewards) > window:
            self._recent_rewards = self._recent_rewards[-window:]

        # Update running accuracy estimate
        if len(self._recent_rewards) > 0:
            self._recent_accuracy = sum(self._recent_rewards) / len(self._recent_rewards)

        # Adjust tonic DA: lower accuracy → higher tonic DA → more exploration
        # Linear interpolation between min and max tonic DA:
        # - At 0% accuracy: use max_tonic_dopamine
        # - At 100% accuracy: use min_tonic_dopamine
        old_tonic = self.tonic_dopamine
        accuracy = self._recent_accuracy
        min_tonic = self.config.min_tonic_dopamine
        max_tonic = self.config.max_tonic_dopamine

        # Smooth update: blend new estimate with old (momentum)
        target_tonic = max_tonic - accuracy * (max_tonic - min_tonic)
        momentum = 0.9  # Keep most of old value for stability
        self.tonic_dopamine = momentum * old_tonic + (1 - momentum) * target_tonic

        # Clamp to valid range
        self.tonic_dopamine = max(min_tonic, min(max_tonic, self.tonic_dopamine))

        return {
            "old_tonic": old_tonic,
            "new_tonic": self.tonic_dopamine,
            "recent_accuracy": self._recent_accuracy,
            "window_size": len(self._recent_rewards),
            "adaptive_exploration_enabled": True,
        }

    def get_state(self) -> dict:
        """Get exploration state for checkpointing.

        Returns:
            Dict with all stateful components:
            - action_counts: UCB action counts
            - total_trials: Total trial count
            - recent_rewards: Recent reward history
            - recent_accuracy: Current accuracy estimate
            - tonic_dopamine: Current tonic DA level
        """
        return {
            "action_counts": self._action_counts.detach().clone(),
            "total_trials": self._total_trials,
            "recent_rewards": list(self._recent_rewards),
            "recent_accuracy": self._recent_accuracy,
            "tonic_dopamine": self.tonic_dopamine,
        }

    def load_state(self, state: dict) -> None:
        """Restore exploration state from checkpoint.

        Args:
            state: Dict from get_state() with exploration state
        """
        self._action_counts = state["action_counts"].to(self.context.device)
        self._total_trials = state["total_trials"]
        self._recent_rewards = list(state["recent_rewards"])
        self._recent_accuracy = state["recent_accuracy"]
        self.tonic_dopamine = state["tonic_dopamine"]

    def reset_state(self) -> None:
        """Reset exploration state (trial boundaries)."""
        # Keep action counts and tonic DA across trials
        # Only clear transient state if needed
        pass

    def to(self, device: torch.device) -> "ExplorationManager":
        """Move all tensors to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.context.device = device
        self._action_counts = self._action_counts.to(device)
        return self

    def reset(self) -> None:
        """Reset exploration state to initial conditions.

        Clears UCB counts, performance history, resets tonic DA.
        Useful for starting new training sessions.
        """
        self._action_counts.zero_()
        self._total_trials = 0
        self._recent_rewards.clear()
        self._recent_accuracy = 0.0
        self.tonic_dopamine = 0.1  # Reset to default initial value

    def grow(self, new_n_actions: int) -> None:
        """Grow exploration state when actions are added to striatum.

        Extends action_counts with zeros for new actions.
        Preserves existing exploration state.

        Args:
            new_n_actions: New total number of actions (must be >= current)
        """
        if new_n_actions < self.n_actions:
            raise ValueError(
                f"Cannot shrink actions: new_n_actions={new_n_actions} < "
                f"current n_actions={self.n_actions}"
            )

        if new_n_actions == self.n_actions:
            return  # No change needed

        # Extend action counts with zeros for new actions
        new_counts = torch.zeros(new_n_actions, device=self.context.device)
        new_counts[:self.n_actions] = self._action_counts
        self._action_counts = new_counts

        self.n_actions = new_n_actions

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about exploration state.

        Returns:
            Dict with:
            - action_counts: List of counts per action
            - total_trials: Total trial count
            - recent_accuracy: Current accuracy estimate
            - tonic_dopamine: Current tonic DA level
            - least_tried_action: Index of least-tried action
            - most_tried_action: Index of most-tried action
        """
        counts = self._action_counts.tolist()
        least_tried = int(self._action_counts.argmin().item())
        most_tried = int(self._action_counts.argmax().item())

        return {
            "action_counts": counts,
            "total_trials": self._total_trials,
            "recent_accuracy": self._recent_accuracy,
            "tonic_dopamine": self.tonic_dopamine,
            "least_tried_action": least_tried,
            "most_tried_action": most_tried,
            "performance_window": len(self._recent_rewards),
        }
