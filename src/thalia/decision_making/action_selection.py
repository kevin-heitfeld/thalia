"""
Decision Making - Action Selection Module.

Standalone action selection utilities that can be used by any brain region
(Striatum, PFC, Motor Cortex, etc.) for converting neural activity into
discrete action choices.

This module provides:
1. Population coding (multiple neurons per action)
2. UCB exploration bonuses
3. Softmax/greedy/epsilon-greedy selection
4. Vote accumulation across timesteps

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import torch


class SelectionMode(Enum):
    """Action selection strategies."""

    SOFTMAX = auto()  # Temperature-based probabilistic selection
    GREEDY = auto()  # Always choose highest-value action
    EPSILON_GREEDY = auto()  # ε chance of random, 1-ε chance of greedy
    UCB = auto()  # Upper Confidence Bound (pure exploration)


@dataclass
class ActionSelectionConfig:
    """Configuration for action selection."""

    # Selection strategy
    mode: SelectionMode = SelectionMode.SOFTMAX

    # Softmax temperature (higher = more exploration)
    temperature: float = 1.0

    # Epsilon-greedy epsilon
    epsilon: float = 0.1

    # UCB exploration constant
    ucb_c: float = 2.0

    # Population coding
    neurons_per_action: int = 1  # 1 = no population coding

    # Vote accumulation
    accumulate_votes: bool = False
    vote_decay: float = 0.0  # Decay factor for accumulated votes


class ActionSelector:
    """Standalone action selector for converting neural votes to actions.

    This class is region-agnostic and can be used by:
    - Striatum (D1/D2 opponent voting)
    - PFC (working memory-guided decisions)
    - Motor Cortex (movement selection)
    - Any region making discrete choices

    Key Features:
    =============
    - **Population Coding**: Multiple neurons vote per action for robustness
    - **UCB Exploration**: Information-seeking bonuses for under-explored actions
    - **Flexible Selection**: Softmax, greedy, epsilon-greedy, or pure UCB
    - **Vote Accumulation**: Integrate evidence across multiple timesteps

    Usage:
    ======

        selector = ActionSelector(
            n_actions=4,
            config=ActionSelectionConfig(
                mode=SelectionMode.SOFTMAX,
                temperature=0.5,
                neurons_per_action=10,
            )
        )

        # Get votes from neural population
        d1_votes = torch.tensor([10, 5, 2, 8])  # 4 actions
        d2_votes = torch.tensor([2, 8, 10, 3])

        # Select action
        action, info = selector.select_action(
            positive_votes=d1_votes,
            negative_votes=d2_votes,
        )

        print(f"Chose action {action}, net votes: {info['net_votes']}")
    """

    def __init__(
        self,
        n_actions: int,
        config: ActionSelectionConfig,
        device: str = "cpu",
    ):
        """Initialize action selector.

        Args:
            n_actions: Number of possible actions
            config: Selection configuration
            device: Torch device
        """
        self.n_actions = n_actions
        self.config = config
        self.device = device

        # Track action counts for UCB
        self.action_counts = torch.zeros(n_actions, device=device)
        self.total_selections = 0

        # Accumulated votes (if enabled)
        self.accumulated_votes = None
        if config.accumulate_votes:
            self.accumulated_votes = torch.zeros(n_actions, device=device)

    def select_action(
        self,
        positive_votes: torch.Tensor,
        negative_votes: Optional[torch.Tensor] = None,
        exploration_bonus: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select an action from votes.

        Args:
            positive_votes: Votes favoring each action [n_actions]
            negative_votes: Votes against each action [n_actions] (optional)
            exploration_bonus: Additional exploration bonus per action (optional)
            mask: Boolean mask for valid actions [n_actions] (True = valid)

        Returns:
            Tuple of (chosen_action, info_dict)

        Info dict contains:
            - net_votes: Final votes used for selection
            - probabilities: Action probabilities (if softmax)
            - ucb_bonus: UCB bonuses applied (if enabled)
            - is_exploring: Whether exploration was used
        """
        # Compute net votes
        net_votes = positive_votes.clone()
        if negative_votes is not None:
            net_votes = net_votes - negative_votes

        # Apply mask (set invalid actions to -inf)
        if mask is not None:
            net_votes = torch.where(mask, net_votes, torch.full_like(net_votes, float("-inf")))

        # Accumulate votes if enabled
        if self.config.accumulate_votes and self.accumulated_votes is not None:
            self.accumulated_votes = self.accumulated_votes * self.config.vote_decay
            self.accumulated_votes += net_votes
            net_votes = self.accumulated_votes.clone()

        # Add exploration bonus
        ucb_bonus = None
        if exploration_bonus is not None:
            net_votes = net_votes + exploration_bonus
            ucb_bonus = exploration_bonus
        elif self.config.ucb_c > 0 and self.total_selections > 0:
            # Compute UCB bonus
            ucb_bonus = self._compute_ucb_bonus()
            net_votes = net_votes + ucb_bonus

        # Select action based on mode
        if self.config.mode == SelectionMode.GREEDY:
            action = int(net_votes.argmax().item())
            probabilities = None
            is_exploring = False

        elif self.config.mode == SelectionMode.EPSILON_GREEDY:
            if torch.rand(1).item() < self.config.epsilon:
                # Random exploration
                valid_actions = torch.arange(self.n_actions, device=self.device)
                if mask is not None:
                    valid_actions = valid_actions[mask]
                action = int(valid_actions[torch.randint(len(valid_actions), (1,))].item())
                is_exploring = True
            else:
                # Greedy
                action = int(net_votes.argmax().item())
                is_exploring = False
            probabilities = None

        elif self.config.mode == SelectionMode.UCB:
            # Pure UCB (always explore least-tried actions)
            action = int(net_votes.argmax().item())
            probabilities = None
            is_exploring = True

        else:  # SOFTMAX
            # Softmax selection
            logits = net_votes / self.config.temperature
            probabilities = torch.softmax(logits, dim=0)
            action = int(torch.multinomial(probabilities, num_samples=1).item())
            is_exploring: bool = bool(
                (probabilities.max() < 0.99).item()
            )  # Consider high-confidence as exploitation

        # Update statistics
        self.action_counts[action] += 1
        self.total_selections += 1

        # Prepare info dict
        info = {
            "net_votes": net_votes.detach(),
            "probabilities": probabilities.detach() if probabilities is not None else None,
            "ucb_bonus": ucb_bonus.detach() if ucb_bonus is not None else None,
            "is_exploring": is_exploring,
            "action_counts": self.action_counts.clone(),
        }

        return action, info

    def _compute_ucb_bonus(self) -> torch.Tensor:
        """Compute Upper Confidence Bound exploration bonus.

        UCB bonus = c * sqrt(log(total_selections) / action_count)

        Returns:
            Exploration bonus [n_actions]
        """
        # Avoid division by zero
        safe_counts = torch.maximum(
            self.action_counts,
            torch.ones_like(self.action_counts),
        )

        bonus = self.config.ucb_c * torch.sqrt(math.log(self.total_selections + 1) / safe_counts)

        return bonus

    def decode_population_votes(
        self,
        spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Decode action votes from population-coded spikes.

        With population coding, multiple neurons vote for each action.
        This aggregates their votes.

        Args:
            spikes: Spike activity [n_neurons] where n_neurons = n_actions * neurons_per_action

        Returns:
            Votes per action [n_actions]
        """
        if self.config.neurons_per_action == 1:
            # No population coding
            return spikes

        # Reshape to [n_actions, neurons_per_action] and sum
        votes = spikes.view(self.n_actions, self.config.neurons_per_action).sum(dim=1)
        return votes

    def reset_statistics(self) -> None:
        """Reset action counts and vote accumulation."""
        self.action_counts.zero_()
        self.total_selections = 0
        if self.accumulated_votes is not None:
            self.accumulated_votes.zero_()

    def get_state(self) -> Dict[str, Any]:
        """Get selector state for checkpointing."""
        state = {
            "action_counts": self.action_counts.clone(),
            "total_selections": self.total_selections,
        }
        if self.accumulated_votes is not None:
            state["accumulated_votes"] = self.accumulated_votes.clone()
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load selector state from checkpoint."""
        self.action_counts = state["action_counts"].to(self.device)
        self.total_selections = state["total_selections"]
        if "accumulated_votes" in state and self.accumulated_votes is not None:
            self.accumulated_votes = state["accumulated_votes"].to(self.device)


__all__ = [
    "ActionSelector",
    "ActionSelectionConfig",
    "SelectionMode",
]
