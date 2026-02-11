"""
Striatum State Tracker - Temporal State Management

This component manages all temporal state variables for the striatum,
extracted from the main Striatum class to consolidate state management
and improve code organization.

**Responsibilities:**
- Track D1/D2 vote accumulation across timesteps within a trial
- Maintain recent spike history for lateral inhibition
- Record trial activity statistics (spike counts, timestep counts)
- Store last action for temporal credit assignment
- Cache last spikes and goal context for learning updates
- Track exploration state and uncertainty
- Monitor reward prediction errors (RPE)
- Maintain homeostatic activity tracking (EMA)
"""

from __future__ import annotations

from typing import Optional

import torch


class StriatumStateTracker:
    """Tracks temporal state variables for striatum.

    Consolidates:
    - Vote accumulation (D1/D2 across timesteps)
    - Recent spike tracking (for lateral inhibition)
    - Trial activity statistics
    - Last action for credit assignment
    - Last spikes for learning
    - Last goal context for learning
    """

    def __init__(
        self,
        n_actions: int,
        n_output: int,  # Total neurons (may be > n_actions with population coding)
        device: torch.device,
    ):
        """Initialize state tracker.

        Args:
            n_actions: Number of discrete actions
            n_output: Total number of output neurons
            device: Torch device for tensors
        """
        self.n_actions = n_actions
        self.device = device

        # Vote accumulation for trial-level decision
        self._d1_votes_accumulated = torch.zeros(n_actions, device=device)
        self._d2_votes_accumulated = torch.zeros(n_actions, device=device)

        # Recent spikes for lateral inhibition
        self.recent_spikes = torch.zeros(n_output, device=device)

        # Last action for credit assignment
        self.last_action: Optional[int] = None

        # Exploration tracking
        self.exploring = False
        self._last_exploration_prob = 0.0

    def accumulate_votes(self, d1_votes: torch.Tensor, d2_votes: torch.Tensor) -> None:
        """Accumulate D1/D2 votes for this timestep.

        Args:
            d1_votes: D1 votes per action [n_actions]
            d2_votes: D2 votes per action [n_actions]
        """
        self._d1_votes_accumulated += d1_votes
        self._d2_votes_accumulated += d2_votes

    def get_net_votes(self) -> torch.Tensor:
        """Get net votes (D1 - D2) for all actions.

        Returns:
            Net votes per action [n_actions]
        """
        return self._d1_votes_accumulated - self._d2_votes_accumulated

    def update_recent_spikes(
        self, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor, decay: float = 0.9
    ) -> None:
        """Update recent spike history with decay for both D1 and D2 pathways.

        Args:
            d1_spikes: Current D1 spikes [d1_size]
            d2_spikes: Current D2 spikes [d2_size]
            decay: Decay factor for exponential averaging
        """
        # Concatenate D1 and D2 spikes to form full MSN spike vector
        combined_spikes = torch.cat([d1_spikes, d2_spikes], dim=0)
        self.recent_spikes = self.recent_spikes.float() * decay + combined_spikes.float()

    def set_last_action(self, action: int, exploring: bool = False) -> None:
        """Set last selected action.

        Args:
            action: Selected action index
            exploring: Whether this was an exploratory action
        """
        self.last_action = action
        self.exploring = exploring

    def update_exploration_stats(self, exploration_prob: float) -> None:
        """Update exploration statistics.

        Args:
            exploration_prob: Probability of exploration
        """
        self._last_exploration_prob = exploration_prob
