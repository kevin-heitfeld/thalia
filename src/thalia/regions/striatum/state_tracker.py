"""
Striatum State Tracker - Temporal State Management

Manages all temporal state variables for the striatum:
- Membrane potentials (via D1/D2 neurons)
- Spike accumulation and recent spike tracking  
- Vote accumulation for trial-level decisions
- Trial statistics (spike counts, timesteps)
- Last action tracking for credit assignment

This consolidates state management that was previously scattered
throughout the main Striatum class.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

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
        self.n_output = n_output
        self.device = device
        
        # Vote accumulation for trial-level decision
        self._d1_votes_accumulated = torch.zeros(n_actions, device=device)
        self._d2_votes_accumulated = torch.zeros(n_actions, device=device)
        
        # Recent spikes for lateral inhibition
        self.recent_spikes = torch.zeros(n_output, device=device)
        
        # Trial activity statistics
        self._trial_spike_count = 0.0
        self._trial_timesteps = 0
        
        # Last action for credit assignment
        self.last_action: Optional[int] = None
        
        # Last spikes for learning (stored by forward pass)
        self._last_d1_spikes: Optional[torch.Tensor] = None
        self._last_d2_spikes: Optional[torch.Tensor] = None
        
        # Last goal context for learning
        self._last_pfc_goal_context: Optional[torch.Tensor] = None
        
        # Exploration tracking
        self.exploring = False
        self._last_uncertainty = 0.0
        self._last_exploration_prob = 0.0
        
        # RPE tracking
        self._last_rpe = 0.0
        self._last_expected = 0.0
        
        # Homeostatic tracking
        self._activity_ema = 0.5  # Exponential moving average of activity
        self._homeostatic_scaling_applied = False
    
    def reset_trial_votes(self) -> None:
        """Reset vote accumulators at start of trial."""
        self._d1_votes_accumulated.zero_()
        self._d2_votes_accumulated.zero_()
    
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
    
    def get_accumulated_votes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get accumulated D1 and D2 votes.
        
        Returns:
            (d1_votes, d2_votes) each [n_actions]
        """
        return self._d1_votes_accumulated.clone(), self._d2_votes_accumulated.clone()
    
    def update_recent_spikes(self, d1_spikes: torch.Tensor, decay: float = 0.9) -> None:
        """Update recent spike history with decay.
        
        Args:
            d1_spikes: Current D1 spikes [n_output]
            decay: Decay factor for exponential averaging
        """
        self.recent_spikes = self.recent_spikes.float() * decay + d1_spikes.float()
    
    def update_trial_activity(self, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor) -> None:
        """Update trial activity statistics.
        
        Args:
            d1_spikes: D1 spikes this timestep
            d2_spikes: D2 spikes this timestep
        """
        self._trial_spike_count += d1_spikes.sum().item() + d2_spikes.sum().item()
        self._trial_timesteps += 1
    
    def get_trial_activity_rate(self) -> float:
        """Get average activity rate for current trial.
        
        Returns:
            Spikes per timestep per neuron
        """
        if self._trial_timesteps == 0:
            return 0.0
        return self._trial_spike_count / (self._trial_timesteps * self.n_output * 2)  # D1 + D2
    
    def store_spikes_for_learning(
        self,
        d1_spikes: torch.Tensor,
        d2_spikes: torch.Tensor,
        pfc_goal_context: Optional[torch.Tensor] = None,
    ) -> None:
        """Store spikes and goal context for learning.
        
        Args:
            d1_spikes: D1 spikes to store
            d2_spikes: D2 spikes to store
            pfc_goal_context: Optional PFC goal context
        """
        self._last_d1_spikes = d1_spikes.clone()
        self._last_d2_spikes = d2_spikes.clone()
        if pfc_goal_context is not None:
            self._last_pfc_goal_context = pfc_goal_context.clone()
        else:
            self._last_pfc_goal_context = None
    
    def get_last_spikes(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get last stored D1/D2 spikes.
        
        Returns:
            (last_d1_spikes, last_d2_spikes)
        """
        return self._last_d1_spikes, self._last_d2_spikes
    
    def get_last_goal_context(self) -> Optional[torch.Tensor]:
        """Get last stored PFC goal context.
        
        Returns:
            Last PFC goal context or None
        """
        return self._last_pfc_goal_context
    
    def set_last_action(self, action: int, exploring: bool = False) -> None:
        """Set last selected action.
        
        Args:
            action: Selected action index
            exploring: Whether this was an exploratory action
        """
        self.last_action = action
        self.exploring = exploring
    
    def update_exploration_stats(self, uncertainty: float, exploration_prob: float) -> None:
        """Update exploration statistics.
        
        Args:
            uncertainty: Current uncertainty value
            exploration_prob: Probability of exploration
        """
        self._last_uncertainty = uncertainty
        self._last_exploration_prob = exploration_prob
    
    def update_rpe_stats(self, rpe: float, expected_value: float) -> None:
        """Update RPE tracking statistics.
        
        Args:
            rpe: Reward prediction error
            expected_value: Expected value for action
        """
        self._last_rpe = rpe
        self._last_expected = expected_value
    
    def reset_state(self) -> None:
        """Reset all temporal state (for new episode)."""
        # Reset vote accumulators
        self._d1_votes_accumulated.zero_()
        self._d2_votes_accumulated.zero_()
        
        # Reset recent spikes
        self.recent_spikes.zero_()
        
        # Reset trial statistics
        self._trial_spike_count = 0.0
        self._trial_timesteps = 0
        
        # Reset action tracking
        self.last_action = None
        self.exploring = False
        
        # Reset learning storage
        self._last_d1_spikes = None
        self._last_d2_spikes = None
        self._last_pfc_goal_context = None
        
        # Reset statistics
        self._last_uncertainty = 0.0
        self._last_exploration_prob = 0.0
        self._last_rpe = 0.0
        self._last_expected = 0.0
        self._homeostatic_scaling_applied = False
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics for state tracker.
        
        Returns:
            Dictionary of diagnostic information
        """
        d1_votes, d2_votes = self.get_accumulated_votes()
        net_votes = self.get_net_votes()
        
        return {
            "last_action": self.last_action,
            "exploring": self.exploring,
            "d1_votes_mean": d1_votes.mean().item(),
            "d2_votes_mean": d2_votes.mean().item(),
            "net_votes_mean": net_votes.mean().item(),
            "net_votes_std": net_votes.std().item(),
            "recent_spikes_mean": self.recent_spikes.mean().item(),
            "trial_activity_rate": self.get_trial_activity_rate(),
            "trial_timesteps": self._trial_timesteps,
            "last_uncertainty": self._last_uncertainty,
            "last_exploration_prob": self._last_exploration_prob,
            "last_rpe": self._last_rpe,
            "last_expected": self._last_expected,
        }
