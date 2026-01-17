"""Action Selection Mixin - Population Coding and UCB-Softmax Selection.

This module provides action selection functionality as a mixin class for the
striatum, handling the complex process of converting neural spikes into
discrete action choices.

**What This Provides**:
=======================
1. **Population coding**: Multiple neurons per action vote together
2. **UCB exploration**: Upper Confidence Bound action bonuses
3. **Softmax selection**: Temperature-based probabilistic choice
4. **Vote accumulation**: Integrate spikes across timesteps

**How Action Selection Works**:
================================

.. code-block:: none

    Striatal Spikes        Population Coding       Action Selection
    ───────────────        ─────────────────       ────────────────

    [0,1,0,1,1,0,...]  →  [votes_action0,     →   softmax()
                           votes_action1,      →   + UCB bonus
                           votes_action2]      →   = chosen action

**Population Coding**:
======================
Instead of 1 neuron = 1 action (brittle, noisy), we use:
- N neurons per action (e.g., 10 neurons represent "move left")
- Votes accumulated across neurons for each action
- More robust to noise, biological plausibility

**UCB Exploration**:
====================
Add bonus to under-explored actions:

.. code-block:: python

    bonus = c * sqrt(log(total_trials) / action_count)

Encourages trying less-explored actions (information-seeking).

**Context**:
============
Mixin design allows Striatum to stay focused on core functionality while
delegating action selection logic to this specialized module.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from thalia.constants.exploration import SOFTMAX_TEMPERATURE_DEFAULT

if TYPE_CHECKING:
    from thalia.regions.striatum.config import StriatumConfig


class ActionSelectionMixin:
    """Mixin providing action selection methods for Striatum.

    Handles population coding, UCB exploration, softmax selection,
    and vote accumulation across timesteps.

    Expects the following attributes on the mixed-in class:
    - config: StriatumConfig
    - n_actions: int
    - neurons_per_action: int
    - device: torch.device
    - d1_weights, d2_weights: nn.Parameter (properties delegating to pathways)
    - state: with .spikes attribute
    - state_tracker: StriatumStateTracker (provides votes, last_action, exploring)
    - exploration_manager: ExplorationManager
    """

    # Type hints for mixin - these are provided by Striatum
    config: "StriatumConfig"
    n_actions: int
    neurons_per_action: int
    # device: provided by LearnableComponent base class as @property
    state_tracker: Any  # StriatumStateTracker
    exploration_manager: Any  # ExplorationManager
    state: Any

    def _get_action_population_indices(self, action: int) -> slice:
        """Get the slice of neuron indices for a given action.

        With population coding enabled:
        - Action 0 → neurons 0 to neurons_per_action-1
        - Action 1 → neurons neurons_per_action to 2*neurons_per_action-1
        - etc.
        """
        start = action * self.neurons_per_action
        end = start + self.neurons_per_action
        return slice(start, end)

    def _decode_action_from_spikes(self, spikes: torch.Tensor) -> int:
        """Decode action from spike pattern using population voting.

        Args:
            spikes: Spike tensor [n_output] (1D)

        Returns the action whose population has the most spikes.
        """
        # Ensure 1D
        if spikes.dim() != 1:
            spikes = spikes.squeeze()

        if not self.config.population_coding:
            # Simple argmax for single-neuron coding
            if spikes.sum() == 0:
                return 0  # Default action
            return int(spikes.argmax().item())

        # Population coding: sum spikes per action population
        votes = torch.zeros(self.n_actions, device=self.device)
        for action in range(self.n_actions):
            pop_slice = self._get_action_population_indices(action)
            votes[action] = spikes[pop_slice].sum()

        return int(votes.argmax().item())

    def _count_population_votes(self, spikes: torch.Tensor) -> torch.Tensor:
        """Count spike votes for each action population.

        Used for D1-D2 subtraction to compute NET signal per action.

        Args:
            spikes: Spike tensor [n_output] (1D)

        Returns:
            Tensor of shape [n_actions] with vote counts per action.
        """
        # Ensure 1D
        if spikes.dim() != 1:
            spikes = spikes.squeeze()

        votes = torch.zeros(self.n_actions, device=self.device)

        for action in range(self.n_actions):
            pop_slice = self._get_action_population_indices(action)
            votes[action] = spikes[pop_slice].sum()

        return votes

    def _get_action_net_values(self) -> List[float]:
        """Get NET (D1-D2) value for each action.

        Used for uncertainty-driven exploration: when NET values are similar
        across actions, we're uncertain and should explore more.

        Returns:
            List of NET values, one per action.
        """
        nets = []
        for action in range(self.n_actions):
            start = action * self.neurons_per_action
            end = start + self.neurons_per_action

            d1_mean = self.d1_pathway.weights[start:end].mean().item()
            d2_mean = self.d2_pathway.weights[start:end].mean().item()
            nets.append(d1_mean - d2_mean)

        return nets

    def get_selected_action(self) -> Optional[int]:
        """Get the last selected action.

        Returns:
            Action index (0 to n_actions-1), or None if no action taken.
        """
        return self.state_tracker.last_action

    def get_accumulated_net_votes(self) -> torch.Tensor:
        """Get accumulated D1-D2 (NET) votes across all timesteps.

        This integrates sparse spiking evidence over the trial for robust
        action selection. Call this at the end of a trial, then use
        reset_accumulated_votes() before the next trial.

        Returns:
            Tensor of shape (n_actions,) with NET = D1_total - D2_total per action.
        """
        return self.state_tracker.get_net_votes()

    def reset_accumulated_votes(self) -> None:
        """Reset D1/D2 vote accumulators for a new trial."""
        self.state_tracker.reset_trial_votes()

    def update_action_counts(self, action: int) -> None:
        """Update UCB action counts after a trial completes.

        Delegates to ExplorationManager for centralized tracking.

        This should be called ONCE per trial by the brain_system after
        action selection is finalized. Not called inside forward() because
        forward() runs multiple times per timestep.

        Args:
            action: The action that was selected (0 to n_actions-1)
        """
        # Delegate to exploration component (always initialized in __init__)
        self.exploration.update_action_counts(action)

    def finalize_action(self, explore: bool = True) -> Dict[str, Any]:
        """Finalize action selection at the end of a trial.

        This consolidates the accumulated NET votes, applies UCB bonus,
        performs softmax (or argmax) selection, updates action counts ONCE,
        and returns diagnostics.

        Args:
            explore: Whether to allow exploration (bias-correcting + tonic DA)

        Returns:
            Dict with keys: selected_action, probs (if softmax), ucb_bonus,
            net_votes, exploring (bool), exploration_prob
        """
        net_votes = self.get_accumulated_net_votes()

        # UCB bonus - delegate to exploration component if available
        ucb_bonus = torch.zeros_like(net_votes)
        if hasattr(self, "exploration"):
            ucb_bonus = self.exploration.compute_ucb_bonus()
        elif self.config.ucb_exploration and self._total_trials > 0:
            # Fallback: compute UCB locally if exploration not available
            c = self.config.ucb_coefficient
            log_t = math.log(self._total_trials + 1)
            for a in range(self.n_actions):
                n_a = max(1, int(self._action_counts[a].item()))
                ucb_bonus[a] = c * math.sqrt(log_t / n_a)

        selection_values = net_votes + ucb_bonus

        # Compute bias-correcting exploration probability (same as forward)
        exploration_prob = 0.0
        if explore:
            action_nets = selection_values.tolist()
            if len(action_nets) > 1:
                net_range = max(action_nets) - min(action_nets)
                temperature = self.config.uncertainty_temperature
                bias_factor = (
                    net_range / (temperature + net_range) if (temperature + net_range) > 0 else 0.0
                )
                min_boost = self.config.min_exploration_boost
                max_boost = 0.5
                exploration_prob = min_boost + bias_factor * (max_boost - min_boost)

            # tonic modulation
            if self.config.tonic_modulates_exploration:
                tonic_boost = self.tonic_dopamine * self.config.tonic_exploration_scale
                exploration_prob = min(0.6, exploration_prob + tonic_boost)

        self.state_tracker.exploring = False
        probs = None
        selected_action = 0

        if explore and torch.rand(1).item() < exploration_prob:
            # Random exploration
            self.state_tracker.exploring = True
            selected_action = int(torch.randint(0, self.n_actions, (1,)).item())
        else:
            if self.config.softmax_action_selection:
                # Use configured temperature, or fall back to default constant
                temperature = getattr(
                    self.config, "softmax_temperature", SOFTMAX_TEMPERATURE_DEFAULT
                )
                selection_values_norm = selection_values - selection_values.max()
                probs = torch.softmax(selection_values_norm / temperature, dim=0)
                selected_action = int(torch.multinomial(probs, 1).item())
            else:
                max_val = selection_values.max().item()
                max_indices = (selection_values == max_val).nonzero(as_tuple=True)[0]
                if len(max_indices) > 1:
                    idx = int(torch.randint(len(max_indices), (1,)).item())
                    selected_action = int(max_indices[idx].item())
                else:
                    selected_action = int(max_indices[0].item())

        # Update bookkeeping ONCE per trial
        self.state_tracker.set_last_action(selected_action, self.state_tracker.exploring)
        self.state_tracker.update_exploration_stats(
            uncertainty=0.0, exploration_prob=exploration_prob
        )
        self.update_action_counts(selected_action)

        return {
            "selected_action": selected_action,
            "probs": probs,
            "ucb_bonus": ucb_bonus,
            "net_votes": net_votes,
            "exploring": self.state_tracker.exploring,
            "exploration_prob": exploration_prob,
        }

    def get_population_votes(self, spikes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the spike count for each action population.

        Useful for analyzing decision confidence with population coding.

        Args:
            spikes: Spike tensor to analyze. If None, uses last output.

        Returns:
            Tensor of shape (n_actions,) with vote counts per action.
        """
        if spikes is None:
            spikes = self.state.spikes
        if spikes is None:
            return torch.zeros(self.n_actions, device=self.device)

        spikes = spikes.squeeze()
        votes = torch.zeros(self.n_actions, device=self.device)

        for action in range(self.n_actions):
            pop_slice = self._get_action_population_indices(action)
            votes[action] = spikes[pop_slice].sum()

        return votes
