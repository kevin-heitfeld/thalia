"""
Mental Simulation Coordinator for Model-Based Planning.

Orchestrates existing brain regions (PFC, hippocampus, striatum, cortex)
to perform mental simulation of action sequences.

NO BACKPROPAGATION. All learning remains local to each region.

Biological Inspiration:
    - Vicarious Trial and Error (Tolman, 1948)
    - Hippocampal theta sequences (Johnson & Redish, 2007)
    - PFC working memory (Miller & Cohen, 2001)
    - Model-based planning (Daw et al., 2005)

Author: Thalia Project
Date: December 10, 2025
Phase: 2 - Model-Based Planning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SimulationConfig:
    """Configuration for mental simulation."""

    # Search parameters
    depth: int = 3  # How many steps to look ahead
    branching_factor: int = 3  # How many actions to consider per state

    # Hippocampal retrieval
    n_similar_experiences: int = 5  # Top-K similar episodes
    similarity_threshold: float = 0.5  # Minimum similarity to use

    # PFC simulation
    simulation_noise: float = 0.1  # Add noise to simulations

    # Value evaluation
    discount_gamma: float = 0.95  # Discount future rewards

    # Computational budget
    max_simulations: int = 100  # Limit planning time


@dataclass
class Rollout:
    """Result of a simulated action sequence."""

    states: List[torch.Tensor]  # Sequence of simulated states
    actions: List[int]  # Sequence of actions
    rewards: List[float]  # Sequence of predicted rewards
    cumulative_value: float  # Total discounted return
    uncertainty: float  # Confidence in simulation


class MentalSimulationCoordinator:
    """
    Coordinates existing brain regions to perform mental simulation.

    Uses:
        - PFC: Maintains simulated state in working memory
        - Hippocampus: Retrieves similar past experiences
        - PFC/Cortex: Predictive coding for next-state prediction
        - Striatum: Evaluates simulated states

    NO separate world model. NO backpropagation. Just coordination!
    """

    def __init__(
        self,
        pfc,  # Prefrontal instance
        hippocampus,  # Hippocampus instance
        striatum,  # Striatum instance
        cortex=None,  # Optional Cortex instance
        config: Optional[SimulationConfig] = None,
    ):
        self.pfc = pfc
        self.hippocampus = hippocampus
        self.striatum = striatum
        self.cortex = cortex
        self.config = config or SimulationConfig()

    def simulate_rollout(
        self,
        current_state: torch.Tensor,
        action_sequence: List[int],
        goal_context: Optional[torch.Tensor] = None,
    ) -> Rollout:
        """
        Simulate a specific action sequence.

        Args:
            current_state: Starting state (spike pattern)
            action_sequence: Sequence of actions to simulate
            goal_context: Optional goal for goal-conditioned values

        Returns:
            rollout: Simulated trajectory with predicted outcomes
        """
        states = [current_state.clone()]
        actions = []
        rewards = []
        cumulative_value = 0.0
        discount = 1.0

        # Load current state into PFC working memory
        simulated_state = current_state.clone()

        for _, action in enumerate(action_sequence):
            # 1. Retrieve similar past experiences from hippocampus
            similar_episodes = self.hippocampus.retrieve_similar(
                query_state=simulated_state,
                query_action=action,
                k=self.config.n_similar_experiences,
            )

            # 2. Predict next state using PFC prediction
            #    Informed by similar past experiences
            next_state_pred = self._predict_next_state(
                current=simulated_state, action=action, similar_experiences=similar_episodes
            )

            # 3. Predict reward from similar experiences
            reward_pred = self._predict_reward(similar_episodes)

            # 4. Evaluate predicted state using striatum
            if goal_context is not None:
                state_value = self.striatum.evaluate_state(
                    next_state_pred, pfc_goal_context=goal_context
                )
            else:
                state_value = self.striatum.evaluate_state(next_state_pred)

            # Store trajectory
            states.append(next_state_pred)
            actions.append(action)
            rewards.append(reward_pred)
            cumulative_value += discount * (reward_pred + self.config.discount_gamma * state_value)
            discount *= self.config.discount_gamma

            # Update simulated state
            simulated_state = next_state_pred

        # Estimate uncertainty from hippocampal retrieval
        uncertainty = self._estimate_uncertainty(similar_episodes)

        return Rollout(
            states=states,
            actions=actions,
            rewards=rewards,
            cumulative_value=cumulative_value,
            uncertainty=uncertainty,
        )

    def plan_best_action(
        self,
        current_state: torch.Tensor,
        available_actions: List[int],
        goal_context: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Rollout]:
        """
        Search for best action using tree search.

        Implements breadth-first search with striatum value guidance.

        Args:
            current_state: Starting state
            available_actions: List of possible actions
            goal_context: Optional goal context from PFC

        Returns:
            best_action: First action of best sequence
            best_rollout: Full best trajectory
        """
        best_action = None
        best_value = float("-inf")
        best_rollout = None

        # Limit actions to consider (branching factor)
        if len(available_actions) > self.config.branching_factor:
            # Use striatum to prioritize which actions to explore
            action_priorities = self._get_action_priorities(
                current_state, available_actions, goal_context
            )
            top_actions = sorted(
                available_actions, key=lambda a: action_priorities[a], reverse=True
            )[: self.config.branching_factor]
        else:
            top_actions = available_actions

        # Simulate each action
        for action in top_actions:
            # Simulate depth steps ahead
            action_sequence = [action] + self._generate_greedy_sequence(
                current_state, action, self.config.depth - 1, goal_context
            )

            rollout = self.simulate_rollout(current_state, action_sequence, goal_context)

            # Keep best
            if rollout.cumulative_value > best_value:
                best_value = rollout.cumulative_value
                best_action = action
                best_rollout = rollout

        return best_action, best_rollout

    def _predict_next_state(
        self, current: torch.Tensor, action: int, similar_experiences: List[Dict]
    ) -> torch.Tensor:
        """
        Predict next state using PFC prediction + hippocampal memory.

        Combines PFC predictive dynamics with episodic memory retrieval
        for more accurate predictions.
        """
        if len(similar_experiences) > 0:
            # Weight predictions by similarity
            weighted_prediction = torch.zeros_like(current)
            total_weight = 0.0

            for exp in similar_experiences:
                similarity = exp["similarity"]
                if similarity > self.config.similarity_threshold:
                    # Use actual outcome from memory
                    weighted_prediction += similarity * exp["next_state"]
                    total_weight += similarity

            if total_weight > 0:
                next_state = weighted_prediction / total_weight
            else:
                # No good matches - use PFC prediction alone
                next_state = self.pfc.predict_next_state(
                    current, action, n_actions=self.striatum.n_actions
                )
        else:
            # No similar experiences - use PFC prediction
            next_state = self.pfc.predict_next_state(
                current, action, n_actions=self.striatum.n_actions
            )

        # Add simulation noise
        noise = torch.randn_like(next_state) * self.config.simulation_noise
        next_state = next_state + noise

        return next_state

    def _predict_reward(self, similar_experiences: List[Dict]) -> float:
        """Predict reward from similar past experiences."""
        if len(similar_experiences) == 0:
            return 0.0

        # Weighted average of similar experiences
        weighted_reward = 0.0
        total_weight = 0.0

        for exp in similar_experiences:
            similarity = exp["similarity"]
            if similarity > self.config.similarity_threshold:
                weighted_reward += similarity * exp["reward"]
                total_weight += similarity

        if total_weight > 0:
            return weighted_reward / total_weight
        else:
            return 0.0

    def _estimate_uncertainty(self, similar_experiences: List[Dict]) -> float:
        """Estimate uncertainty based on hippocampal retrieval."""
        if len(similar_experiences) == 0:
            return 1.0  # Max uncertainty

        # Uncertainty = 1 - average similarity
        avg_similarity = sum(exp["similarity"] for exp in similar_experiences) / len(
            similar_experiences
        )
        return float(1.0 - avg_similarity)

    def _get_action_priorities(
        self, state: torch.Tensor, actions: List[int], goal_context: Optional[torch.Tensor]
    ) -> Dict[int, float]:
        """
        Get action priorities from striatum (for pruning search tree).

        Uses striatum's learned values to prioritize which actions to explore.
        """
        priorities = {}
        for action in actions:
            # Simple priority: use striatum's evaluation
            # In full implementation, would simulate one step for each action
            value = self.striatum.evaluate_state(state, goal_context)
            priorities[action] = value
        return priorities

    def _generate_greedy_sequence(
        self,
        start_state: torch.Tensor,
        first_action: int,
        remaining_depth: int,
        goal_context: Optional[torch.Tensor],
    ) -> List[int]:
        """
        Generate greedy action sequence for remainder of rollout.

        Uses one-step lookahead to greedily select remaining actions.
        """
        sequence = []
        state = start_state

        for _ in range(remaining_depth):
            # Predict next state from first action
            similar = self.hippocampus.retrieve_similar(
                query_state=state, k=self.config.n_similar_experiences
            )

            next_state = self._predict_next_state(state, first_action, similar)

            # Choose best action from this state
            best_action = 0
            best_value = float("-inf")

            for action in range(self.striatum.n_actions):
                value = self.striatum.evaluate_state(next_state, goal_context)
                if value > best_value:
                    best_value = value
                    best_action = action

            sequence.append(best_action)
            state = next_state

        return sequence
