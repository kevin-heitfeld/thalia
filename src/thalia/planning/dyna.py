"""
Dyna-Style Background Planning.

Combines model-free learning (striatum) with model-based planning
(mental simulation coordinator).

NO separate world model. Uses existing region coordination.

Reference:
    Sutton (1990): Integrated architectures for learning, planning, and reacting

Author: Thalia Project
Date: December 10, 2025
Phase: 2 - Model-Based Planning
"""

from dataclasses import dataclass
from typing import Optional, Dict
import random
import torch


@dataclass
class DynaConfig:
    """Configuration for Dyna planning."""

    # Planning budget
    n_planning_steps: int = 5  # Simulations per real experience

    # Learning from simulations
    simulation_lr_scale: float = 0.5  # Discount simulated updates

    # Prioritization
    use_prioritized_sweeping: bool = True  # Focus on important states
    priority_threshold: float = 0.1  # Minimum priority


class DynaPlanner:
    """
    Dyna algorithm: Combine real experience with simulated planning.

    Process:
        1. Real experience → update striatum values
        2. Sample previous states from hippocampus
        3. Simulate outcomes using mental simulation coordinator
        4. Update striatum values from simulated experience

    NO separate world model - uses existing region coordination!
    """

    def __init__(
        self,
        coordinator,  # MentalSimulationCoordinator instance
        striatum,  # Striatum instance
        hippocampus,  # Hippocampus instance
        config: Optional[DynaConfig] = None
    ):
        self.coordinator = coordinator
        self.striatum = striatum
        self.hippocampus = hippocampus
        self.config = config or DynaConfig()

        # Priority queue for states to simulate (for prioritized sweeping)
        self.state_priorities: Dict[int, float] = {}

    def process_real_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        goal_context: Optional[torch.Tensor] = None
    ) -> None:
        """
        Process real experience and trigger background planning.

        Args:
            state, action, reward, next_state, done: Real transition
            goal_context: Optional goal context
        """
        # 1. Update striatum from real experience (model-free)
        # Note: Actual TD update would happen in Brain.deliver_reward()
        # Here we just trigger the planning

        # 2. Hippocampus stores experience (already happens automatically)

        # 3. Update priority for this state (for prioritized sweeping)
        if self.config.use_prioritized_sweeping:
            # Compute TD error as priority (approximation)
            if hasattr(self.striatum, 'value_estimates') and self.striatum.value_estimates is not None:
                current_value = self.striatum.evaluate_state(state, goal_context)
                next_value = self.striatum.evaluate_state(next_state, goal_context)
                td_error = abs(reward + 0.95 * next_value - current_value)
                self.state_priorities[self._state_hash(state)] = td_error

        # 4. Do background planning
        self.do_planning(goal_context)

    def do_planning(self, goal_context: Optional[torch.Tensor] = None) -> None:
        """
        Background planning: Simulate additional experience and learn.

        This is the "thinking" phase - using existing regions to imagine
        what would happen and learn from simulations.
        """
        for _ in range(self.config.n_planning_steps):
            # Sample a previous state to start simulation from
            if self.config.use_prioritized_sweeping and self.state_priorities:
                # Sample proportional to priority (TD error)
                sampled_state = self._sample_by_priority()
            else:
                # Random sample from hippocampus
                sampled_state = self._sample_random_state()

            if sampled_state is None:
                continue  # No experiences yet

            # Simulate action from this state
            available_actions = list(range(self.striatum.n_actions))
            action = random.choice(available_actions)

            # Use coordinator to simulate outcome
            rollout = self.coordinator.simulate_rollout(
                current_state=sampled_state,
                action_sequence=[action],
                goal_context=goal_context
            )

            if len(rollout.states) < 2 or len(rollout.rewards) < 1:
                continue  # Invalid rollout

            # Extract simulated experience
            sim_next_state = rollout.states[1]  # State after action
            sim_reward = rollout.rewards[0]

            # NOTE: In full implementation, would update striatum values here
            # For now, this is a placeholder for the learning update
            # The actual update would use the striatum's learning mechanism
            # with scaled learning rate for simulated experience:
            #
            # original_lr = self.striatum.config.learning_rate
            # self.striatum.config.learning_rate *= self.config.simulation_lr_scale
            #
            # self.striatum.update_value_estimate(action, sim_reward)
            #
            # self.striatum.config.learning_rate = original_lr

    def _sample_by_priority(self) -> Optional[torch.Tensor]:
        """Sample state proportional to TD error magnitude."""
        if not self.state_priorities:
            return None

        # Convert priorities to probabilities
        states = list(self.state_priorities.keys())
        priorities = torch.tensor([self.state_priorities[s] for s in states])

        if priorities.sum() == 0:
            return self._sample_random_state()

        probs = priorities / priorities.sum()

        # Sample
        idx = torch.multinomial(probs, 1).item()
        state_hash = states[idx]

        # Retrieve actual state from hippocampus
        return self._retrieve_state_by_hash(state_hash)

    def _sample_random_state(self) -> Optional[torch.Tensor]:
        """Sample random state from hippocampal memory."""
        if len(self.hippocampus.episode_buffer) == 0:
            return None

        episode = random.choice(self.hippocampus.episode_buffer)
        return episode.state

    def _state_hash(self, state: torch.Tensor) -> int:
        """Simple hash for state (for priority dict)."""
        return hash(state.cpu().numpy().tobytes())

    def _retrieve_state_by_hash(self, state_hash: int) -> Optional[torch.Tensor]:
        """Retrieve state from hippocampus by hash."""
        # Simplified - in practice would need proper state->hash mapping
        for episode in self.hippocampus.episode_buffer:
            if self._state_hash(episode.state) == state_hash:
                return episode.state
        return None
