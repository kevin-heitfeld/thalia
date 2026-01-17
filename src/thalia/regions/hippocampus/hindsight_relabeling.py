"""
Hindsight Experience Replay (HER) for Multi-Goal Learning.

Implements goal relabeling for failed episodes: "What if my actual outcome
WAS my goal?" This dramatically improves sample efficiency by learning from
every episode, even failures.

Key Idea:
- Original episode: goal=red, achieved=blue → failure (no reward)
- Hindsight relabeling: goal=blue, achieved=blue → success (reward=1)
- Learn: "In that state, action led to blue outcome" (useful!)

Biology:
- Implemented in hippocampus (episodic memory system)
- Replay during consolidation (sleep/offline periods)
- PFC provides goal context for relabeling

References:
    Andrychowicz et al. (2017): Hindsight Experience Replay
    Foster & Wilson (2006): Reverse replay in hippocampus
    Schaul et al. (2015): Universal Value Function Approximators

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from thalia.components.synapses.weight_init import WeightInitializer


class HERStrategy(Enum):
    """Strategy for selecting hindsight goals."""

    FINAL = "final"  # Use final achieved state as goal
    FUTURE = "future"  # Sample from future achieved states
    EPISODE = "episode"  # Sample from any state in episode
    RANDOM = "random"  # Sample random goal (baseline)


@dataclass
class HERConfig:
    """Configuration for Hindsight Experience Replay."""

    # Core parameters
    strategy: HERStrategy = HERStrategy.FUTURE  # Which states to use as hindsight goals
    k_hindsight: int = 4  # Number of hindsight replays per real experience
    replay_ratio: float = 0.8  # Fraction of replays that are hindsight (vs real)

    # Goal representation
    goal_dim: int = 128  # Dimension of goal vectors (should match PFC size)
    goal_tolerance: float = 0.1  # Distance threshold for "goal achieved"

    # Episode buffer
    max_episode_length: int = 100  # Maximum timesteps to store
    buffer_size: int = 1000  # Maximum episodes to keep

    # Biological constraints
    replay_during_consolidation: bool = True  # Only replay during sleep/offline
    prioritize_recent: bool = True  # Recent episodes weighted higher

    # Device
    device: str = "cpu"


@dataclass
class EpisodeTransition:
    """Single transition in an episode."""

    state: torch.Tensor  # State representation [state_dim]
    action: int  # Action taken
    next_state: torch.Tensor  # Resulting state [state_dim]
    goal: torch.Tensor  # Original goal [goal_dim]
    reward: float  # Original reward received
    done: bool  # Episode terminated?

    # Metadata
    timestep: int  # Position in episode
    achieved_goal: Optional[torch.Tensor] = None  # What was actually achieved [goal_dim]


class EpisodeBuffer:
    """Buffer for storing complete episodes for HER relabeling."""

    def __init__(self, config: HERConfig):
        self.config = config
        self.episodes: List[List[EpisodeTransition]] = []
        self.current_episode: List[EpisodeTransition] = []

    def add_transition(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        goal: torch.Tensor,
        reward: float,
        done: bool,
        achieved_goal: Optional[torch.Tensor] = None,
    ):
        """Add transition to current episode."""
        transition = EpisodeTransition(
            state=state.clone(),
            action=action,
            next_state=next_state.clone(),
            goal=goal.clone(),
            reward=reward,
            done=done,
            timestep=len(self.current_episode),
            achieved_goal=achieved_goal.clone() if achieved_goal is not None else None,
        )
        self.current_episode.append(transition)

        # Store episode when done
        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []

            # Prune old episodes
            if len(self.episodes) > self.config.buffer_size:
                self.episodes.pop(0)

    def end_episode(self):
        """Manually end current episode (even if not done)."""
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def get_episodes(self, n: Optional[int] = None) -> List[List[EpisodeTransition]]:
        """Get recent episodes."""
        if n is None:
            return self.episodes
        return self.episodes[-n:]

    def clear(self):
        """Clear all stored episodes."""
        self.episodes.clear()
        self.current_episode.clear()


class HindsightRelabeler:
    """
    Relabel failed episodes with achieved goals for learning.

    Core algorithm:
    1. Store complete episodes in buffer
    2. For each transition, generate k hindsight goals
    3. Check if those goals were achieved
    4. Create synthetic "success" experiences
    5. Return both real and hindsight experiences for learning
    """

    def __init__(self, config: HERConfig):
        self.config = config
        self.episode_buffer = EpisodeBuffer(config)

    def add_transition(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        goal: torch.Tensor,
        reward: float,
        done: bool,
        achieved_goal: Optional[torch.Tensor] = None,
    ):
        """Add transition to buffer for later relabeling."""
        # If achieved_goal not provided, use next_state as proxy
        if achieved_goal is None:
            achieved_goal = next_state

        self.episode_buffer.add_transition(
            state, action, next_state, goal, reward, done, achieved_goal
        )

    def sample_hindsight_goals(
        self, episode: List[EpisodeTransition], transition_idx: int, k: int
    ) -> List[torch.Tensor]:
        """
        Sample k hindsight goals for a transition.

        Args:
            episode: Full episode
            transition_idx: Index of transition to relabel
            k: Number of goals to sample

        Returns:
            hindsight_goals: List of k goal tensors
        """
        if self.config.strategy == HERStrategy.FINAL:
            # Use final achieved state as goal
            final_goal = episode[-1].achieved_goal
            assert final_goal is not None, "achieved_goal cannot be None for HER"
            return [final_goal] * k

        elif self.config.strategy == HERStrategy.FUTURE:
            # Sample from future achieved states in episode
            future_indices = range(transition_idx + 1, len(episode))
            if len(future_indices) == 0:
                # No future states, use final
                final_goal = episode[-1].achieved_goal
                assert final_goal is not None, "achieved_goal must be set"
                return [final_goal] * k

            # Sample k indices (with replacement if needed)
            sampled_indices = torch.randint(len(future_indices), (k,), device=self.config.device)
            goals: list[torch.Tensor] = []
            for idx in sampled_indices:
                goal = episode[transition_idx + 1 + int(idx.item())].achieved_goal
                assert goal is not None, "achieved_goal must be set"
                goals.append(goal)
            return goals

        elif self.config.strategy == HERStrategy.EPISODE:
            # Sample from any state in episode
            sampled_indices = torch.randint(len(episode), (k,), device=self.config.device)
            goals = [episode[idx.item()].achieved_goal for idx in sampled_indices]
            return goals

        elif self.config.strategy == HERStrategy.RANDOM:
            # Sample random goals (baseline - not very useful)
            return [
                WeightInitializer.gaussian(
                    self.config.goal_dim, 1, mean=0.0, std=1.0, device=self.config.device
                ).squeeze()
                for _ in range(k)
            ]

        else:
            raise ValueError(f"Unknown HER strategy: {self.config.strategy}")

    def check_goal_achieved(self, achieved_goal: torch.Tensor, target_goal: torch.Tensor) -> bool:
        """Check if achieved goal matches target goal."""
        # Convert to float for distance computation (handles bool/spike tensors)
        achieved_float = (
            achieved_goal.float() if achieved_goal.dtype == torch.bool else achieved_goal
        )
        target_float = target_goal.float() if target_goal.dtype == torch.bool else target_goal
        distance = torch.norm(achieved_float - target_float)
        return bool(distance < self.config.goal_tolerance)

    def relabel_episode(self, episode: List[EpisodeTransition]) -> List[EpisodeTransition]:
        """
        Generate hindsight experiences from episode.

        For each transition:
        1. Keep original (real) experience
        2. Generate k hindsight alternatives with achieved goals
        3. Assign reward=1 if hindsight goal was achieved, else 0

        Returns:
            augmented_transitions: Original + hindsight experiences
        """
        augmented = []

        for i, transition in enumerate(episode):
            # Add original transition
            augmented.append(transition)

            # Generate hindsight goals
            hindsight_goals = self.sample_hindsight_goals(episode, i, self.config.k_hindsight)

            # Create hindsight transitions
            for hindsight_goal in hindsight_goals:
                # Check if this goal was achieved
                achieved_goal = transition.achieved_goal
                assert achieved_goal is not None, "achieved_goal must be set"
                achieved = self.check_goal_achieved(achieved_goal, hindsight_goal)
                hindsight_reward = 1.0 if achieved else 0.0

                # Create relabeled transition
                state_clone = transition.state
                next_state = transition.next_state
                assert (
                    state_clone is not None and next_state is not None
                ), "state tensors must be set"
                hindsight_transition = EpisodeTransition(
                    state=state_clone.clone(),
                    action=transition.action,
                    next_state=next_state.clone(),
                    goal=hindsight_goal.clone(),  # RELABELED GOAL
                    reward=hindsight_reward,  # RELABELED REWARD
                    done=transition.done,
                    timestep=transition.timestep,
                    achieved_goal=achieved_goal.clone(),
                )
                augmented.append(hindsight_transition)

        return augmented

    def sample_replay_batch(
        self, batch_size: int = 32, recent_episodes: int = 10
    ) -> List[EpisodeTransition]:
        """
        Sample batch of experiences for replay learning.

        Mixes real and hindsight experiences according to replay_ratio.

        Args:
            batch_size: Number of transitions to sample
            recent_episodes: How many recent episodes to consider

        Returns:
            batch: List of transitions (real + hindsight mix)
        """
        episodes = self.episode_buffer.get_episodes(recent_episodes)
        if len(episodes) == 0:
            return []

        batch = []

        # Determine split between real and hindsight
        n_hindsight = int(batch_size * self.config.replay_ratio)
        n_real = batch_size - n_hindsight

        # Sample real experiences
        for _ in range(n_real):
            episode_idx = int(torch.randint(len(episodes), (1,)).item())
            episode = episodes[episode_idx]
            transition_idx = int(torch.randint(len(episode), (1,)).item())
            transition = episode[transition_idx]
            batch.append(transition)

        # Sample hindsight experiences
        for _ in range(n_hindsight):
            episode_idx_h = int(torch.randint(len(episodes), (1,)).item())
            episode = episodes[episode_idx_h]
            # Relabel this episode
            hindsight_transitions = self.relabel_episode(episode)
            # Sample from hindsight (skip originals)
            hindsight_only = [
                t
                for t in hindsight_transitions
                if not any(torch.equal(t.goal, orig.goal) for orig in episode)
            ]
            if len(hindsight_only) > 0:
                hindsight_idx = int(torch.randint(len(hindsight_only), (1,)).item())
                transition = hindsight_only[hindsight_idx]
                batch.append(transition)

        return batch

    def get_replay_statistics(self) -> Dict[str, Any]:
        """Get statistics about replay buffer and hindsight generation."""
        episodes = self.episode_buffer.get_episodes()

        if len(episodes) == 0:
            return {
                "n_episodes": 0,
                "n_transitions": 0,
                "avg_episode_length": 0.0,
                "success_rate": 0.0,
            }

        n_transitions = sum(len(ep) for ep in episodes)
        avg_length = n_transitions / len(episodes)

        # Count successes (original reward > 0)
        n_success = sum(1 for ep in episodes if any(t.reward > 0 for t in ep))
        success_rate = n_success / len(episodes)

        return {
            "n_episodes": len(episodes),
            "n_transitions": n_transitions,
            "avg_episode_length": avg_length,
            "success_rate": success_rate,
        }


class HippocampalHERIntegration:
    """
    Integration of HER with hippocampus for biologically-plausible replay.

    Key features:
    - Replay during consolidation (offline/sleep periods)
    - Priority replay of recent episodes
    - Goal context from PFC
    - Episodic memory storage
    """

    def __init__(self, config: HERConfig):
        self.config = config
        self.relabeler = HindsightRelabeler(config)
        self.consolidation_mode = False

    def add_experience(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        goal: torch.Tensor,
        reward: float,
        done: bool,
        achieved_goal: Optional[torch.Tensor] = None,
    ):
        """Add experience during active learning."""
        self.relabeler.add_transition(state, action, next_state, goal, reward, done, achieved_goal)

    def enter_consolidation(self):
        """Enter consolidation mode (sleep/offline)."""
        self.consolidation_mode = True

    def exit_consolidation(self):
        """Exit consolidation mode (wake/online)."""
        self.consolidation_mode = False

    def replay_for_learning(self, batch_size: int = 32) -> List[EpisodeTransition]:
        """
        Generate replay batch for learning.

        During consolidation: Sample from buffer with hindsight relabeling
        During active learning: Return empty (no replay)

        Returns:
            batch: Transitions to learn from
        """
        if not self.consolidation_mode:
            return []

        return self.relabeler.sample_replay_batch(batch_size)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics about HER system."""
        stats = self.relabeler.get_replay_statistics()
        stats["consolidation_mode"] = self.consolidation_mode
        stats["strategy"] = self.config.strategy.value
        stats["k_hindsight"] = self.config.k_hindsight
        return stats
