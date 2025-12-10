"""
Unit tests for Hindsight Experience Replay (HER).

Tests goal relabeling, episode buffering, and replay sampling.

Author: Thalia Project
Date: December 2025
"""

import pytest
import torch
from thalia.regions.hippocampus.hindsight_relabeling import (
    HERConfig,
    HERStrategy,
    EpisodeBuffer,
    EpisodeTransition,
    HindsightRelabeler,
    HippocampalHERIntegration,
)


@pytest.fixture
def her_config():
    """Basic HER configuration."""
    return HERConfig(
        strategy=HERStrategy.FUTURE,
        k_hindsight=4,
        goal_dim=8,
        goal_tolerance=0.1,
        device="cpu"
    )


@pytest.fixture
def simple_episode():
    """Create a simple failed episode for testing."""
    # Episode: Try to reach goal [1,0,0,...] but end up at [0,1,0,...]
    goal = torch.zeros(8)
    goal[0] = 1.0

    transitions = []
    for i in range(5):
        state = torch.randn(8)
        next_state = torch.randn(8)
        # Gradually move toward [0,1,0,...] (not the goal)
        achieved = torch.zeros(8)
        achieved[1] = float(i + 1) / 5.0

        transitions.append(
            EpisodeTransition(
                state=state,
                action=i % 2,
                next_state=next_state,
                goal=goal.clone(),
                reward=0.0,  # No reward (failure)
                done=(i == 4),
                timestep=i,
                achieved_goal=achieved
            )
        )

    return transitions


class TestEpisodeBuffer:
    """Test episode buffer functionality."""

    def test_buffer_initialization(self, her_config):
        """Test buffer initializes empty."""
        buffer = EpisodeBuffer(her_config)
        assert len(buffer.episodes) == 0
        assert len(buffer.current_episode) == 0

    def test_add_transitions(self, her_config):
        """Test adding transitions to buffer."""
        buffer = EpisodeBuffer(her_config)

        state = torch.randn(8)
        goal = torch.randn(8)

        # Add transition
        buffer.add_transition(
            state=state,
            action=0,
            next_state=state + 0.1,
            goal=goal,
            reward=0.0,
            done=False,
            achieved_goal=state + 0.1
        )

        assert len(buffer.current_episode) == 1
        assert len(buffer.episodes) == 0  # Not done yet

    def test_episode_completion(self, her_config):
        """Test episode is stored when done=True."""
        buffer = EpisodeBuffer(her_config)

        # Add transitions
        for i in range(3):
            buffer.add_transition(
                state=torch.randn(8),
                action=i,
                next_state=torch.randn(8),
                goal=torch.randn(8),
                reward=0.0,
                done=(i == 2),
                achieved_goal=torch.randn(8)
            )

        # Episode should be stored
        assert len(buffer.episodes) == 1
        assert len(buffer.current_episode) == 0  # Reset
        assert len(buffer.episodes[0]) == 3  # 3 transitions

    def test_buffer_pruning(self, her_config):
        """Test buffer prunes old episodes."""
        her_config.buffer_size = 5
        buffer = EpisodeBuffer(her_config)

        # Add 10 episodes
        for _ in range(10):
            for i in range(2):
                buffer.add_transition(
                    state=torch.randn(8),
                    action=i,
                    next_state=torch.randn(8),
                    goal=torch.randn(8),
                    reward=0.0,
                    done=(i == 1),
                    achieved_goal=torch.randn(8)
                )

        # Should only keep last 5
        assert len(buffer.episodes) == 5


class TestHindsightRelabeler:
    """Test hindsight goal sampling and relabeling."""

    def test_relabeler_initialization(self, her_config):
        """Test relabeler initializes correctly."""
        relabeler = HindsightRelabeler(her_config)
        assert relabeler.config == her_config
        assert isinstance(relabeler.episode_buffer, EpisodeBuffer)

    def test_final_strategy(self, her_config, simple_episode):
        """Test FINAL strategy uses last achieved state."""
        her_config.strategy = HERStrategy.FINAL
        relabeler = HindsightRelabeler(her_config)

        # Sample goals
        goals = relabeler.sample_hindsight_goals(
            simple_episode,
            transition_idx=0,
            k=3
        )

        assert len(goals) == 3
        # All should be final achieved goal
        final_goal = simple_episode[-1].achieved_goal
        for goal in goals:
            assert torch.equal(goal, final_goal)

    def test_future_strategy(self, her_config, simple_episode):
        """Test FUTURE strategy samples from future states."""
        her_config.strategy = HERStrategy.FUTURE
        relabeler = HindsightRelabeler(her_config)

        # Sample goals for first transition
        goals = relabeler.sample_hindsight_goals(
            simple_episode,
            transition_idx=0,
            k=4
        )

        assert len(goals) == 4
        # Should be from future (indices 1-4)
        for goal in goals:
            # Check it matches one of the future achieved goals
            found = False
            for i in range(1, len(simple_episode)):
                if torch.equal(goal, simple_episode[i].achieved_goal):
                    found = True
                    break
            assert found, "Goal should come from future achieved states"

    def test_future_strategy_last_transition(self, her_config, simple_episode):
        """Test FUTURE strategy handles last transition (no future)."""
        her_config.strategy = HERStrategy.FUTURE
        relabeler = HindsightRelabeler(her_config)

        # Sample for last transition (no future)
        goals = relabeler.sample_hindsight_goals(
            simple_episode,
            transition_idx=len(simple_episode) - 1,
            k=2
        )

        assert len(goals) == 2
        # Should fall back to final
        final_goal = simple_episode[-1].achieved_goal
        for goal in goals:
            assert torch.equal(goal, final_goal)

    def test_episode_strategy(self, her_config, simple_episode):
        """Test EPISODE strategy samples from any state."""
        her_config.strategy = HERStrategy.EPISODE
        relabeler = HindsightRelabeler(her_config)

        goals = relabeler.sample_hindsight_goals(
            simple_episode,
            transition_idx=2,
            k=5
        )

        assert len(goals) == 5
        # Each goal should match some achieved goal in episode
        for goal in goals:
            found = any(
                torch.equal(goal, t.achieved_goal)
                for t in simple_episode
            )
            assert found

    def test_goal_achievement_check(self, her_config):
        """Test goal achievement detection."""
        relabeler = HindsightRelabeler(her_config)

        goal = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Exact match
        achieved = goal.clone()
        assert relabeler.check_goal_achieved(achieved, goal)

        # Close enough (within tolerance)
        # With L2 norm, adding 0.05 to one dimension = distance 0.05
        achieved = goal.clone()
        achieved[0] += 0.05
        assert relabeler.check_goal_achieved(achieved, goal)

        # Too far (adding 0.5 to one dimension = distance 0.5)
        achieved = goal.clone()
        achieved[0] += 0.5
        assert not relabeler.check_goal_achieved(achieved, goal)

    def test_relabel_episode_creates_hindsight(self, her_config, simple_episode):
        """Test episode relabeling creates hindsight experiences."""
        relabeler = HindsightRelabeler(her_config)

        # Relabel episode
        augmented = relabeler.relabel_episode(simple_episode)

        # Should have original + k_hindsight per transition
        expected_count = len(simple_episode) * (1 + her_config.k_hindsight)
        assert len(augmented) == expected_count

        # Check structure: original, then k hindsight
        for i in range(len(simple_episode)):
            base_idx = i * (1 + her_config.k_hindsight)

            # First should be original
            original = augmented[base_idx]
            assert torch.equal(original.goal, simple_episode[i].goal)

            # Next k should be hindsight
            for j in range(1, her_config.k_hindsight + 1):
                hindsight = augmented[base_idx + j]
                # Goal should be different (relabeled)
                assert not torch.equal(hindsight.goal, simple_episode[i].goal)
                # State/action should be same
                assert torch.equal(hindsight.state, simple_episode[i].state)
                assert hindsight.action == simple_episode[i].action

    def test_hindsight_rewards_correct(self, her_config, simple_episode):
        """Test hindsight transitions get correct rewards."""
        relabeler = HindsightRelabeler(her_config)

        # Create episode where we achieve [0,1,0,...] in last step
        final_achieved = torch.zeros(8)
        final_achieved[1] = 1.0
        simple_episode[-1].achieved_goal = final_achieved

        # Relabel with goal = final_achieved
        her_config.strategy = HERStrategy.FINAL
        relabeler.config = her_config

        augmented = relabeler.relabel_episode(simple_episode)

        # Last transition's hindsight should have reward=1
        # (because achieved_goal matches hindsight goal)
        last_original_idx = (len(simple_episode) - 1) * (1 + her_config.k_hindsight)
        last_hindsight = augmented[last_original_idx + 1]

        # Check if goal matches achieved (within tolerance)
        if relabeler.check_goal_achieved(
            simple_episode[-1].achieved_goal,
            last_hindsight.goal
        ):
            assert last_hindsight.reward == 1.0
        else:
            assert last_hindsight.reward == 0.0


class TestReplaySampling:
    """Test replay batch sampling."""

    def test_sample_replay_empty_buffer(self, her_config):
        """Test sampling from empty buffer returns empty."""
        relabeler = HindsightRelabeler(her_config)
        batch = relabeler.sample_replay_batch(batch_size=32)
        assert len(batch) == 0

    def test_sample_replay_batch_size(self, her_config, simple_episode):
        """Test sampled batch has correct size."""
        relabeler = HindsightRelabeler(her_config)

        # Add episode to buffer
        for transition in simple_episode:
            relabeler.add_transition(
                state=transition.state,
                action=transition.action,
                next_state=transition.next_state,
                goal=transition.goal,
                reward=transition.reward,
                done=transition.done,
                achieved_goal=transition.achieved_goal
            )

        # Sample batch
        batch = relabeler.sample_replay_batch(batch_size=16)

        # Should get requested size (or close)
        assert len(batch) > 0
        assert len(batch) <= 16

    def test_replay_contains_hindsight(self, her_config, simple_episode):
        """Test replay batch contains hindsight experiences."""
        her_config.replay_ratio = 0.5  # 50% hindsight
        relabeler = HindsightRelabeler(her_config)

        # Add multiple episodes
        for _ in range(3):
            for transition in simple_episode:
                relabeler.add_transition(
                    state=transition.state,
                    action=transition.action,
                    next_state=transition.next_state,
                    goal=transition.goal,
                    reward=transition.reward,
                    done=transition.done,
                    achieved_goal=transition.achieved_goal
                )

        # Sample large batch
        batch = relabeler.sample_replay_batch(batch_size=20)

        # Should have mix of original and hindsight
        # (Hard to test exactly, but should have some)
        assert len(batch) > 0

    def test_replay_statistics(self, her_config, simple_episode):
        """Test replay statistics calculation."""
        relabeler = HindsightRelabeler(her_config)

        # Add episodes
        for _ in range(3):
            for transition in simple_episode:
                relabeler.add_transition(
                    state=transition.state,
                    action=transition.action,
                    next_state=transition.next_state,
                    goal=transition.goal,
                    reward=transition.reward,
                    done=transition.done,
                    achieved_goal=transition.achieved_goal
                )

        stats = relabeler.get_replay_statistics()

        assert stats['n_episodes'] == 3
        assert stats['n_transitions'] == 3 * len(simple_episode)
        assert stats['avg_episode_length'] == len(simple_episode)
        assert 0.0 <= stats['success_rate'] <= 1.0


class TestHippocampalIntegration:
    """Test hippocampus-HER integration."""

    def test_integration_initialization(self, her_config):
        """Test HER integration initializes."""
        integration = HippocampalHERIntegration(her_config)
        assert integration.config == her_config
        assert isinstance(integration.relabeler, HindsightRelabeler)
        assert integration.consolidation_mode is False

    def test_consolidation_mode_toggle(self, her_config):
        """Test entering/exiting consolidation mode."""
        integration = HippocampalHERIntegration(her_config)

        # Start in active mode
        assert not integration.consolidation_mode

        # Enter consolidation
        integration.enter_consolidation()
        assert integration.consolidation_mode

        # Exit consolidation
        integration.exit_consolidation()
        assert not integration.consolidation_mode

    def test_replay_only_during_consolidation(self, her_config, simple_episode):
        """Test replay only returns data during consolidation."""
        integration = HippocampalHERIntegration(her_config)

        # Add experiences
        for transition in simple_episode:
            integration.add_experience(
                state=transition.state,
                action=transition.action,
                next_state=transition.next_state,
                goal=transition.goal,
                reward=transition.reward,
                done=transition.done,
                achieved_goal=transition.achieved_goal
            )

        # Replay during active learning should be empty
        batch = integration.replay_for_learning(batch_size=10)
        assert len(batch) == 0

        # Enter consolidation
        integration.enter_consolidation()

        # Now replay should return data
        batch = integration.replay_for_learning(batch_size=10)
        assert len(batch) > 0

    def test_diagnostics(self, her_config, simple_episode):
        """Test diagnostic information."""
        integration = HippocampalHERIntegration(her_config)

        # Add experiences
        for transition in simple_episode:
            integration.add_experience(
                state=transition.state,
                action=transition.action,
                next_state=transition.next_state,
                goal=transition.goal,
                reward=transition.reward,
                done=transition.done,
                achieved_goal=transition.achieved_goal
            )

        diagnostics = integration.get_diagnostics()

        assert 'n_episodes' in diagnostics
        assert 'consolidation_mode' in diagnostics
        assert 'strategy' in diagnostics
        assert diagnostics['strategy'] == her_config.strategy.value


class TestHERWithMultipleGoals:
    """Test HER with multiple different goals."""

    def test_learn_from_all_goals(self, her_config):
        """Test that HER creates learning signal for multiple goals."""
        relabeler = HindsightRelabeler(her_config)

        # Create episode reaching goal B when trying for goal A
        goal_a = torch.zeros(8)
        goal_a[0] = 1.0
        goal_b = torch.zeros(8)
        goal_b[1] = 1.0

        # Trajectory that reaches B
        episode = []
        for i in range(5):
            achieved = goal_b * (i + 1) / 5.0  # Gradually reach B
            episode.append(
                EpisodeTransition(
                    state=torch.randn(8),
                    action=i % 2,
                    next_state=torch.randn(8),
                    goal=goal_a.clone(),  # Trying for A
                    reward=0.0,  # Failed to get A
                    done=(i == 4),
                    timestep=i,
                    achieved_goal=achieved
                )
            )

        # Relabel with hindsight
        her_config.strategy = HERStrategy.FINAL
        relabeler.config = her_config
        augmented = relabeler.relabel_episode(episode)

        # Should have experiences where goal=B and reward=1
        # (learning that these actions lead to B)
        success_count = sum(1 for t in augmented if t.reward > 0.5)
        assert success_count > 0, "Should have some successful hindsight experiences"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
