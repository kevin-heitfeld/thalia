"""
Tests for ExplorationManager

Tests the exploration management functionality including UCB tracking,
adaptive tonic dopamine adjustment, and state management.
"""

import pytest
import torch

from thalia.regions.striatum.exploration import (
    ExplorationManager,
    ExplorationConfig,
)


@pytest.fixture
def device():
    """Device for testing (CPU)."""
    return torch.device("cpu")


@pytest.fixture
def exploration_config():
    """Default exploration configuration."""
    return ExplorationConfig(
        ucb_exploration=True,
        ucb_coefficient=1.0,
        adaptive_exploration=True,
        performance_window=20,
        min_tonic_dopamine=0.0,
        max_tonic_dopamine=0.3,
        tonic_modulates_exploration=True,
        tonic_exploration_scale=0.5,
    )


@pytest.fixture
def exploration_manager(device, exploration_config):
    """Create exploration manager for testing."""
    return ExplorationManager(
        n_actions=2,
        config=exploration_config,
        device=device,
        initial_tonic_dopamine=0.1,
    )


class TestExplorationManagerInit:
    """Test exploration manager initialization."""

    def test_initialization(self, exploration_manager, device):
        """Test that exploration manager initializes correctly."""
        assert exploration_manager.n_actions == 2
        assert exploration_manager.device == device
        assert exploration_manager.tonic_dopamine == 0.1
        assert exploration_manager._total_trials == 0
        assert len(exploration_manager._recent_rewards) == 0
        assert exploration_manager._recent_accuracy == 0.0
        assert exploration_manager._action_counts.shape == (2,)
        assert exploration_manager._action_counts.sum() == 0


class TestUCBTracking:
    """Test UCB (Upper Confidence Bound) tracking."""

    def test_update_action_counts(self, exploration_manager):
        """Test that action counts are updated correctly."""
        # Take action 0 three times
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(0)

        assert exploration_manager._action_counts[0] == 3
        assert exploration_manager._action_counts[1] == 0
        assert exploration_manager._total_trials == 3

        # Take action 1 once
        exploration_manager.update_action_counts(1)

        assert exploration_manager._action_counts[0] == 3
        assert exploration_manager._action_counts[1] == 1
        assert exploration_manager._total_trials == 4

    def test_compute_ucb_bonus_zero_trials(self, exploration_manager):
        """Test UCB bonus when no trials have been taken."""
        ucb_bonus = exploration_manager.compute_ucb_bonus()

        assert ucb_bonus.shape == (2,)
        assert torch.all(ucb_bonus == 0)

    def test_compute_ucb_bonus_after_trials(self, exploration_manager):
        """Test UCB bonus increases for less-tried actions."""
        # Take action 0 many times, action 1 once
        for _ in range(10):
            exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(1)

        ucb_bonus = exploration_manager.compute_ucb_bonus()

        # Action 1 should have higher bonus (tried less)
        assert ucb_bonus[1] > ucb_bonus[0]
        assert ucb_bonus[0] > 0  # Both should be positive
        assert ucb_bonus[1] > 0

    def test_compute_ucb_bonus_disabled(self, device):
        """Test UCB bonus returns zeros when disabled."""
        config = ExplorationConfig(ucb_exploration=False)
        manager = ExplorationManager(
            n_actions=2,
            config=config,
            device=device,
            initial_tonic_dopamine=0.1,
        )

        # Take some actions
        manager.update_action_counts(0)
        manager.update_action_counts(1)

        ucb_bonus = manager.compute_ucb_bonus()

        assert torch.all(ucb_bonus == 0)


class TestAdaptiveExploration:
    """Test adaptive tonic dopamine adjustment."""

    def test_adjust_tonic_dopamine_success(self, exploration_manager):
        """Test tonic DA decreases with successful performance."""
        initial_tonic = exploration_manager.tonic_dopamine

        # Simulate several successful trials (reward > 0)
        for _ in range(10):
            result = exploration_manager.adjust_tonic_dopamine(reward=1.0)

        # Tonic DA should decrease (more exploitation)
        assert exploration_manager.tonic_dopamine < initial_tonic
        assert result["new_tonic"] < result["old_tonic"]
        assert result["adaptive_exploration_enabled"] is True

    def test_adjust_tonic_dopamine_failure(self, exploration_manager):
        """Test tonic DA increases with poor performance."""
        initial_tonic = exploration_manager.tonic_dopamine

        # Simulate several failed trials (reward <= 0)
        for _ in range(10):
            result = exploration_manager.adjust_tonic_dopamine(reward=0.0)

        # Tonic DA should increase (more exploration)
        assert exploration_manager.tonic_dopamine > initial_tonic
        assert result["new_tonic"] > result["old_tonic"]

    def test_adjust_tonic_dopamine_clamping(self, exploration_manager):
        """Test tonic DA is clamped to valid range."""
        config = exploration_manager.config

        # Try to drive tonic DA very low with successes
        for _ in range(100):
            exploration_manager.adjust_tonic_dopamine(reward=1.0)

        assert exploration_manager.tonic_dopamine >= config.min_tonic_dopamine

        # Try to drive tonic DA very high with failures
        for _ in range(100):
            exploration_manager.adjust_tonic_dopamine(reward=0.0)

        assert exploration_manager.tonic_dopamine <= config.max_tonic_dopamine

    def test_performance_window(self, exploration_manager):
        """Test that only recent trials are tracked."""
        window = exploration_manager.config.performance_window

        # Add more rewards than window size
        for _ in range(window + 10):
            exploration_manager.adjust_tonic_dopamine(reward=1.0)

        # Should only keep window size
        assert len(exploration_manager._recent_rewards) == window

    def test_recent_accuracy_calculation(self, exploration_manager):
        """Test accuracy calculation from recent rewards."""
        # 7 successes, 3 failures
        for _ in range(7):
            exploration_manager.adjust_tonic_dopamine(reward=1.0)
        for _ in range(3):
            exploration_manager.adjust_tonic_dopamine(reward=0.0)

        # Accuracy should be 0.7
        assert abs(exploration_manager._recent_accuracy - 0.7) < 0.01

    def test_adaptive_exploration_disabled(self, device):
        """Test that tonic DA doesn't change when adaptive exploration is disabled."""
        config = ExplorationConfig(adaptive_exploration=False)
        manager = ExplorationManager(
            n_actions=2,
            config=config,
            device=device,
            initial_tonic_dopamine=0.15,
        )

        initial_tonic = manager.tonic_dopamine

        # Try to adjust with rewards
        result = manager.adjust_tonic_dopamine(reward=1.0)

        assert manager.tonic_dopamine == initial_tonic
        assert result["adaptive_exploration_enabled"] is False


class TestStateManagement:
    """Test state saving and loading."""

    def test_get_state(self, exploration_manager):
        """Test state extraction."""
        # Set up some state
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(1)
        exploration_manager.adjust_tonic_dopamine(reward=1.0)

        state = exploration_manager.get_state()

        assert "action_counts" in state
        assert "total_trials" in state
        assert "recent_rewards" in state
        assert "recent_accuracy" in state
        assert "tonic_dopamine" in state

        assert state["total_trials"] == 2
        assert len(state["recent_rewards"]) == 1

    def test_load_state(self, device, exploration_config):
        """Test state restoration."""
        # Create manager and build up state
        manager1 = ExplorationManager(
            n_actions=2,
            config=exploration_config,
            device=device,
            initial_tonic_dopamine=0.1,
        )

        manager1.update_action_counts(0)
        manager1.update_action_counts(0)
        manager1.update_action_counts(1)
        manager1.adjust_tonic_dopamine(reward=1.0)
        manager1.adjust_tonic_dopamine(reward=0.0)

        state1 = manager1.get_state()

        # Create new manager and load state
        manager2 = ExplorationManager(
            n_actions=2,
            config=exploration_config,
            device=device,
            initial_tonic_dopamine=0.2,  # Different initial value
        )

        manager2.load_state(state1)

        # Check state matches
        assert torch.all(manager2._action_counts == manager1._action_counts)
        assert manager2._total_trials == manager1._total_trials
        assert manager2._recent_rewards == manager1._recent_rewards
        assert manager2._recent_accuracy == manager1._recent_accuracy
        assert manager2.tonic_dopamine == manager1.tonic_dopamine

    def test_reset(self, exploration_manager):
        """Test state reset."""
        # Build up state
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(1)
        exploration_manager.adjust_tonic_dopamine(reward=1.0)

        # Reset
        exploration_manager.reset()

        # Check everything is cleared
        assert exploration_manager._total_trials == 0
        assert torch.all(exploration_manager._action_counts == 0)
        assert len(exploration_manager._recent_rewards) == 0
        assert exploration_manager._recent_accuracy == 0.0
        assert exploration_manager.tonic_dopamine == 0.1  # Reset to default


class TestGrowth:
    """Test growing action space."""

    def test_grow_actions(self, exploration_manager):
        """Test adding new actions."""
        # Build up state for existing actions
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(1)
        old_counts = exploration_manager._action_counts.clone()

        # Grow to 4 actions
        exploration_manager.grow(new_n_actions=4)

        # Check expansion
        assert exploration_manager.n_actions == 4
        assert exploration_manager._action_counts.shape == (4,)

        # Old counts preserved
        assert torch.all(exploration_manager._action_counts[:2] == old_counts)

        # New counts initialized to zero
        assert exploration_manager._action_counts[2] == 0
        assert exploration_manager._action_counts[3] == 0

    def test_grow_no_change(self, exploration_manager):
        """Test that growing to same size is a no-op."""
        old_counts = exploration_manager._action_counts.clone()

        exploration_manager.grow(new_n_actions=2)

        assert exploration_manager.n_actions == 2
        assert torch.all(exploration_manager._action_counts == old_counts)

    def test_grow_shrink_error(self, exploration_manager):
        """Test that shrinking raises an error."""
        with pytest.raises(ValueError, match="Cannot shrink actions"):
            exploration_manager.grow(new_n_actions=1)


class TestDiagnostics:
    """Test diagnostic information."""

    def test_get_diagnostics(self, exploration_manager):
        """Test diagnostic output."""
        # Build up some state
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(0)
        exploration_manager.update_action_counts(1)
        exploration_manager.adjust_tonic_dopamine(reward=1.0)

        diagnostics = exploration_manager.get_diagnostics()

        assert "action_counts" in diagnostics
        assert "total_trials" in diagnostics
        assert "recent_accuracy" in diagnostics
        assert "tonic_dopamine" in diagnostics
        assert "least_tried_action" in diagnostics
        assert "most_tried_action" in diagnostics

        assert diagnostics["total_trials"] == 4
        assert diagnostics["least_tried_action"] == 1  # Action 1 tried once
        assert diagnostics["most_tried_action"] == 0  # Action 0 tried three times
        assert diagnostics["action_counts"] == [3, 1]
