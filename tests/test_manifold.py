"""
Tests for manifold analysis - ActivityTracker and ThoughtTrajectory.
"""

import pytest
import torch
from thalia.dynamics import ActivityTracker, ThoughtTrajectory


class TestActivityTracker:
    """Test ActivityTracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a test tracker."""
        return ActivityTracker(max_history=1000)

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.max_history == 1000
        assert len(tracker.history) == 0
        assert len(tracker.timestamps) == 0

    def test_record_single(self, tracker):
        """Test recording a single activity."""
        activity = torch.randn(100)
        tracker.record(activity)

        assert len(tracker.history) == 1
        assert len(tracker.timestamps) == 1
        assert tracker.timestamps[0] == 0

    def test_record_multiple(self, tracker):
        """Test recording multiple activities."""
        for i in range(10):
            activity = torch.randn(100)
            tracker.record(activity)

        assert len(tracker.history) == 10
        assert tracker.timestamps == list(range(10))

    def test_record_batched(self, tracker):
        """Test recording batched activity (averages over batch)."""
        activity = torch.randn(8, 100)  # Batch of 8
        tracker.record(activity)

        assert len(tracker.history) == 1
        assert tracker.history[0].shape == (100,)  # Should be unbatched

    def test_max_history_trimming(self):
        """Test that history is trimmed when exceeding max."""
        tracker = ActivityTracker(max_history=10)

        for i in range(20):
            tracker.record(torch.randn(50))

        assert len(tracker.history) == 10
        # Should keep most recent
        assert tracker.timestamps[-1] == 19

    def test_reset(self, tracker):
        """Test reset clears history."""
        for i in range(5):
            tracker.record(torch.randn(100))

        assert len(tracker.history) > 0

        tracker.reset()

        assert len(tracker.history) == 0
        assert len(tracker.timestamps) == 0
        assert tracker._t == 0

    def test_get_trajectory(self, tracker):
        """Test getting trajectory tensor."""
        for i in range(10):
            tracker.record(torch.ones(50) * i)

        trajectory = tracker.get_trajectory()

        assert trajectory.shape == (10, 50)
        assert torch.allclose(trajectory[0], torch.zeros(50))
        assert torch.allclose(trajectory[9], torch.ones(50) * 9)

    def test_get_trajectory_empty(self, tracker):
        """Test getting trajectory when empty."""
        trajectory = tracker.get_trajectory()

        assert len(trajectory) == 0

    def test_get_trajectory_smoothed(self, tracker):
        """Test getting smoothed trajectory."""
        for i in range(20):
            # Noisy signal
            tracker.record(torch.ones(50) * i + torch.randn(50) * 0.1)

        smoothed = tracker.get_trajectory(smooth_window=3)

        assert smoothed.shape == (20, 50)

    def test_project_pca(self, tracker):
        """Test PCA projection."""
        # Create data with some structure
        for i in range(100):
            activity = torch.randn(50)
            activity[0] = i  # First dimension varies linearly
            tracker.record(activity)

        coords, variance = tracker.project_pca(n_components=3)

        assert coords.shape == (100, 3)
        assert len(variance) == 3
        # First component should explain most variance
        assert variance[0] >= variance[1]

    def test_project_pca_empty(self, tracker):
        """Test PCA with empty tracker."""
        coords, variance = tracker.project_pca()

        assert len(coords) == 0
        assert len(variance) == 0

    def test_distance_to_patterns(self, tracker):
        """Test computing distance to patterns."""
        # Record some activity
        for i in range(10):
            tracker.record(torch.ones(50) * i)

        # Create patterns
        patterns = [
            torch.zeros(50),
            torch.ones(50) * 5,
            torch.ones(50) * 9,
        ]

        distances = tracker.distance_to_patterns(patterns)

        assert distances.shape == (10, 3)
        # First timestep should be closest to first pattern
        assert distances[0, 0] < distances[0, 1]
        assert distances[0, 0] < distances[0, 2]

    def test_distance_to_patterns_empty(self, tracker):
        """Test distance with empty tracker or patterns."""
        patterns = [torch.zeros(50)]

        distances = tracker.distance_to_patterns(patterns)
        assert len(distances) == 0

        # Also test empty patterns
        tracker.record(torch.randn(50))
        distances = tracker.distance_to_patterns([])
        assert len(distances) == 0

    def test_find_transitions(self, tracker):
        """Test finding transitions between attractors."""
        # Create trajectory that visits different attractors
        pattern1 = torch.zeros(50)
        pattern2 = torch.ones(50)

        # Stay near pattern1 for a bit
        for i in range(10):
            tracker.record(pattern1 + torch.randn(50) * 0.1)

        # Transition to pattern2
        for i in range(10):
            tracker.record(pattern2 + torch.randn(50) * 0.1)

        patterns = [pattern1, pattern2]
        transitions = tracker.find_transitions(patterns, threshold=0.5)

        # Should find at least one transition
        assert len(transitions) >= 1
        # Transition should be from pattern 0 to pattern 1
        if len(transitions) > 0:
            t, from_p, to_p = transitions[0]
            assert from_p == 0
            assert to_p == 1


class TestThoughtTrajectory:
    """Test ThoughtTrajectory functionality."""

    @pytest.fixture
    def trajectory(self):
        """Create a test trajectory."""
        return ThoughtTrajectory()

    def test_initialization(self, trajectory):
        """Test trajectory initialization."""
        assert len(trajectory.states) == 0
        assert len(trajectory.times) == 0
        assert len(trajectory.durations) == 0

    def test_add_state(self, trajectory):
        """Test adding states."""
        trajectory.add_state(0, time=0)
        trajectory.add_state(1, time=10)
        trajectory.add_state(2, time=25)

        assert trajectory.states == [0, 1, 2]
        assert trajectory.times == [0, 10, 25]
        assert trajectory.durations == [10, 15]  # Durations between states

    def test_add_same_state(self, trajectory):
        """Test that adding same state doesn't duplicate."""
        trajectory.add_state(0, time=0)
        trajectory.add_state(0, time=5)  # Same state
        trajectory.add_state(0, time=10)  # Same state again

        assert len(trajectory.states) == 1

    def test_get_sequence(self, trajectory):
        """Test getting state sequence."""
        trajectory.add_state(0, time=0)
        trajectory.add_state(1, time=10)
        trajectory.add_state(0, time=20)

        sequence = trajectory.get_sequence()

        assert sequence == [0, 1, 0]
        # Should be a copy
        sequence.append(99)
        assert trajectory.states == [0, 1, 0]

    def test_get_transitions(self, trajectory):
        """Test getting transition list."""
        trajectory.add_state(0, time=0)
        trajectory.add_state(1, time=10)
        trajectory.add_state(2, time=20)
        trajectory.add_state(0, time=30)

        transitions = trajectory.get_transitions()

        assert transitions == [(0, 1), (1, 2), (2, 0)]

    def test_mean_dwell_time(self, trajectory):
        """Test computing mean dwell time."""
        trajectory.add_state(0, time=0)
        trajectory.add_state(1, time=10)  # Dwell in 0 for 10
        trajectory.add_state(2, time=30)  # Dwell in 1 for 20

        mean_dwell = trajectory.mean_dwell_time()

        assert mean_dwell == 15.0  # (10 + 20) / 2

    def test_mean_dwell_time_empty(self, trajectory):
        """Test mean dwell time with no durations."""
        assert trajectory.mean_dwell_time() == 0.0

        trajectory.add_state(0, time=0)  # Single state, no transitions
        assert trajectory.mean_dwell_time() == 0.0

    def test_repr(self, trajectory):
        """Test string representation."""
        trajectory.add_state(0, time=0)
        trajectory.add_state(1, time=10)
        trajectory.add_state(2, time=20)

        repr_str = repr(trajectory)

        assert "ThoughtTrajectory" in repr_str
        assert "0 -> 1 -> 2" in repr_str


class TestIntegration:
    """Integration tests for activity tracking and thought trajectories."""

    def test_tracker_to_trajectory(self):
        """Test converting activity history to thought trajectory."""
        tracker = ActivityTracker()
        trajectory = ThoughtTrajectory()

        # Create distinct patterns
        patterns = [
            torch.zeros(50),
            torch.ones(50),
        ]

        # Simulate activity near different patterns
        for t in range(20):
            if t < 10:
                activity = patterns[0] + torch.randn(50) * 0.1
            else:
                activity = patterns[1] + torch.randn(50) * 0.1
            tracker.record(activity)

        # Find transitions and convert to trajectory
        transitions = tracker.find_transitions(patterns, threshold=0.5)

        # Build trajectory from transitions
        if len(transitions) > 0:
            # Add initial state (assume started at first transition's source)
            trajectory.add_state(transitions[0][1], time=0)
            for t, from_p, to_p in transitions:
                trajectory.add_state(to_p, time=t)

        # Should have detected the transition
        assert len(trajectory.states) >= 1
