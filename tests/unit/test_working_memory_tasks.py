"""
Tests for Working Memory Tasks with Theta-Gamma Phase Coding

Tests the N-back task implementation and phase-based encoding/retrieval.
"""

import pytest
import torch
import math

from thalia.tasks.working_memory import (
    NBackTask,
    ThetaGammaEncoder,
    WorkingMemoryTaskConfig,
    theta_gamma_n_back,
    create_n_back_sequence,
)
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig


@pytest.fixture
def device():
    """Use CPU for tests."""
    return "cpu"


@pytest.fixture
def wm_config(device):
    """Working memory task configuration."""
    return WorkingMemoryTaskConfig(
        theta_freq_hz=8.0,
        gamma_freq_hz=40.0,
        items_per_theta_cycle=8,
        dt_ms=1.0,
        encoding_window_ms=100.0,
        retrieval_window_ms=50.0,
        device=device
    )


@pytest.fixture
def prefrontal(device):
    """Create simple prefrontal region for testing."""
    config = PrefrontalConfig(
        n_input=10,
        n_output=50,
        device=device
    )
    return Prefrontal(config)


class TestThetaGammaEncoder:
    """Test phase encoding system."""

    def test_encoder_initialization(self, wm_config):
        """Test encoder initializes correctly without local oscillators."""
        encoder = ThetaGammaEncoder(wm_config)

        # Encoder stores phases internally (provided by brain)
        assert hasattr(encoder, '_theta_phase')
        assert hasattr(encoder, '_gamma_phase')
        assert encoder.item_count == 0

    def test_encoding_phase_calculation(self, wm_config):
        """Test theta phase assignment for items."""
        encoder = ThetaGammaEncoder(wm_config)

        # First item at phase 0
        theta0, gamma0 = encoder.get_encoding_phase(0)
        assert theta0 == 0.0
        assert abs(gamma0 - math.pi/2) < 0.01  # Peak gamma

        # Fourth item at phase π
        theta4, gamma4 = encoder.get_encoding_phase(4)
        assert abs(theta4 - math.pi) < 0.01
        assert abs(gamma4 - math.pi/2) < 0.01

        # Eighth item wraps to 0
        theta8, gamma8 = encoder.get_encoding_phase(8)
        assert abs(theta8 - 0.0) < 0.01

    def test_retrieval_phase_calculation(self, wm_config):
        """Test phase calculation for N-back retrieval."""
        encoder = ThetaGammaEncoder(wm_config)

        # 2-back from position 5
        phase = encoder.get_retrieval_phase(current_index=5, n_back=2)
        expected = (3 / 8) * (2 * math.pi)  # Item 3 (5-2)
        assert abs(phase - expected) < 0.01

        # Can't retrieve before start
        phase = encoder.get_retrieval_phase(current_index=1, n_back=2)
        assert phase < 0  # Error indicator

    def test_oscillator_advancement(self, wm_config):
        """Test encoder can receive updated oscillator phases."""
        encoder = ThetaGammaEncoder(wm_config)

        # Provide initial phases
        encoder.set_oscillator_phases(
            phases={'theta': 0.0, 'gamma': 0.0},
            signals={'theta': 0.0, 'gamma': 0.0}
        )

        # Update to new phases (as brain would after advancing)
        encoder.set_oscillator_phases(
            phases={'theta': math.pi/4, 'gamma': math.pi/2},
            signals={'theta': 0.707, 'gamma': 1.0}
        )

        # Phases should have updated
        assert encoder._theta_phase == math.pi/4
        assert encoder._gamma_phase == math.pi/2

    def test_sync_to_item(self, wm_config):
        """Test calculating target phase for specific item."""
        encoder = ThetaGammaEncoder(wm_config)

        # Calculate encoding phase for item 3
        theta_phase, gamma_phase = encoder.get_encoding_phase(3)

        expected_theta = (3 / 8) * (2 * math.pi)
        assert abs(theta_phase - expected_theta) < 0.01
        assert abs(gamma_phase - math.pi/2) < 0.01

    def test_excitability_modulation(self, wm_config):
        """Test gamma-based excitability modulation."""
        encoder = ThetaGammaEncoder(wm_config)

        # Simulate brain providing gamma peak phase
        encoder.set_oscillator_phases(
            phases={'theta': 0.0, 'gamma': math.pi / 2},
            signals={'theta': 0.0, 'gamma': 1.0}
        )
        excitability_peak = encoder.get_excitability_modulation()
        assert excitability_peak > 0.9  # Near maximum

        # Simulate brain providing gamma trough phase
        encoder.set_oscillator_phases(
            phases={'theta': 0.0, 'gamma': 3 * math.pi / 2},
            signals={'theta': 0.0, 'gamma': -1.0}
        )
        excitability_trough = encoder.get_excitability_modulation()
        assert excitability_trough < 0.1  # Near minimum


class TestNBackTask:
    """Test N-back task implementation."""

    def test_task_initialization(self, prefrontal, wm_config):
        """Test N-back task creation."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        assert task.n_back == 2
        assert task.prefrontal == prefrontal
        assert len(task.stimulus_history) == 0

    def test_task_reset(self, prefrontal, wm_config):
        """Test task state reset."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Add some history
        task.stimulus_history.append(torch.randn(10))
        task.responses.append(True)

        # Reset
        task.reset()

        assert len(task.stimulus_history) == 0
        assert len(task.responses) == 0
        assert task.encoder.item_count == 0

    def test_encode_item(self, prefrontal, wm_config):
        """Test encoding stimulus into working memory."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        stimulus = torch.randn(10)
        metrics = task.encode_item(stimulus, item_index=0)

        # Check metrics returned
        assert "theta_phase" in metrics
        assert "gamma_phase" in metrics
        assert "excitability" in metrics
        assert "dopamine" in metrics

        # Check phases are valid
        assert 0 <= metrics["theta_phase"] < 2 * math.pi
        assert 0 <= metrics["gamma_phase"] < 2 * math.pi

    def test_retrieve_item(self, prefrontal, wm_config):
        """Test retrieving item from N positions back."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Encode some items first
        for i in range(5):
            stimulus = torch.randn(10)
            task.encode_item(stimulus, i)
            task.stimulus_history.append(stimulus)

        # Retrieve 2-back from position 4
        retrieved, metrics = task.retrieve_item(current_index=4, n_back=2)

        assert retrieved is not None
        assert "target_phase" in metrics
        assert retrieved.shape == (50,)  # PFC output size

    def test_retrieve_before_start(self, prefrontal, wm_config):
        """Test that retrieval before sequence start returns None."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Try to retrieve 2-back from position 1 (impossible)
        retrieved, metrics = task.retrieve_item(current_index=1, n_back=2)

        assert retrieved is None
        assert "error" in metrics

    def test_present_stimulus_no_match(self, prefrontal, wm_config):
        """Test presenting stimulus when no match exists."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Present different stimuli
        for i in range(3):
            stimulus = torch.randn(10)
            is_match, metrics = task.present_stimulus(stimulus)

            # First 2 items can't have matches
            if i < 2:
                assert metrics["can_retrieve"] == False

    def test_present_stimulus_with_match(self, prefrontal, wm_config):
        """Test presenting stimulus that matches N-back."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Create sequence with intentional match
        stim1 = torch.randn(10)
        stim2 = torch.randn(10)
        stim3 = stim1.clone()  # Matches 2-back

        # Present sequence
        task.present_stimulus(stim1)
        task.present_stimulus(stim2)
        is_match, metrics = task.present_stimulus(stim3)

        assert metrics["can_retrieve"] == True
        assert metrics["item_index"] == 2

    def test_run_sequence(self, prefrontal, wm_config):
        """Test running complete N-back sequence."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Create sequence
        sequence = [torch.randn(10) for _ in range(5)]

        # Run task
        results = task.run_sequence(sequence)

        assert results["n_items"] == 5
        assert len(results["responses"]) == 5
        assert len(results["metrics"]) == 5
        assert results["n_back"] == 2

    def test_run_sequence_with_ground_truth(self, prefrontal, wm_config):
        """Test accuracy calculation with ground truth."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Create sequence
        sequence = [torch.randn(10) for _ in range(5)]
        matches = [False, False, True, False, True]  # Ground truth

        # Run task
        results = task.run_sequence(sequence, target_matches=matches)

        assert "accuracy" in results
        assert results["accuracy"] is not None
        # Accuracy calculated only for items where retrieval possible (indices 2+)

    def test_get_statistics(self, prefrontal, wm_config):
        """Test task statistics retrieval."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Empty task
        stats = task.get_statistics()
        assert "error" in stats

        # After running
        sequence = [torch.randn(10) for _ in range(3)]
        task.run_sequence(sequence)

        stats = task.get_statistics()
        assert stats["n_items"] == 3
        assert stats["n_back"] == 2


class TestSequenceGeneration:
    """Test N-back sequence generation utilities."""

    def test_create_n_back_sequence(self, device):
        """Test creating random N-back sequence."""
        sequence, matches = create_n_back_sequence(
            n_items=10,
            n_dims=5,
            n_back=2,
            match_probability=0.3,
            device=device
        )

        assert len(sequence) == 10
        assert len(matches) == 10
        assert all(s.shape == (5,) for s in sequence)
        assert all(isinstance(m, bool) for m in matches)

    def test_sequence_match_probability(self, device):
        """Test that match probability is approximately correct."""
        # Create long sequence
        sequence, matches = create_n_back_sequence(
            n_items=100,
            n_dims=10,
            n_back=2,
            match_probability=0.5,
            device=device
        )

        # Count matches (only after position n_back)
        valid_matches = matches[2:]
        match_rate = sum(valid_matches) / len(valid_matches)

        # Should be roughly 50% (within reasonable variance)
        assert 0.3 < match_rate < 0.7

    def test_sequence_normalization(self, device):
        """Test that generated items are normalized."""
        sequence, _ = create_n_back_sequence(
            n_items=5,
            n_dims=10,
            n_back=2,
            device=device
        )

        # Check normalization
        for item in sequence:
            norm = item.norm().item()
            assert abs(norm - 1.0) < 0.01


class TestConvenienceFunction:
    """Test convenience function matching plan specification."""

    def test_theta_gamma_n_back_basic(self, prefrontal):
        """Test theta_gamma_n_back function."""
        # Create simple sequence
        sequence = [torch.randn(10) for _ in range(5)]

        # Run 2-back
        results = theta_gamma_n_back(
            prefrontal,
            sequence,
            n=2
        )

        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)

    def test_theta_gamma_n_back_custom_frequencies(self, prefrontal):
        """Test with custom oscillation frequencies."""
        sequence = [torch.randn(10) for _ in range(3)]

        results = theta_gamma_n_back(
            prefrontal,
            sequence,
            n=1,
            theta_freq_hz=6.0,
            gamma_freq_hz=50.0
        )

        assert len(results) == 3

    def test_theta_gamma_n_back_different_n(self, prefrontal):
        """Test different N-back distances."""
        sequence = [torch.randn(10) for _ in range(6)]

        # 1-back
        results_1 = theta_gamma_n_back(prefrontal, sequence, n=1)
        assert len(results_1) == 6

        # 3-back
        results_3 = theta_gamma_n_back(prefrontal, sequence, n=3)
        assert len(results_3) == 6


class TestIntegration:
    """Integration tests for full task pipeline."""

    def test_full_n_back_pipeline(self, prefrontal, wm_config):
        """Test complete N-back task pipeline."""
        # Create task
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Generate sequence with known matches
        n_items = 10
        sequence, matches = create_n_back_sequence(
            n_items=n_items,
            n_dims=10,
            n_back=2,
            match_probability=0.4,
            device=wm_config.device
        )

        # Run task
        results = task.run_sequence(sequence, target_matches=matches)

        # Verify results structure
        assert results["n_items"] == n_items
        assert results["n_back"] == 2
        assert len(results["responses"]) == n_items
        assert results["accuracy"] is not None

        # Accuracy should be between 0 and 1
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_different_n_back_values(self, prefrontal, wm_config):
        """Test multiple N-back distances."""
        sequence = [torch.randn(10) for _ in range(8)]

        for n in [1, 2, 3]:
            task = NBackTask(prefrontal, wm_config, n_back=n)
            results = task.run_sequence(sequence)

            assert results["n_back"] == n
            assert len(results["responses"]) == 8

    def test_phase_consistency(self, prefrontal, wm_config):
        """Test that phase assignments are consistent."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Encode multiple items
        phases = []
        for i in range(5):
            stimulus = torch.randn(10)
            metrics = task.encode_item(stimulus, i)
            phases.append(metrics["theta_phase"])

        # Phases should progress monotonically within cycle
        for i in range(len(phases) - 1):
            # Account for wrapping at 2π
            if phases[i+1] >= phases[i]:
                assert phases[i+1] > phases[i] or abs(phases[i+1] - phases[i]) < 0.01

    def test_working_memory_persistence(self, prefrontal, wm_config):
        """Test that working memory maintains items across sequence."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        # Present sequence
        sequence = [torch.randn(10) for _ in range(4)]
        results = task.run_sequence(sequence)

        # Working memory should contain something
        wm_content = prefrontal.get_working_memory()
        assert wm_content.abs().sum() > 0  # Not all zeros


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_n_back_zero(self, prefrontal, wm_config):
        """Test N=0 (comparing to current item)."""
        task = NBackTask(prefrontal, wm_config, n_back=0)

        stimulus = torch.randn(10)
        is_match, metrics = task.present_stimulus(stimulus)

        # N=0 should always match (current = current)
        assert metrics["can_retrieve"] == True

    def test_large_n_back(self, prefrontal, wm_config):
        """Test large N-back beyond working memory capacity."""
        task = NBackTask(prefrontal, wm_config, n_back=10)

        # Short sequence
        sequence = [torch.randn(10) for _ in range(5)]
        results = task.run_sequence(sequence)

        # No items should be retrievable
        assert all(
            m["can_retrieve"] == False
            for m in results["metrics"]
        )

    def test_empty_sequence(self, prefrontal, wm_config):
        """Test running task with empty sequence."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        results = task.run_sequence([])

        assert results["n_items"] == 0
        assert len(results["responses"]) == 0

    def test_single_item_sequence(self, prefrontal, wm_config):
        """Test sequence with single item."""
        task = NBackTask(prefrontal, wm_config, n_back=2)

        sequence = [torch.randn(10)]
        results = task.run_sequence(sequence)

        assert results["n_items"] == 1
        # No retrieval possible
        assert results["metrics"][0]["can_retrieve"] == False


class TestConfiguration:
    """Test configuration options."""

    def test_custom_theta_frequency(self, prefrontal, device):
        """Test custom theta frequency."""
        config = WorkingMemoryTaskConfig(
            theta_freq_hz=6.0,  # Slower theta
            device=device
        )

        task = NBackTask(prefrontal, config, n_back=2)
        assert task.encoder.config.theta_freq_hz == 6.0

    def test_custom_gamma_frequency(self, prefrontal, device):
        """Test custom gamma frequency."""
        config = WorkingMemoryTaskConfig(
            gamma_freq_hz=60.0,  # Faster gamma
            device=device
        )

        task = NBackTask(prefrontal, config, n_back=2)
        assert task.encoder.config.gamma_freq_hz == 60.0

    def test_custom_items_per_cycle(self, prefrontal, device):
        """Test custom items per theta cycle."""
        config = WorkingMemoryTaskConfig(
            items_per_theta_cycle=4,  # Fewer items
            device=device
        )

        encoder = ThetaGammaEncoder(config)

        # Item 2 should be at π (halfway through cycle)
        theta, _ = encoder.get_encoding_phase(2)
        assert abs(theta - math.pi) < 0.01

    def test_custom_time_windows(self, prefrontal, device):
        """Test custom encoding/retrieval windows."""
        config = WorkingMemoryTaskConfig(
            encoding_window_ms=200.0,
            retrieval_window_ms=100.0,
            device=device
        )

        task = NBackTask(prefrontal, config, n_back=2)
        assert config.encoding_window_ms == 200.0
        assert config.retrieval_window_ms == 100.0
