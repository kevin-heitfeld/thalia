"""Tests for Hebbian learning rules and phase homeostasis."""

import torch

from thalia.learning.hebbian import (
    hebbian_update,
    synaptic_scaling,
    PredictiveCoding,
)
from thalia.learning.phase_homeostasis import (
    PhaseHomeostasis,
    update_bcm_threshold,
    update_homeostatic_excitability,
)


class TestHebbianUpdate:
    """Tests for hebbian_update function."""

    def test_basic_update(self):
        """Test that coincident spikes increase weight."""
        weights = torch.ones(5, 10) * 0.5
        input_spikes = torch.zeros(1, 10)
        output_spikes = torch.zeros(1, 5)

        # Coincident spikes on neuron 0 → 0
        input_spikes[0, 0] = 1.0
        output_spikes[0, 0] = 1.0

        dw = hebbian_update(weights, input_spikes, output_spikes,
                            learning_rate=0.01, w_max=1.3,
                            heterosynaptic_ratio=0.5)
        new_weights = (weights + dw).clamp(0.0, 1.3)

        # Weight 0→0 should increase
        assert new_weights[0, 0] > weights[0, 0]
        # Other weights should be unchanged (dw=0 for non-active output)
        assert new_weights[1, 0] == weights[1, 0]

    def test_soft_bound(self):
        """Test that updates respect soft bound at w_max."""
        weights = torch.ones(1, 1) * 0.99
        input_spikes = torch.ones(1, 1)
        output_spikes = torch.ones(1, 1)

        dw = hebbian_update(weights, input_spikes, output_spikes,
                            learning_rate=0.1, w_max=1.0,
                            heterosynaptic_ratio=0.5)
        new_weights = (weights + dw).clamp(0.0, 1.0)

        # Should not exceed w_max
        assert new_weights[0, 0] <= 1.0
        # But should still increase somewhat
        assert new_weights[0, 0] > weights[0, 0]

    def test_no_spikes_no_change(self):
        """Test that no spikes means no weight change."""
        weights = torch.ones(3, 5) * 0.5
        input_spikes = torch.zeros(1, 5)
        output_spikes = torch.zeros(1, 3)

        dw = hebbian_update(weights, input_spikes, output_spikes,
                            learning_rate=0.01, w_max=1.3,
                            heterosynaptic_ratio=0.5)
        new_weights = (weights + dw).clamp(0.0, 1.3)

        assert torch.allclose(new_weights, weights)

    def test_stp_gating_reduces_learning(self):
        """Test that low STP resources reduce learning rate."""
        weights = torch.ones(5, 10) * 0.5
        input_spikes = torch.zeros(1, 10)
        output_spikes = torch.zeros(1, 5)

        # Coincident spikes
        input_spikes[0, 0] = 1.0
        output_spikes[0, 0] = 1.0

        # Without STP gating
        dw_no_stp = hebbian_update(
            weights.clone(), input_spikes, output_spikes,
            learning_rate=0.1, w_max=1.3,
            heterosynaptic_ratio=0.5)
        new_weights_no_stp = (weights.clone() + dw_no_stp).clamp(0.0, 1.3)

        # With full STP resources (should be same as no gating)
        full_resources = torch.ones(5, 10)
        dw_full_stp = hebbian_update(
            weights.clone(), input_spikes, output_spikes,
            learning_rate=0.1, w_max=1.3,
            heterosynaptic_ratio=0.5,
            stp_resources=full_resources)
        new_weights_full_stp = (weights.clone() + dw_full_stp).clamp(0.0, 1.3)

        # With depleted STP resources (should learn less)
        depleted_resources = torch.ones(5, 10) * 0.2
        dw_depleted = hebbian_update(
            weights.clone(), input_spikes, output_spikes,
            learning_rate=0.1, w_max=1.3,
            heterosynaptic_ratio=0.5,
            stp_resources=depleted_resources)
        new_weights_depleted = (weights.clone() + dw_depleted).clamp(0.0, 1.3)

        # Full resources should produce same update as no STP
        assert torch.allclose(new_weights_full_stp, new_weights_no_stp)

        # Depleted resources should produce smaller update
        delta_full = new_weights_full_stp[0, 0] - weights[0, 0]
        delta_depleted = new_weights_depleted[0, 0] - weights[0, 0]
        assert delta_depleted < delta_full
        assert delta_depleted > 0  # Still some learning


class TestSynapticScaling:
    """Tests for synaptic_scaling function."""

    def test_scaling_direction(self):
        """Test that scaling moves weights toward target norm."""
        # Weights with high norm - use small w_max to see proper scaling down
        weights_high = torch.ones(5, 10) * 0.8
        target_norm = 0.3
        w_max = 1.0

        scaled_high = synaptic_scaling(weights_high, target_norm, tau=1.0, w_max=w_max)

        # Norm should decrease
        assert scaled_high.norm() < weights_high.norm()

        # Weights with low norm
        weights_low = torch.ones(5, 10) * 0.1
        scaled_low = synaptic_scaling(weights_low, target_norm, tau=1.0, w_max=w_max)

        # Norm should increase
        assert scaled_low.norm() > weights_low.norm()

    def test_slow_time_constant(self):
        """Test that large tau means slow changes."""
        weights = torch.ones(5, 10) * 2.0
        target_norm = 0.5  # Different from weight norm
        w_max = 3.0

        scaled_fast = synaptic_scaling(weights, target_norm, tau=1.0, w_max=w_max)
        scaled_slow = synaptic_scaling(weights, target_norm, tau=100.0, w_max=w_max)

        # Fast scaling should change more
        fast_change = (weights - scaled_fast).abs().sum()
        slow_change = (weights - scaled_slow).abs().sum()

        assert fast_change > slow_change


class TestPredictiveCoding:
    """Tests for PredictiveCoding class."""

    def test_initialization(self):
        """Test PredictiveCoding initializes correctly."""
        pc = PredictiveCoding(n_output=10, gamma_period=25)

        assert pc.n_output == 10
        assert pc.gamma_period == 25
        assert pc.last_gamma_winner == -1

    def test_accumulate_spikes(self):
        """Test spike accumulation."""
        pc = PredictiveCoding(n_output=5)

        spikes = torch.zeros(1, 5)
        spikes[0, 2] = 1.0

        pc.accumulate_spikes(spikes)

        assert pc.gamma_spike_counts[2] == 1.0
        assert pc.gamma_spike_counts.sum() == 1.0

    def test_reset(self):
        """Test reset clears state."""
        pc = PredictiveCoding(n_output=5)

        pc.accumulate_spikes(torch.ones(1, 5))
        assert pc.gamma_spike_counts.sum() > 0

        pc.reset()
        assert pc.gamma_spike_counts.sum() == 0


class TestPhaseHomeostasis:
    """Tests for PhaseHomeostasis class."""

    def test_initialization(self):
        """Test PhaseHomeostasis initializes correctly."""
        ph = PhaseHomeostasis(n_output=10)

        assert ph.n_output == 10
        stats = ph.get_stats()
        assert stats["mean"] == 0.0

    def test_record_win(self):
        """Test that recording wins updates counts."""
        ph = PhaseHomeostasis(n_output=5)

        # Record wins for neuron 0 and 2
        ph.record_win(0)
        ph.record_win(0)
        ph.record_win(2)

        # After update_cycle, we see the averages
        ph.update_cycle()
        stats = ph.get_stats()
        wins = stats["wins"]

        assert wins[0] > 0
        assert wins[2] > 0

    def test_reset_cycle(self):
        """Test that reset_cycle clears per-cycle counts."""
        ph = PhaseHomeostasis(n_output=5)

        ph.record_win(0)
        ph.reset_cycle()

        # Win count should be reset (but avg is still 0 before update)
        ph.record_win(1)
        ph.update_cycle()

        stats = ph.get_stats()
        wins = stats["wins"]
        # Neuron 1 should have the win, not neuron 0
        assert wins[1] > 0


class TestBCMThresholdUpdate:
    """Tests for update_bcm_threshold function."""

    def test_threshold_increases_with_high_activity(self):
        """Test that high activity increases BCM threshold."""
        threshold = torch.ones(1, 5) * 0.5
        target_rate = 10.0  # Hz
        high_activity = 50.0  # Much higher than target

        new_threshold = update_bcm_threshold(
            threshold, high_activity, target_rate,
            tau=200.0, min_threshold=0.01, max_threshold=2.0
        )

        assert (new_threshold >= threshold).all()

    def test_threshold_decreases_with_low_activity(self):
        """Test that low activity decreases BCM threshold."""
        threshold = torch.ones(1, 5) * 0.5
        target_rate = 10.0  # Hz
        low_activity = 1.0  # Much lower than target

        new_threshold = update_bcm_threshold(
            threshold, low_activity, target_rate,
            tau=200.0, min_threshold=0.01, max_threshold=2.0
        )

        assert (new_threshold <= threshold).all()

    def test_threshold_clamped(self):
        """Test that threshold is clamped to valid range."""
        threshold = torch.ones(1, 5) * 0.5

        # Extreme high activity
        new_threshold = update_bcm_threshold(
            threshold, 1000.0, 10.0,
            tau=200.0, min_threshold=0.01, max_threshold=2.0
        )
        assert (new_threshold <= 2.0).all()

        # Extreme low activity
        new_threshold = update_bcm_threshold(
            threshold, 0.0, 10.0,
            tau=200.0, min_threshold=0.01, max_threshold=2.0
        )
        assert (new_threshold >= 0.01).all()


class TestHomeostaticExcitability:
    """Tests for update_homeostatic_excitability function."""

    def test_excitability_increases_when_firing_low(self):
        """Test that low firing rate increases excitability."""
        current_rate = torch.zeros(1, 5)  # Not firing
        avg_rate = torch.zeros(1, 5)
        excitability = torch.zeros(1, 5)
        target_rate = 0.01  # Target is 10 spikes/1000 timesteps

        new_avg, new_excit = update_homeostatic_excitability(
            current_rate, avg_rate, excitability,
            target_rate=target_rate,
            tau=10.0,
            strength=0.1,
            v_threshold=1.0,
            bounds=(-3.0, 3.0),
        )

        # Excitability should increase (positive adjustment)
        assert (new_excit > excitability).all()

    def test_excitability_decreases_when_firing_high(self):
        """Test that high firing rate decreases excitability."""
        current_rate = torch.ones(1, 5) * 0.1  # Firing a lot
        avg_rate = torch.ones(1, 5) * 0.1
        excitability = torch.zeros(1, 5)
        target_rate = 0.001  # Target is much lower

        new_avg, new_excit = update_homeostatic_excitability(
            current_rate, avg_rate, excitability,
            target_rate=target_rate,
            tau=10.0,
            strength=0.1,
            v_threshold=1.0,
            bounds=(-3.0, 3.0),
        )

        # Excitability should decrease (negative adjustment)
        assert (new_excit < excitability).all()

    def test_excitability_clamped_to_bounds(self):
        """Test that excitability is clamped to specified bounds."""
        current_rate = torch.zeros(1, 5)
        avg_rate = torch.zeros(1, 5)
        excitability = torch.ones(1, 5) * 2.9  # Near upper bound

        new_avg, new_excit = update_homeostatic_excitability(
            current_rate, avg_rate, excitability,
            target_rate=0.1,  # High target → want to increase excitability
            tau=1.0,  # Fast update
            strength=1.0,  # Strong update
            v_threshold=1.0,
            bounds=(-3.0, 3.0),
        )

        # Should be clamped at 3.0
        assert (new_excit <= 3.0).all()

        # Test lower bound
        excitability = torch.ones(1, 5) * -2.9
        new_avg, new_excit = update_homeostatic_excitability(
            current_rate=torch.ones(1, 5) * 0.5,  # High rate
            avg_firing_rate=torch.ones(1, 5) * 0.5,
            excitability=excitability,
            target_rate=0.001,  # Low target → want to decrease
            tau=1.0,
            strength=1.0,
            v_threshold=1.0,
            bounds=(-3.0, 3.0),
        )

        assert (new_excit >= -3.0).all()

    def test_avg_rate_exponential_moving_average(self):
        """Test that avg_rate follows exponential moving average."""
        current_rate = torch.ones(1, 5) * 0.1
        avg_rate = torch.zeros(1, 5)
        excitability = torch.zeros(1, 5)
        tau = 10.0

        new_avg, _ = update_homeostatic_excitability(
            current_rate, avg_rate, excitability,
            target_rate=0.05,
            tau=tau,
            strength=0.01,
            v_threshold=1.0,
            bounds=(-3.0, 3.0),
        )

        # Expected: avg + (current - avg) / tau = 0 + (0.1 - 0) / 10 = 0.01
        expected = avg_rate + (current_rate - avg_rate) / tau
        assert torch.allclose(new_avg, expected)

    def test_pure_function_no_mutation(self):
        """Test that the function doesn't mutate input tensors."""
        current_rate = torch.ones(1, 5) * 0.05
        avg_rate = torch.ones(1, 5) * 0.03
        excitability = torch.ones(1, 5) * 0.5

        # Clone for comparison
        avg_rate_orig = avg_rate.clone()
        excitability_orig = excitability.clone()

        new_avg, new_excit = update_homeostatic_excitability(
            current_rate, avg_rate, excitability,
            target_rate=0.05,
            tau=10.0,
            strength=0.01,
            v_threshold=1.0,
            bounds=(-3.0, 3.0),
        )

        # Original tensors should be unchanged
        assert torch.equal(avg_rate, avg_rate_orig)
        assert torch.equal(excitability, excitability_orig)
        # Returned tensors should be different objects
        assert new_avg is not avg_rate
        assert new_excit is not excitability
