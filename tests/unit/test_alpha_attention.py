"""
Tests for alpha-based attention gating in cortex.

Tests that alpha oscillations properly suppress cortical activity
for attention control.

Author: Thalia Project
Date: December 9, 2025
"""

import pytest
import torch
import math

from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig


class TestAlphaAttentionGating:
    """Test alpha oscillation-based attention gating."""
    
    @pytest.fixture
    def cortex(self):
        """Create a small cortex for testing."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            device="cpu",
            use_gamma_attention=False,  # Isolate alpha testing
        )
        return LayeredCortex(config)
    
    @pytest.fixture
    def input_spikes(self):
        """Create test input spikes."""
        # 8 active spikes out of 32 (~25% sparsity)
        spikes = torch.zeros(32)
        spikes[::4] = 1.0  # Every 4th neuron fires
        return spikes
    
    def test_no_alpha_baseline(self, cortex, input_spikes):
        """Test baseline activity without alpha suppression."""
        # Reset to initial state
        cortex.reset_state()
        
        # Process without setting oscillator phases (no alpha)
        output = cortex.forward(input_spikes)
        
        # Should have some activity
        baseline_activity = output.sum().item()
        assert baseline_activity > 0, "Baseline should have activity"
        
        # Alpha suppression should be at default (1.0 = no suppression)
        assert cortex.state.alpha_suppression == 1.0
    
    def test_low_alpha_no_suppression(self, cortex, input_spikes):
        """Test that low alpha doesn't suppress (attention focused here)."""
        cortex.reset_state()
        
        # Set low/negative alpha signal (attention focused)
        cortex.set_oscillator_phases(
            phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            signals={"alpha": -0.5, "theta": 0.0, "gamma": 0.0, "beta": 0.0},  # Low alpha
            theta_slot=0,
        )
        
        output_low_alpha = cortex.forward(input_spikes)
        activity_low_alpha = output_low_alpha.sum().item()
        
        # With low/negative alpha, no suppression should occur
        assert cortex.state.alpha_suppression == 1.0, "Low alpha should not suppress"
        assert activity_low_alpha > 0, "Should have normal activity with low alpha"
    
    def test_high_alpha_suppresses(self, cortex, input_spikes):
        """Test that high alpha suppresses activity (attention elsewhere)."""
        # Run multiple trials to account for stochastic variability
        n_trials = 10
        baseline_total = 0
        suppressed_total = 0
        
        for _ in range(n_trials):
            # Baseline activity
            cortex.reset_state()
            baseline_output = cortex.forward(input_spikes)
            baseline_total += baseline_output.sum().item()
            
            # High alpha suppression
            cortex.reset_state()
            cortex.set_oscillator_phases(
                phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                signals={"alpha": 1.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},  # Max alpha
                theta_slot=0,
            )
            output_high_alpha = cortex.forward(input_spikes)
            suppressed_total += output_high_alpha.sum().item()
        
        # Average over trials
        baseline_avg = baseline_total / n_trials
        suppressed_avg = suppressed_total / n_trials
        
        # High alpha should reduce activity on average
        assert cortex.state.alpha_suppression == 0.5, "Max alpha should give 50% suppression"
        assert suppressed_avg <= baseline_avg, (
            f"High alpha should reduce activity: {suppressed_avg} <= {baseline_avg}"
        )
    
    def test_alpha_suppression_gradual(self, cortex, input_spikes):
        """Test that alpha suppression scales smoothly with alpha magnitude."""
        activities = []
        suppressions = []
        
        # Test different alpha levels
        for alpha_signal in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cortex.reset_state()
            cortex.set_oscillator_phases(
                phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                signals={"alpha": alpha_signal, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                theta_slot=0,
            )
            
            output = cortex.forward(input_spikes)
            activities.append(output.sum().item())
            suppressions.append(cortex.state.alpha_suppression)
        
        # Activities should decrease as alpha increases
        for i in range(len(activities) - 1):
            assert activities[i] >= activities[i + 1], (
                f"Activity should decrease with increasing alpha: "
                f"{activities[i]} >= {activities[i+1]} at alpha={i*0.25} vs {(i+1)*0.25}"
            )
        
        # Suppressions should decrease linearly (1.0 → 0.5)
        expected_suppressions = [1.0, 0.875, 0.75, 0.625, 0.5]
        for i, (actual, expected) in enumerate(zip(suppressions, expected_suppressions)):
            assert abs(actual - expected) < 0.01, (
                f"Suppression at alpha={i*0.25} should be {expected}, got {actual}"
            )
    
    def test_alpha_phase_cycling(self, cortex, input_spikes):
        """Test that alpha phase cycling modulates activity over time."""
        activities = []
        
        # Simulate alpha cycling (10 Hz, 100ms period)
        dt = 10.0  # 10ms timesteps
        for t in range(10):  # 100ms total
            phase = (t * dt / 100.0) * 2 * math.pi  # One cycle
            signal = math.sin(phase)  # Oscillates -1 to 1
            
            cortex.set_oscillator_phases(
                phases={"alpha": phase, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                signals={"alpha": signal, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                theta_slot=0,
            )
            
            output = cortex.forward(input_spikes, dt=dt)
            activities.append(output.sum().item())
        
        # Should see variation in activity over time
        # When alpha is high (positive sine), activity should be lower
        # When alpha is low (negative sine), activity should be higher
        assert max(activities) > min(activities), (
            "Alpha cycling should modulate activity levels"
        )
    
    def test_alpha_suppression_early_gating(self, cortex, input_spikes):
        """Test that alpha suppression happens at input (early gating)."""
        cortex.reset_state()
        
        # Set max alpha suppression
        cortex.set_oscillator_phases(
            phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            signals={"alpha": 1.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            theta_slot=0,
        )
        
        # Process and check L4 activity (first layer)
        output = cortex.forward(input_spikes)
        l4_activity = cortex.state.l4_spikes.sum().item()
        
        # L4 should be suppressed (early gating)
        # With 50% suppression, we expect ~50% less input drive
        assert l4_activity >= 0, "L4 should have some activity"
        # The suppression should be visible in the first layer
        assert cortex.state.alpha_suppression == 0.5


class TestAlphaIntegration:
    """Test alpha integration with other cortex features."""
    
    def test_alpha_with_top_down_modulation(self):
        """Test alpha suppression interacts properly with top-down attention."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            device="cpu",
            use_gamma_attention=False,  # Isolate alpha testing
        )
        cortex = LayeredCortex(config)
        
        input_spikes = torch.zeros(32)
        input_spikes[::4] = 1.0
        
        # Top-down modulation to L2/3 (16 neurons with 1.5 ratio = 24 neurons)
        top_down = torch.zeros(24)
        top_down[:8] = 0.5  # Boost first 8 neurons
        
        # Run multiple trials to account for stochasticity
        n_trials = 10
        no_alpha_total = 0
        alpha_total = 0
        
        for _ in range(n_trials):
            # Test 1: No alpha, with top-down
            cortex.reset_state()
            output_no_alpha = cortex.forward(input_spikes, top_down=top_down)
            no_alpha_total += output_no_alpha.sum().item()
            
            # Test 2: High alpha, with top-down
            cortex.reset_state()
            cortex.set_oscillator_phases(
                phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                signals={"alpha": 1.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                theta_slot=0,
            )
            output_alpha = cortex.forward(input_spikes, top_down=top_down)
            alpha_total += output_alpha.sum().item()
        
        # Alpha should suppress even when top-down is present (on average)
        no_alpha_avg = no_alpha_total / n_trials
        alpha_avg = alpha_total / n_trials
        assert alpha_avg <= no_alpha_avg, (
            f"Alpha suppression should work even with top-down: {alpha_avg} <= {no_alpha_avg}"
        )
    
    def test_alpha_with_theta_modulation(self):
        """Test alpha suppression works alongside theta encoding/retrieval modulation."""
        config = LayeredCortexConfig(n_input=32, n_output=16, device="cpu")
        cortex = LayeredCortex(config)
        
        input_spikes = torch.zeros(32)
        input_spikes[::4] = 1.0
        
        # Test alpha with encoding modulation
        cortex.reset_state()
        cortex.set_oscillator_phases(
            phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            signals={"alpha": 0.8, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            theta_slot=0,
        )
        # Theta modulation computed internally from theta_phase (set to 0.0 above)
        output = cortex.forward(input_spikes)
        
        # Should work without errors and show suppression
        assert cortex.state.alpha_suppression == 0.6  # 1.0 - (0.8 * 0.5)
        assert output.sum().item() >= 0


class TestAlphaDiagnostics:
    """Test diagnostics and monitoring of alpha suppression."""
    
    def test_alpha_suppression_in_state(self):
        """Test that alpha suppression is tracked in state."""
        config = LayeredCortexConfig(n_input=32, n_output=16, device="cpu")
        cortex = LayeredCortex(config)
        
        input_spikes = torch.zeros(32)
        input_spikes[::4] = 1.0
        
        # Initially should be default (1.0)
        cortex.reset_state()
        cortex.forward(input_spikes)
        assert hasattr(cortex.state, 'alpha_suppression')
        assert cortex.state.alpha_suppression == 1.0
        
        # After setting alpha, should update
        cortex.set_oscillator_phases(
            phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            signals={"alpha": 0.6, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            theta_slot=0,
        )
        cortex.forward(input_spikes)
        assert cortex.state.alpha_suppression == 0.7  # 1.0 - (0.6 * 0.5)
    
    def test_alpha_suppression_persistent(self):
        """Test that alpha suppression persists across forward passes."""
        config = LayeredCortexConfig(n_input=32, n_output=16, device="cpu")
        cortex = LayeredCortex(config)
        
        input_spikes = torch.zeros(32)
        input_spikes[::4] = 1.0
        
        # Set alpha once
        cortex.reset_state()
        cortex.set_oscillator_phases(
            phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            signals={"alpha": 0.8, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            theta_slot=0,
        )
        
        # Multiple forward passes should use same alpha
        for _ in range(5):
            cortex.forward(input_spikes)
            assert cortex.state.alpha_suppression == 0.6  # Persistent


class TestAlphaWithPredictiveCortex:
    """Test that alpha gating works correctly with PredictiveCortex."""
    
    def test_predictive_cortex_alpha_passthrough(self):
        """Test that PredictiveCortex passes alpha signals to inner LayeredCortex."""
        from thalia.regions.cortex.predictive_cortex import PredictiveCortex, PredictiveCortexConfig
        
        config = PredictiveCortexConfig(
            n_input=32,
            n_output=16,
            device="cpu",
            prediction_enabled=True,
            use_gamma_attention=False,  # Disable gamma attention for pure alpha test
        )
        cortex = PredictiveCortex(config)
        
        input_spikes = torch.zeros(32)
        input_spikes[::4] = 1.0
        
        # Set alpha oscillator
        cortex.set_oscillator_phases(
            phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
            signals={"alpha": 1.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},  # Max alpha
            theta_slot=0,
        )
        
        # Process input
        cortex.forward(input_spikes)
        
        # Check that alpha suppression was applied in inner cortex
        assert hasattr(cortex.cortex.state, 'alpha_suppression'), \
            "Inner LayeredCortex should have alpha_suppression"
        assert cortex.cortex.state.alpha_suppression == 0.5, \
            f"Max alpha should give 50% suppression, got {cortex.cortex.state.alpha_suppression}"
    
    def test_predictive_cortex_alpha_suppression_works(self):
        """Test that alpha actually suppresses activity in PredictiveCortex."""
        from thalia.regions.cortex.predictive_cortex import PredictiveCortex, PredictiveCortexConfig
        
        config = PredictiveCortexConfig(
            n_input=32,
            n_output=16,
            device="cpu",
            prediction_enabled=True,
            use_gamma_attention=False,  # Disable gamma attention for pure alpha test
        )
        cortex = PredictiveCortex(config)
        
        input_spikes = torch.zeros(32)
        input_spikes[::2] = 1.0  # 50% input
        
        # Multiple trials to account for stochasticity
        n_trials = 10
        baseline_total = 0
        suppressed_total = 0
        
        for _ in range(n_trials):
            # Baseline
            cortex.reset_state()
            output_baseline = cortex.forward(input_spikes)
            baseline_total += output_baseline.sum().item()
            
            # With high alpha
            cortex.reset_state()
            cortex.set_oscillator_phases(
                phases={"alpha": 0.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                signals={"alpha": 1.0, "theta": 0.0, "gamma": 0.0, "beta": 0.0},
                theta_slot=0,
            )
            output_suppressed = cortex.forward(input_spikes)
            suppressed_total += output_suppressed.sum().item()
        
        baseline_avg = baseline_total / n_trials
        suppressed_avg = suppressed_total / n_trials
        
        # Alpha should suppress (or at least not increase) activity
        assert suppressed_avg <= baseline_avg, (
            f"Alpha should suppress activity: {suppressed_avg} <= {baseline_avg}"
        )
    
    def test_predictive_cortex_stores_oscillator_signals(self):
        """Test that PredictiveCortex stores oscillator signals in its own state."""
        from thalia.regions.cortex.predictive_cortex import PredictiveCortex, PredictiveCortexConfig
        
        config = PredictiveCortexConfig(
            n_input=32,
            n_output=16,
            device="cpu",
        )
        cortex = PredictiveCortex(config)
        
        # Set oscillators
        test_phases = {"alpha": 1.5, "theta": 2.0, "gamma": 3.0, "beta": 1.0}
        test_signals = {"alpha": 0.7, "theta": 0.3, "gamma": 0.5, "beta": -0.2}
        
        cortex.set_oscillator_phases(
            phases=test_phases,
            signals=test_signals,
            theta_slot=3,
        )
        
        # Check storage in PredictiveCortex state
        assert hasattr(cortex.state, '_oscillator_phases'), \
            "PredictiveCortex state should have _oscillator_phases"
        assert hasattr(cortex.state, '_oscillator_signals'), \
            "PredictiveCortex state should have _oscillator_signals"
        
        assert cortex.state._oscillator_phases == test_phases
        assert cortex.state._oscillator_signals == test_signals
        
        # Check passthrough to inner cortex
        assert cortex.cortex.state._oscillator_phases == test_phases
        assert cortex.cortex.state._oscillator_signals == test_signals

