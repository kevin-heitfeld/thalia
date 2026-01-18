"""
Tests for DynamicBrain OscillatorManager integration (Phase 1.7.2).

Tests that DynamicBrain correctly integrates OscillatorManager for:
- Oscillator initialization and advancement
- Phase broadcasting to components
- Oscillator state save/load
- Oscillator diagnostics

Author: Thalia Project
Date: December 15, 2025
"""

import math

import pytest
import torch

from thalia.config import GlobalConfig, LayerSizeCalculator
from thalia.core.brain_builder import BrainBuilder


class TestOscillatorManagerIntegration:
    """Test OscillatorManager integration in DynamicBrain."""

    @pytest.fixture
    def global_config(self):
        """Create test global config with theta frequency."""
        return GlobalConfig(device="cpu", dt_ms=1.0, theta_frequency_hz=8.0)

    @pytest.fixture
    def simple_brain(self, global_config):
        """Create simple brain for testing."""
        brain = (
            BrainBuilder(global_config)
            .add_component("input", "thalamic_relay", input_size=64, relay_size=64, trn_size=0)
            .add_component(
                "cortex",
                "layered_cortex",
                **LayerSizeCalculator().cortex_from_output(32),
            )
            .connect("input", "cortex", pathway_type="axonal_projection")
            .build()
        )
        return brain

    def test_oscillator_manager_exists(self, simple_brain):
        """Test that OscillatorManager is initialized."""
        assert hasattr(simple_brain, "oscillators")
        assert simple_brain.oscillators is not None

    def test_oscillators_initialized(self, simple_brain):
        """Test that all 6 oscillators are initialized."""
        assert hasattr(simple_brain.oscillators, "delta")
        assert hasattr(simple_brain.oscillators, "theta")
        assert hasattr(simple_brain.oscillators, "alpha")
        assert hasattr(simple_brain.oscillators, "beta")
        assert hasattr(simple_brain.oscillators, "gamma")
        assert hasattr(simple_brain.oscillators, "ripple")

        # Check initial phases for all oscillators
        for osc_name in ["delta", "theta", "alpha", "beta", "gamma", "ripple"]:
            osc = getattr(simple_brain.oscillators, osc_name)
            assert osc.phase >= 0.0

    def test_oscillators_advance_during_forward(self, simple_brain):
        """Test that oscillators advance during forward pass."""
        # Get initial phases
        initial_theta_phase = simple_brain.oscillators.theta.phase
        initial_alpha_phase = simple_brain.oscillators.alpha.phase

        # Run forward pass
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=10)

        # Phases should have advanced
        assert simple_brain.oscillators.theta.phase != initial_theta_phase
        assert simple_brain.oscillators.alpha.phase != initial_alpha_phase

        # Note: Gamma oscillator is disabled by default (amplitude=0) as it should
        # emerge from the L6→TRN feedback loop. If gamma is explicitly enabled,
        # its phase would advance, but we don't test that here.

    def test_oscillator_phases_wrap(self, simple_brain):
        """Test that oscillator phases wrap around 2π."""
        # Run enough timesteps for phase to wrap
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=500)

        # All phases should be in [0, 2π)
        for osc_name in ["delta", "theta", "alpha", "beta", "gamma", "ripple"]:
            osc = getattr(simple_brain.oscillators, osc_name)
            assert 0.0 <= osc.phase < 2.0 * math.pi

    def test_oscillator_diagnostics(self, simple_brain):
        """Test that oscillator diagnostics are collected."""
        # Run forward pass
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=5)

        # Get diagnostics
        diag = simple_brain.get_diagnostics()

        # Should include oscillators
        assert "oscillators" in diag
        osc_diag = diag["oscillators"]

        # Should have all oscillators
        assert "delta" in osc_diag
        assert "theta" in osc_diag
        assert "alpha" in osc_diag
        assert "beta" in osc_diag
        assert "gamma" in osc_diag
        assert "ripple" in osc_diag

        # Each should have phase and frequency
        for osc_name in ["delta", "theta", "alpha", "beta", "gamma", "ripple"]:
            assert "phase" in osc_diag[osc_name]
            assert "frequency_hz" in osc_diag[osc_name]
            assert isinstance(osc_diag[osc_name]["phase"], float)
            assert isinstance(osc_diag[osc_name]["frequency_hz"], float)

    def test_oscillator_state_save_load(self, simple_brain):
        """Test that oscillator states are saved and loaded correctly."""
        # Run forward to set some state
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=50)

        # Save state
        state = simple_brain.get_full_state()

        # Verify oscillator state is saved
        assert "oscillators" in state
        assert "delta" in state["oscillators"]
        assert "theta" in state["oscillators"]
        assert "alpha" in state["oscillators"]
        assert "beta" in state["oscillators"]
        assert "gamma" in state["oscillators"]
        assert "ripple" in state["oscillators"]

        # Save current phases
        saved_delta_phase = simple_brain.oscillators.delta.phase
        saved_theta_phase = simple_brain.oscillators.theta.phase
        saved_alpha_phase = simple_brain.oscillators.alpha.phase

        # Run more to change phases
        simple_brain.forward(input_data, n_timesteps=50)

        # Phases should have changed
        assert simple_brain.oscillators.theta.phase != saved_theta_phase

        # Load state
        simple_brain.load_full_state(state)

        # Phases should be restored
        assert abs(simple_brain.oscillators.delta.phase - saved_delta_phase) < 1e-6
        assert abs(simple_brain.oscillators.theta.phase - saved_theta_phase) < 1e-6
        assert abs(simple_brain.oscillators.alpha.phase - saved_alpha_phase) < 1e-6

    def test_oscillator_reset(self, simple_brain):
        """Test that oscillators reset correctly."""
        # Run forward to change state
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=50)

        # Phases should have advanced
        assert simple_brain.oscillators.theta.phase > 0.0

        # Reset
        simple_brain.reset_state()

        # Phases should be reset (may not be exactly 0 if initial_phase != 0)
        # but should be back to initial values
        theta_phase = simple_brain.oscillators.theta.phase
        assert 0.0 <= theta_phase < 0.5  # Should be near 0

    def test_theta_frequency_from_config(self, global_config):
        """Test that theta frequency comes from config."""
        # Config has theta_frequency_hz=8.0
        brain = (
            BrainBuilder(global_config)
            .add_component("input", "thalamic_relay", input_size=64, relay_size=64, trn_size=0)
            .build()
        )

        # Theta should have frequency from config
        assert brain.oscillators.theta.frequency_hz == 8.0

    def test_broadcast_oscillator_phases_called(self, simple_brain, monkeypatch):
        """Test that _broadcast_oscillator_phases is called during forward."""
        broadcast_called = {"count": 0}

        original_broadcast = simple_brain._broadcast_oscillator_phases

        def tracked_broadcast():
            broadcast_called["count"] += 1
            return original_broadcast()

        monkeypatch.setattr(simple_brain, "_broadcast_oscillator_phases", tracked_broadcast)

        # Run forward pass
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=10)

        # Should have been called at least once during execution
        # (Event-driven mode calls it differently than synchronous mode)
        assert broadcast_called["count"] >= 1

    def test_oscillators_in_event_driven_mode(self, simple_brain):
        """Test that oscillators work in event-driven execution mode."""
        # Get initial phase
        initial_phase = simple_brain.oscillators.theta.phase

        # Run forward pass
        input_data = {"input": torch.randn(64)}
        simple_brain.forward(input_data, n_timesteps=10)

        # Phase should have advanced
        assert simple_brain.oscillators.theta.phase != initial_phase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
