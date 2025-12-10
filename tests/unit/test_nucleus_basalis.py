"""
Unit tests for NucleusBasalisSystem (centralized acetylcholine/attention management).

Tests cover:
- ACh computation from prediction error
- Encoding vs retrieval mode determination
- Phasic ACh bursts for novelty
- Tonic + phasic dynamics
- Exponential decay (fastest of the three)
- State management (get/set/reset)
- Health checks
"""

import pytest
import torch

from thalia.core.nucleus_basalis import NucleusBasalisSystem, NucleusBasalisConfig


class TestNucleusBasalisSystem:
    """Test suite for Nucleus Basalis acetylcholine system."""

    @pytest.fixture
    def nb(self):
        """Create NB system with default config."""
        config = NucleusBasalisConfig(
            baseline_ach=0.2,
            ach_decay_per_ms=0.98,  # τ=50ms (fastest decay)
            encoding_threshold=0.5,
            novelty_gain=2.0,
        )
        return NucleusBasalisSystem(config)

    def test_initialization(self, nb):
        """Test NB initializes with correct baseline."""
        assert nb.config.baseline_ach == 0.2
        assert nb._global_ach == 0.2
        assert nb._tonic_ach == 0.2
        assert nb._phasic_ach == 0.0

    def test_ach_from_prediction_error(self, nb):
        """Test ACh increases with prediction error (novelty)."""
        nb.reset_state()

        # Low PE → low ACh (retrieval mode)
        for _ in range(10):
            nb.update(dt_ms=1.0, prediction_error=0.0)
        low_ach = nb.get_acetylcholine()

        # High PE → high ACh (encoding mode)
        nb.reset_state()
        for _ in range(10):
            nb.update(dt_ms=1.0, prediction_error=0.8)
        high_ach = nb.get_acetylcholine()

        assert high_ach > low_ach

    def test_encoding_mode_detection(self, nb):
        """Test is_encoding_mode switches at threshold."""
        nb.reset_state()

        # Low ACh → retrieval mode
        nb.update(dt_ms=1.0, prediction_error=0.0)
        assert not nb.is_encoding_mode()

        # High ACh → encoding mode
        nb.reset_state()
        for _ in range(10):
            nb.update(dt_ms=1.0, prediction_error=0.9)
        assert nb.is_encoding_mode()

    def test_encoding_strength(self, nb):
        """Test get_encoding_strength returns correct range."""
        nb.reset_state()

        # ACh below threshold → encoding strength near 0
        nb._global_ach = 0.2
        low_strength = nb.get_encoding_strength()
        assert 0.0 <= low_strength < 0.5

        # ACh above threshold → encoding strength near 1
        nb._global_ach = 0.9
        high_strength = nb.get_encoding_strength()
        assert 0.5 < high_strength <= 1.0

    def test_phasic_burst_from_novelty(self, nb):
        """Test phasic ACh burst when novelty detected."""
        nb.reset_state()
        initial_ach = nb.get_acetylcholine()

        # Trigger attention (high prediction error)
        nb.trigger_attention(attention_strength=1.0)
        burst_ach = nb.get_acetylcholine()

        assert burst_ach > initial_ach
        assert nb._phasic_ach > 0.0

    def test_fastest_decay_of_three_systems(self, nb):
        """Test ACh decays fastest (τ=50ms vs 100ms for NE, 200ms for DA)."""
        nb.reset_state()

        # Create phasic burst
        nb.trigger_attention(attention_strength=1.0)
        peak_ach = nb.get_acetylcholine()

        # Decay over 100ms (2x tau)
        for _ in range(100):
            nb.update(dt_ms=1.0, prediction_error=0.0)

        decayed_ach = nb.get_acetylcholine()

        # Should decay significantly (τ=50ms means ~86% decay in 100ms)
        assert decayed_ach < peak_ach * 0.2

    def test_baseline_adaptation(self, nb):
        """Test baseline ACh adapts to task demands."""
        nb.reset_state()

        # Sustained high prediction error
        for _ in range(100):
            nb.update(dt_ms=1.0, prediction_error=0.6)

        # Tonic should be elevated above initial baseline
        assert nb._tonic_ach > nb.config.baseline_ach

    def test_state_persistence(self, nb):
        """Test get_state/set_state for checkpointing."""
        # Set some state
        nb.update(dt_ms=1.0, prediction_error=0.5)
        nb.trigger_attention(attention_strength=0.8)

        # Save state
        state = nb.get_state()

        # Create new NB and restore
        nb2 = NucleusBasalisSystem(nb.config)
        nb2.set_state(state)

        # Should match
        assert nb2._global_ach == pytest.approx(nb._global_ach)
        assert nb2._tonic_ach == pytest.approx(nb._tonic_ach)
        assert nb2._phasic_ach == pytest.approx(nb._phasic_ach)

    def test_reset_state(self, nb):
        """Test reset returns to baseline."""
        # Modify state
        nb.update(dt_ms=1.0, prediction_error=0.8)
        nb.trigger_attention(attention_strength=1.0)

        assert nb._global_ach != nb.config.baseline_ach

        # Reset
        nb.reset_state()

        assert nb._global_ach == nb.config.baseline_ach
        assert nb._tonic_ach == nb.config.baseline_ach
        assert nb._phasic_ach == 0.0

    def test_health_check_healthy(self, nb):
        """Test health check passes for normal operation."""
        nb.update(dt_ms=1.0, prediction_error=0.3)
        health = nb.check_health()

        assert health["is_healthy"] is True
        assert len(health["issues"]) == 0

    def test_health_check_detects_runaway(self, nb):
        """Test health check detects runaway ACh."""
        # Force runaway ACh
        nb._global_ach = 5.0

        health = nb.check_health()

        assert health["is_healthy"] is False
        assert any("acetylcholine too high" in issue.lower() for issue in health["issues"])

    def test_health_check_detects_negative(self, nb):
        """Test health check detects negative ACh."""
        # Force negative ACh
        nb._global_ach = -0.3

        health = nb.check_health()

        assert health["is_healthy"] is False
        assert any("negative" in issue.lower() for issue in health["issues"])

    def test_novelty_gain_parameter(self):
        """Test novelty_gain affects ACh response to PE."""
        # Low gain
        low_config = NucleusBasalisConfig(novelty_gain=0.5)
        low_nb = NucleusBasalisSystem(low_config)

        # High gain
        high_config = NucleusBasalisConfig(novelty_gain=3.0)
        high_nb = NucleusBasalisSystem(high_config)

        # Same prediction error
        for _ in range(10):
            low_nb.update(dt_ms=1.0, prediction_error=0.5)
            high_nb.update(dt_ms=1.0, prediction_error=0.5)

        # High gain should produce higher ACh
        assert high_nb.get_acetylcholine() > low_nb.get_acetylcholine()

    def test_encoding_threshold_parameter(self):
        """Test encoding_threshold affects mode switching."""
        # Low threshold (easier to enter encoding mode)
        low_config = NucleusBasalisConfig(encoding_threshold=0.3)
        low_nb = NucleusBasalisSystem(low_config)

        # High threshold (harder to enter encoding mode)
        high_config = NucleusBasalisConfig(encoding_threshold=0.7)
        high_nb = NucleusBasalisSystem(high_config)

        # Medium ACh level
        low_nb._global_ach = 0.5
        high_nb._global_ach = 0.5

        # Low threshold → encoding mode
        assert low_nb.is_encoding_mode()

        # High threshold → retrieval mode
        assert not high_nb.is_encoding_mode()

    def test_dt_scaling(self, nb):
        """Test decay scales correctly with dt."""
        nb.reset_state()

        # Create burst
        nb.trigger_attention(attention_strength=1.0)

        # Decay with dt=1ms
        nb_copy = NucleusBasalisSystem(nb.config)
        nb_copy.set_state(nb.get_state())

        for _ in range(10):
            nb.update(dt_ms=1.0, prediction_error=0.0)

        # Decay with dt=10ms (fewer steps)
        nb_copy.update(dt_ms=10.0, prediction_error=0.0)

        # Should be similar
        assert nb.get_acetylcholine() == pytest.approx(
            nb_copy.get_acetylcholine(), abs=0.05
        )

    def test_get_diagnostics(self, nb):
        """Test diagnostic information."""
        nb.update(dt_ms=1.0, prediction_error=0.4)
        nb.trigger_attention(attention_strength=0.7)

        diag = nb.get_diagnostics()

        assert "global_ach" in diag
        assert "tonic_ach" in diag
        assert "phasic_ach" in diag
        assert "is_encoding" in diag
        assert "encoding_strength" in diag
        assert "last_pe" in diag

    def test_zero_pe_returns_to_baseline(self, nb):
        """Test zero prediction error returns ACh to baseline over time."""
        nb.reset_state()

        # Elevated state
        for _ in range(20):
            nb.update(dt_ms=1.0, prediction_error=0.7)

        elevated_ach = nb.get_acetylcholine()
        assert elevated_ach > nb.config.baseline_ach

        # Zero PE for extended period
        for _ in range(200):
            nb.update(dt_ms=1.0, prediction_error=0.0)

        baseline_ach = nb.get_acetylcholine()

        # Should approach baseline (retrieval mode)
        assert abs(baseline_ach - nb.config.baseline_ach) < 0.1

    def test_clipping_prevents_overflow(self, nb):
        """Test ACh is clamped to valid range."""
        nb.reset_state()

        # Extreme prediction error repeatedly
        for _ in range(100):
            nb.update(dt_ms=1.0, prediction_error=1.0)
            nb.trigger_attention(attention_strength=1.0)

        # Should be clipped
        ach = nb.get_acetylcholine()
        assert ach >= 0.0
        assert ach <= 2.0  # Reasonable upper bound

    def test_mode_switching_dynamics(self, nb):
        """Test smooth transitions between encoding and retrieval."""
        nb.reset_state()

        # Start in retrieval (low ACh)
        assert not nb.is_encoding_mode()

        # Gradually increase PE → should eventually switch to encoding
        for i in range(20):
            nb.update(dt_ms=1.0, prediction_error=0.05 * i)

        # Should be in encoding mode now
        assert nb.is_encoding_mode()

        # Gradually decrease PE → should return to retrieval
        for i in range(20, 0, -1):
            nb.update(dt_ms=1.0, prediction_error=0.05 * i)

        # Should be back in retrieval mode
        assert not nb.is_encoding_mode()

    def test_multiple_attention_triggers(self, nb):
        """Test multiple attention triggers."""
        nb.reset_state()

        # First trigger
        nb.trigger_attention(attention_strength=0.5)
        first_ach = nb.get_acetylcholine()

        # Let it decay partially
        for _ in range(25):  # Fast decay (τ=50ms)
            nb.update(dt_ms=1.0, prediction_error=0.0)

        # Second trigger
        nb.trigger_attention(attention_strength=0.8)
        second_ach = nb.get_acetylcholine()

        # Second should be higher than partially decayed first
        assert second_ach > first_ach * 0.5
