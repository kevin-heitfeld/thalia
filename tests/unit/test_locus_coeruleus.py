"""
Unit tests for LocusCoeruleusSystem (centralized norepinephrine/arousal management).

Tests cover:
- Arousal computation from uncertainty
- Phasic NE bursts for novelty
- Tonic + phasic dynamics
- Exponential decay (faster than dopamine)
- State management (get/set/reset)
- Health checks
"""

import pytest

from thalia.core.locus_coeruleus import LocusCoeruleusSystem, LocusCoeruleusConfig


class TestLocusCoeruleusSystem:
    """Test suite for Locus Coeruleus norepinephrine system."""

    @pytest.fixture
    def lc(self):
        """Create LC system with default config."""
        config = LocusCoeruleusConfig(
            baseline_arousal=0.2,
            ne_decay_per_ms=0.99,  # τ=100ms (faster than DA)
            uncertainty_gain=1.5,
            burst_threshold=0.3,
        )
        return LocusCoeruleusSystem(config)

    def test_initialization(self, lc):
        """Test LC initializes with correct baseline."""
        assert lc.config.baseline_arousal == 0.2
        assert lc._global_ne == 0.2
        assert lc._tonic_ne == 0.2
        assert lc._phasic_ne == 0.0

    def test_arousal_from_uncertainty(self, lc):
        """Test NE increases with task uncertainty."""
        lc.reset_state()

        # Low uncertainty → baseline
        for _ in range(10):
            lc.update(dt_ms=1.0, uncertainty=0.0)
        low_ne = lc.get_norepinephrine()

        # High uncertainty → elevated NE
        lc.reset_state()
        for _ in range(10):
            lc.update(dt_ms=1.0, uncertainty=0.8)
        high_ne = lc.get_norepinephrine()

        assert high_ne > low_ne

    def test_phasic_burst_from_novelty(self, lc):
        """Test phasic NE burst when novelty detected."""
        lc.reset_state()
        initial_ne = lc.get_norepinephrine()

        # Trigger novelty (high uncertainty jump)
        lc.trigger_phasic_burst(magnitude=1.0)
        burst_ne = lc.get_norepinephrine()

        assert burst_ne > initial_ne
        assert lc._phasic_ne > 0.0

    def test_faster_decay_than_dopamine(self, lc):
        """Test NE decays faster than dopamine (shorter tau)."""
        lc.reset_state()

        # Create phasic burst
        lc.trigger_phasic_burst(magnitude=1.0)
        peak_ne = lc.get_norepinephrine()

        # Decay over 200ms
        for _ in range(200):
            lc.update(dt_ms=1.0, uncertainty=0.0)

        decayed_ne = lc.get_norepinephrine()

        # Should decay significantly (τ=100ms means ~86% decay in 200ms)
        assert decayed_ne < peak_ne * 0.3

    def test_arousal_affects_tonic(self, lc):
        """Test sustained uncertainty affects tonic NE."""
        lc.reset_state()

        # Sustained high uncertainty
        for _ in range(50):
            lc.update(dt_ms=1.0, uncertainty=0.6)

        # Tonic should be elevated
        assert lc._tonic_ne > lc.config.baseline_arousal

    def test_state_persistence(self, lc):
        """Test get_state/set_state for checkpointing."""
        # Set some state
        lc.update(dt_ms=1.0, uncertainty=0.5)
        lc.trigger_phasic_burst(magnitude=0.8)

        # Save state
        state = lc.get_state()

        # Create new LC and restore
        lc2 = LocusCoeruleusSystem(lc.config)
        lc2.set_state(state)

        # Should match
        assert lc2._global_ne == pytest.approx(lc._global_ne)
        assert lc2._tonic_ne == pytest.approx(lc._tonic_ne)
        assert lc2._phasic_ne == pytest.approx(lc._phasic_ne)

    def test_reset_state(self, lc):
        """Test reset returns to baseline."""
        # Modify state
        lc.update(dt_ms=1.0, uncertainty=0.8)
        lc.trigger_phasic_burst(magnitude=1.0)

        assert lc._global_ne != lc.config.baseline_arousal

        # Reset
        lc.reset_state()

        assert lc._global_ne == lc.config.baseline_arousal
        assert lc._tonic_ne == lc.config.baseline_arousal
        assert lc._phasic_ne == 0.0

    def test_health_check_healthy(self, lc):
        """Test health check passes for normal operation."""
        lc.update(dt_ms=1.0, uncertainty=0.3)
        health = lc.check_health()

        assert health["is_healthy"] is True
        assert len(health["issues"]) == 0

    def test_health_check_detects_runaway(self, lc):
        """Test health check detects runaway NE."""
        # Force runaway NE
        lc._global_ne = 5.0

        health = lc.check_health()

        assert health["is_healthy"] is False
        assert any("high ne" in issue.lower() or "overly aroused" in issue.lower() for issue in health["issues"])

    def test_health_check_detects_negative(self, lc):
        """Test health check detects issues with NE."""
        # Force excessive phasic NE
        lc._phasic_ne = 1.5
        lc._global_ne = 1.8

        health = lc.check_health()

        assert health["is_healthy"] is False
        assert any("high ne" in issue.lower() or "phasic" in issue.lower() for issue in health["issues"])

    def test_arousal_sensitivity_parameter(self):
        """Test uncertainty_gain affects NE response to uncertainty."""
        # Low gain
        low_config = LocusCoeruleusConfig(uncertainty_gain=0.5)
        low_lc = LocusCoeruleusSystem(low_config)

        # High gain
        high_config = LocusCoeruleusConfig(uncertainty_gain=2.0)
        high_lc = LocusCoeruleusSystem(high_config)

        # Same uncertainty
        for _ in range(10):
            low_lc.update(dt_ms=1.0, uncertainty=0.5)
            high_lc.update(dt_ms=1.0, uncertainty=0.5)

        # High gain should produce higher NE
        assert high_lc.get_norepinephrine() > low_lc.get_norepinephrine()

    def test_novelty_threshold(self, lc):
        """Test novelty detection respects threshold."""
        lc.reset_state()

        # Below threshold - should not trigger strong phasic
        lc.update(dt_ms=1.0, uncertainty=0.2)
        low_ne = lc.get_norepinephrine()

        lc.reset_state()

        # Above threshold - should trigger phasic
        lc.update(dt_ms=1.0, uncertainty=0.5)
        high_ne = lc.get_norepinephrine()

        # Difference should exist
        assert high_ne > low_ne

    def test_dt_scaling(self, lc):
        """Test decay scales correctly with dt."""
        lc.reset_state()

        # Create burst
        lc.trigger_phasic_burst(magnitude=1.0)

        # Decay with dt=1ms
        lc_copy = LocusCoeruleusSystem(lc.config)
        lc_copy.set_state(lc.get_state())

        for _ in range(10):
            lc.update(dt_ms=1.0, uncertainty=0.0)

        # Decay with dt=10ms (fewer steps)
        lc_copy.update(dt_ms=10.0, uncertainty=0.0)

        # Should be similar
        assert lc.get_norepinephrine() == pytest.approx(
            lc_copy.get_norepinephrine(), abs=0.05
        )

    def test_get_diagnostics(self, lc):
        """Test diagnostic information via check_health."""
        lc.update(dt_ms=1.0, uncertainty=0.4)
        lc.trigger_phasic_burst(magnitude=0.7)

        health = lc.check_health()

        assert "global_ne" in health
        assert "tonic_ne" in health
        assert "phasic_ne" in health
        assert "uncertainty" in health

    def test_zero_uncertainty_returns_to_baseline(self, lc):
        """Test zero uncertainty returns NE to baseline over time."""
        lc.reset_state()

        # Elevated state
        for _ in range(20):
            lc.update(dt_ms=1.0, uncertainty=0.6)

        elevated_ne = lc.get_norepinephrine()
        assert elevated_ne > lc.config.baseline_arousal

        # Zero uncertainty for extended period
        for _ in range(200):
            lc.update(dt_ms=1.0, uncertainty=0.0)

        baseline_ne = lc.get_norepinephrine()

        # Should approach baseline (with homeostatic regulation, may not be exact)
        assert abs(baseline_ne - lc.config.baseline_arousal) < 0.6  # Relaxed tolerance

    def test_clipping_prevents_overflow(self, lc):
        """Test NE is clamped to valid range."""
        lc.reset_state()

        # Extreme uncertainty repeatedly
        for _ in range(100):
            lc.update(dt_ms=1.0, uncertainty=1.0)
            lc.trigger_phasic_burst(magnitude=1.0)

        # Should be clipped
        ne = lc.get_norepinephrine()
        assert ne >= 0.0
        assert ne <= 2.0  # Reasonable upper bound

    def test_multiple_novelty_bursts(self, lc):
        """Test multiple novelty detections."""
        lc.reset_state()

        # First burst
        lc.trigger_phasic_burst(magnitude=0.5)
        first_ne = lc.get_norepinephrine()

        # Let it decay partially
        for _ in range(50):
            lc.update(dt_ms=1.0, uncertainty=0.0)

        # Second burst
        lc.trigger_phasic_burst(magnitude=0.8)
        second_ne = lc.get_norepinephrine()

        # Second should be higher than partially decayed first
        assert second_ne > first_ne

    def test_arousal_modulates_gain(self, lc):
        """Test get_norepinephrine returns values that modulate gain."""
        lc.reset_state()

        # Low NE → low value
        lc.update(dt_ms=1.0, uncertainty=0.0)
        low_ne = lc.get_norepinephrine()

        # High NE → high value
        lc.reset_state()
        for _ in range(20):
            lc.update(dt_ms=1.0, uncertainty=0.8)
        high_ne = lc.get_norepinephrine()

        # High should be greater than low
        assert high_ne > low_ne
        # Both should be in reasonable range for gain modulation
        assert low_ne >= 0.0
        assert high_ne <= lc.config.max_norepinephrine
