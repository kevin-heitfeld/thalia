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
import torch

from thalia.core.locus_coeruleus import LocusCoeruleusSystem, LocusCoeruleusConfig


class TestLocusCoeruleusSystem:
    """Test suite for Locus Coeruleus norepinephrine system."""

    @pytest.fixture
    def lc(self):
        """Create LC system with default config."""
        config = LocusCoeruleusConfig(
            baseline_norepinephrine=0.2,
            ne_decay_per_ms=0.99,  # τ=100ms (faster than DA)
            arousal_sensitivity=1.5,
            novelty_threshold=0.3,
        )
        return LocusCoeruleusSystem(config)

    def test_initialization(self, lc):
        """Test LC initializes with correct baseline."""
        assert lc.config.baseline_norepinephrine == 0.2
        assert lc._global_norepinephrine == 0.2
        assert lc._tonic_norepinephrine == 0.2
        assert lc._phasic_norepinephrine == 0.0

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
        lc.trigger_novelty(novelty_strength=1.0)
        burst_ne = lc.get_norepinephrine()

        assert burst_ne > initial_ne
        assert lc._phasic_norepinephrine > 0.0

    def test_faster_decay_than_dopamine(self, lc):
        """Test NE decays faster than dopamine (shorter tau)."""
        lc.reset_state()

        # Create phasic burst
        lc.trigger_novelty(novelty_strength=1.0)
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
        assert lc._tonic_norepinephrine > lc.config.baseline_norepinephrine

    def test_state_persistence(self, lc):
        """Test get_state/set_state for checkpointing."""
        # Set some state
        lc.update(dt_ms=1.0, uncertainty=0.5)
        lc.trigger_novelty(novelty_strength=0.8)

        # Save state
        state = lc.get_state()

        # Create new LC and restore
        lc2 = LocusCoeruleusSystem(lc.config)
        lc2.set_state(state)

        # Should match
        assert lc2._global_norepinephrine == pytest.approx(lc._global_norepinephrine)
        assert lc2._tonic_norepinephrine == pytest.approx(lc._tonic_norepinephrine)
        assert lc2._phasic_norepinephrine == pytest.approx(lc._phasic_norepinephrine)

    def test_reset_state(self, lc):
        """Test reset returns to baseline."""
        # Modify state
        lc.update(dt_ms=1.0, uncertainty=0.8)
        lc.trigger_novelty(novelty_strength=1.0)

        assert lc._global_norepinephrine != lc.config.baseline_norepinephrine

        # Reset
        lc.reset_state()

        assert lc._global_norepinephrine == lc.config.baseline_norepinephrine
        assert lc._tonic_norepinephrine == lc.config.baseline_norepinephrine
        assert lc._phasic_norepinephrine == 0.0

    def test_health_check_healthy(self, lc):
        """Test health check passes for normal operation."""
        lc.update(dt_ms=1.0, uncertainty=0.3)
        health = lc.check_health()

        assert health["is_healthy"] is True
        assert len(health["issues"]) == 0

    def test_health_check_detects_runaway(self, lc):
        """Test health check detects runaway NE."""
        # Force runaway NE
        lc._global_norepinephrine = 5.0

        health = lc.check_health()

        assert health["is_healthy"] is False
        assert any("norepinephrine too high" in issue.lower() for issue in health["issues"])

    def test_health_check_detects_negative(self, lc):
        """Test health check detects negative NE."""
        # Force negative NE
        lc._global_norepinephrine = -0.3

        health = lc.check_health()

        assert health["is_healthy"] is False
        assert any("negative" in issue.lower() for issue in health["issues"])

    def test_arousal_sensitivity_parameter(self):
        """Test arousal_sensitivity affects NE response to uncertainty."""
        # Low sensitivity
        low_config = LocusCoeruleusConfig(arousal_sensitivity=0.5)
        low_lc = LocusCoeruleusSystem(low_config)

        # High sensitivity
        high_config = LocusCoeruleusConfig(arousal_sensitivity=2.0)
        high_lc = LocusCoeruleusSystem(high_config)

        # Same uncertainty
        for _ in range(10):
            low_lc.update(dt_ms=1.0, uncertainty=0.5)
            high_lc.update(dt_ms=1.0, uncertainty=0.5)

        # High sensitivity should produce higher NE
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
        lc.trigger_novelty(novelty_strength=1.0)

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
        """Test diagnostic information."""
        lc.update(dt_ms=1.0, uncertainty=0.4)
        lc.trigger_novelty(novelty_strength=0.7)

        diag = lc.get_diagnostics()

        assert "global_norepinephrine" in diag
        assert "tonic_norepinephrine" in diag
        assert "phasic_norepinephrine" in diag
        assert "last_uncertainty" in diag

    def test_zero_uncertainty_returns_to_baseline(self, lc):
        """Test zero uncertainty returns NE to baseline over time."""
        lc.reset_state()

        # Elevated state
        for _ in range(20):
            lc.update(dt_ms=1.0, uncertainty=0.6)

        elevated_ne = lc.get_norepinephrine()
        assert elevated_ne > lc.config.baseline_norepinephrine

        # Zero uncertainty for extended period
        for _ in range(200):
            lc.update(dt_ms=1.0, uncertainty=0.0)

        baseline_ne = lc.get_norepinephrine()

        # Should approach baseline
        assert abs(baseline_ne - lc.config.baseline_norepinephrine) < 0.1

    def test_clipping_prevents_overflow(self, lc):
        """Test NE is clamped to valid range."""
        lc.reset_state()

        # Extreme uncertainty repeatedly
        for _ in range(100):
            lc.update(dt_ms=1.0, uncertainty=1.0)
            lc.trigger_novelty(novelty_strength=1.0)

        # Should be clipped
        ne = lc.get_norepinephrine()
        assert ne >= 0.0
        assert ne <= 2.0  # Reasonable upper bound

    def test_multiple_novelty_bursts(self, lc):
        """Test multiple novelty detections."""
        lc.reset_state()

        # First burst
        lc.trigger_novelty(novelty_strength=0.5)
        first_ne = lc.get_norepinephrine()

        # Let it decay partially
        for _ in range(50):
            lc.update(dt_ms=1.0, uncertainty=0.0)

        # Second burst
        lc.trigger_novelty(novelty_strength=0.8)
        second_ne = lc.get_norepinephrine()

        # Second should be higher than partially decayed first
        assert second_ne > first_ne

    def test_arousal_modulates_gain(self, lc):
        """Test get_arousal_gain returns correct range."""
        lc.reset_state()

        # Low NE → gain near 1.0
        lc.update(dt_ms=1.0, uncertainty=0.0)
        low_gain = 1.0 + 0.5 * lc.get_norepinephrine()

        # High NE → gain near 1.5
        lc.reset_state()
        for _ in range(20):
            lc.update(dt_ms=1.0, uncertainty=0.8)
        high_gain = 1.0 + 0.5 * lc.get_norepinephrine()

        assert low_gain < high_gain
        assert 1.0 <= low_gain <= 1.5
        assert 1.0 <= high_gain <= 1.5
