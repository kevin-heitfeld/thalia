"""
Unit tests for VTADopamineSystem (centralized dopamine management).

Tests cover:
- Tonic dopamine computation from intrinsic rewards
- Phasic dopamine bursts/dips from external rewards
- Adaptive RPE normalization
- Exponential decay dynamics
- State management (get/set/reset)
- Health checks
"""

import pytest
import torch

from thalia.core.vta import VTADopamineSystem, VTAConfig


class TestVTADopamineSystem:
    """Test suite for VTA dopamine system."""

    @pytest.fixture
    def vta(self):
        """Create VTA system with default config."""
        config = VTAConfig(
            phasic_decay_per_ms=0.995,  # τ=200ms
            tonic_alpha=0.05,
            rpe_avg_tau=0.9,
            rpe_clip=2.0,
        )
        return VTADopamineSystem(config)

    def test_initialization(self, vta):
        """Test VTA initializes with zero dopamine."""
        assert vta.config.phasic_decay_per_ms == 0.995
        assert vta._global_dopamine == 0.0
        assert vta._tonic_dopamine == 0.0
        assert vta._phasic_dopamine == 0.0

    def test_tonic_dopamine_from_intrinsic_reward(self, vta):
        """Test tonic dopamine increases with intrinsic reward."""
        # No intrinsic reward → stays near zero
        vta.update(dt_ms=1.0, intrinsic_reward=0.0)
        assert vta._tonic_dopamine == pytest.approx(0.0, abs=0.01)

        # Positive intrinsic reward → above zero
        vta.reset_state()
        for _ in range(20):
            vta.update(dt_ms=1.0, intrinsic_reward=0.5)
        assert vta._tonic_dopamine > 0.0

        # Negative intrinsic reward → below zero
        vta.reset_state()
        for _ in range(20):
            vta.update(dt_ms=1.0, intrinsic_reward=-0.3)
        assert vta._tonic_dopamine < 0.0

    def test_phasic_burst_from_external_reward(self, vta):
        """Test phasic dopamine burst when external reward delivered."""
        vta.reset_state()
        initial_da = vta.get_global_dopamine()

        # Deliver reward → phasic burst
        vta.deliver_reward(external_reward=1.0, expected_value=0.0)
        burst_da = vta.get_global_dopamine()

        assert burst_da > initial_da
        assert vta._phasic_dopamine > 0.0

    def test_phasic_dip_from_negative_rpe(self, vta):
        """Test phasic dopamine dip when reward worse than expected."""
        vta.reset_state()
        vta._tonic_dopamine = 0.5  # Set baseline

        # Expected reward but got nothing → dip
        vta.deliver_reward(external_reward=0.0, expected_value=1.0)
        dip_da = vta.get_global_dopamine()

        assert dip_da < 0.5
        assert vta._phasic_dopamine < 0.0

    def test_dopamine_decay(self, vta):
        """Test dopamine decays back to tonic baseline."""
        vta.reset_state()

        # Create phasic burst
        vta.deliver_reward(external_reward=1.0, expected_value=0.0)
        peak_da = vta.get_global_dopamine()

        # Decay over time (no new rewards)
        for _ in range(500):  # ~500ms
            vta.update(dt_ms=1.0, intrinsic_reward=0.0)

        decayed_da = vta.get_global_dopamine()

        assert decayed_da < peak_da
        # Should approach tonic baseline
        assert abs(decayed_da - vta._tonic_dopamine) < 0.1

    def test_adaptive_rpe_normalization(self, vta):
        """Test RPE normalization adapts to reward magnitude."""
        vta.reset_state()

        # First reward → _avg_abs_rpe starts adapting from initial 0.5
        vta.deliver_reward(external_reward=1.0, expected_value=0.0)
        first_avg = vta._avg_abs_rpe

        # After many large rewards, normalization adapts
        for _ in range(20):
            vta.deliver_reward(external_reward=1.0, expected_value=0.0)
            vta.update(dt_ms=10.0, intrinsic_reward=0.0)

        later_avg = vta._avg_abs_rpe

        # Running average should have stabilized and be different from initial
        # (EMA causes it to converge toward recent |RPE| values)
        assert abs(later_avg - first_avg) > 0.1  # Significant adaptation occurred

    def test_state_persistence(self, vta):
        """Test get_state/set_state for checkpointing."""
        # Set some state
        vta.update(dt_ms=1.0, intrinsic_reward=0.3)
        vta.deliver_reward(external_reward=1.0, expected_value=0.5)

        # Save state
        state = vta.get_state()

        # Create new VTA and restore
        vta2 = VTADopamineSystem(vta.config)
        vta2.set_state(state)

        # Should match
        assert vta2._global_dopamine == pytest.approx(vta._global_dopamine)
        assert vta2._tonic_dopamine == pytest.approx(vta._tonic_dopamine)
        assert vta2._phasic_dopamine == pytest.approx(vta._phasic_dopamine)

    def test_reset_state(self, vta):
        """Test reset returns to baseline."""
        # Modify state
        vta.update(dt_ms=1.0, intrinsic_reward=0.5)
        vta.deliver_reward(external_reward=1.0, expected_value=0.0)

        assert vta._global_dopamine != 0.0

        # Reset
        vta.reset_state()

        assert vta._global_dopamine == 0.0
        assert vta._tonic_dopamine == 0.0
        assert vta._phasic_dopamine == 0.0

    def test_health_check_healthy(self, vta):
        """Test health check passes for normal state."""
        vta.reset_state()
        vta.update(dt_ms=1.0, intrinsic_reward=0.1)
        vta.deliver_reward(external_reward=0.5, expected_value=0.5)  # Deliver at least one reward

        health = vta.check_health()

        assert health["is_healthy"] is True
        assert len(health["issues"]) == 0

    def test_health_check_detects_runaway(self, vta):
        """Test health check detects runaway dopamine."""
        # Force very high dopamine
        vta._global_dopamine = 2.0

        health = vta.check_health()

        assert health["is_healthy"] is False
        assert any("high dopamine" in issue.lower() for issue in health["issues"])

    def test_health_check_detects_negative(self, vta):
        """Test health check detects excessive negative dopamine."""
        # Force very negative dopamine
        vta._global_dopamine = -2.0

        health = vta.check_health()

        assert health["is_healthy"] is False
        assert any("high dopamine" in issue.lower() for issue in health["issues"])

    def test_zero_rpe_no_phasic(self, vta):
        """Test zero RPE produces no phasic dopamine."""
        vta.reset_state()

        # Reward matches expectation exactly
        vta.deliver_reward(external_reward=0.5, expected_value=0.5)

        # Should have minimal phasic component
        assert abs(vta._phasic_dopamine) < 0.1

    def test_dt_scaling(self, vta):
        """Test decay scales correctly with dt."""
        vta.reset_state()

        # Create burst
        vta.deliver_reward(external_reward=1.0, expected_value=0.0)

        # Decay with dt=1ms
        vta_copy = VTADopamineSystem(vta.config)
        vta_copy.set_state(vta.get_state())

        for _ in range(10):
            vta.update(dt_ms=1.0, intrinsic_reward=0.0)

        # Decay with dt=10ms (fewer steps)
        vta_copy.update(dt_ms=10.0, intrinsic_reward=0.0)

        # Should be similar (decay is per-ms)
        assert vta.get_global_dopamine() == pytest.approx(
            vta_copy.get_global_dopamine(), abs=0.05
        )

    def test_check_health_returns_metrics(self, vta):
        """Test health check returns diagnostic information."""
        vta.update(dt_ms=1.0, intrinsic_reward=0.2)
        vta.deliver_reward(external_reward=1.0, expected_value=0.3)

        health = vta.check_health()

        # Should have key metrics
        assert "is_healthy" in health
        assert "issues" in health
        assert "tonic" in health
        assert "phasic" in health
        assert "global" in health
        assert health["tonic"] == vta._tonic_dopamine
        assert health["phasic"] == vta._phasic_dopamine
        assert health["global"] == vta._global_dopamine

    def test_multiple_rewards_per_timestep(self, vta):
        """Test multiple reward deliveries between updates."""
        vta.reset_state()

        # Deliver multiple rewards
        vta.deliver_reward(external_reward=0.5, expected_value=0.0)
        vta.deliver_reward(external_reward=0.5, expected_value=0.0)

        # Update once
        vta.update(dt_ms=1.0, intrinsic_reward=0.0)

        # Should reflect both rewards (second overwrites first)
        assert vta._phasic_dopamine > 0.0

    def test_clipping_prevents_overflow(self, vta):
        """Test dopamine is clamped to valid range."""
        vta.reset_state()

        # Deliver huge reward repeatedly
        for _ in range(100):
            vta.deliver_reward(external_reward=10.0, expected_value=0.0)
            vta.update(dt_ms=1.0, intrinsic_reward=1.0)

        # Should be clipped
        da = vta.get_global_dopamine()
        assert da >= vta.config.min_dopamine
        assert da <= vta.config.max_dopamine

    def test_config_parameters_affect_behavior(self):
        """Test different configs produce different behavior."""
        # Fast decay config
        fast_config = VTAConfig(phasic_decay_per_ms=0.9)  # τ~10ms
        fast_vta = VTADopamineSystem(fast_config)

        # Slow decay config
        slow_config = VTAConfig(phasic_decay_per_ms=0.999)  # τ~1000ms
        slow_vta = VTADopamineSystem(slow_config)

        # Both get same reward
        fast_vta.deliver_reward(external_reward=1.0, expected_value=0.0)
        slow_vta.deliver_reward(external_reward=1.0, expected_value=0.0)

        # Decay for 100ms
        for _ in range(100):
            fast_vta.update(dt_ms=1.0, intrinsic_reward=0.0)
            slow_vta.update(dt_ms=1.0, intrinsic_reward=0.0)

        # Fast should have decayed more
        assert fast_vta.get_global_dopamine() < slow_vta.get_global_dopamine()
