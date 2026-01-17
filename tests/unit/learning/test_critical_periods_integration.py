"""
Unit tests for critical periods integration into curriculum training.

Tests the integration of CriticalPeriodGating with CurriculumTrainer,
including domain mappings, learning rate modulation, and metrics collection.
"""

from unittest.mock import Mock

import pytest

from thalia.learning.critical_periods import CriticalPeriodGating
from thalia.training.curriculum.stage_manager import (
    CurriculumTrainer,
    StageConfig,
)


class TestCriticalPeriodIntegration:
    """Test suite for critical periods integration."""

    def test_critical_period_gating_initialized(self):
        """Test that CurriculumTrainer initializes critical period gating."""
        brain = Mock()
        trainer = CurriculumTrainer(brain=brain, verbose=False)

        # Test contract: trainer should have critical period gating
        assert hasattr(trainer, "critical_period_gating")
        assert isinstance(trainer.critical_period_gating, CriticalPeriodGating)

        # Test contract: gating should be functional (can get status)
        status = trainer.critical_period_gating.get_window_status("phonology", age=0)
        assert "phase" in status, "Critical period gating should provide phase status"
        assert "multiplier" in status, "Critical period gating should provide multiplier"

    def test_stage_config_has_critical_period_fields(self):
        """Test that StageConfig includes critical period configuration."""
        config = StageConfig()

        assert hasattr(config, "enable_critical_periods")
        assert hasattr(config, "domain_mappings")
        assert config.enable_critical_periods is True  # Default enabled
        assert isinstance(config.domain_mappings, dict)

    def test_critical_period_modulation_phonology_peak(self):
        """Test that phonology domain gets peak multiplier during window."""
        # Use the public CriticalPeriodGating API directly
        gating = CriticalPeriodGating()

        # Mid-phonology window (0-50k)
        status = gating.get_window_status("phonology", age=25000)

        # Should be at peak phase with 1.2x multiplier
        assert status["phase"] == "peak"
        assert status["multiplier"] == pytest.approx(1.2, abs=0.01)

    def test_critical_period_modulation_phonology_late(self):
        """Test that phonology domain declines after window closes."""
        # Use the public CriticalPeriodGating API directly
        gating = CriticalPeriodGating()

        # Long after phonology window (0-50k)
        status = gating.get_window_status("phonology", age=200000)

        # Should be in late/declining phase with reduced multiplier
        assert status["phase"] == "late"
        assert status["multiplier"] < 0.5

    def test_critical_period_modulation_motor_peak(self):
        """Test that motor domain gets peak multiplier during window."""
        # Use the public CriticalPeriodGating API directly
        gating = CriticalPeriodGating()

        # Within motor window (0-75k)
        status = gating.get_window_status("motor", age=30000)

        # Motor has higher peak (1.25x) than other domains
        assert status["phase"] == "peak"
        assert status["multiplier"] == pytest.approx(1.25, abs=0.01)

    def test_critical_period_modulation_multiple_domains(self):
        """Test that multiple domains can be queried simultaneously."""
        # Use the public CriticalPeriodGating API directly
        gating = CriticalPeriodGating()

        # Grammar peak, semantics starting (step 100k)
        grammar_status = gating.get_window_status("grammar", age=100000)
        semantics_status = gating.get_window_status("semantics", age=100000)

        # Grammar should be at peak
        assert grammar_status["phase"] == "peak"
        assert grammar_status["multiplier"] > 1.0

        # Semantics should also be active
        assert semantics_status["multiplier"] > 0.0

        # Verify averaging logic would work
        avg = (grammar_status["multiplier"] + semantics_status["multiplier"]) / 2
        assert avg > 0.5  # Combined effect should be positive

    def test_critical_period_modulation_unknown_domain(self):
        """Test that unknown domains raise appropriate errors."""
        from thalia.core.errors import ConfigurationError

        gating = CriticalPeriodGating()

        # Unknown domains should raise ConfigurationError
        with pytest.raises(ConfigurationError, match="(?i)(unknown|domain|not found|invalid)"):
            gating.get_window_status("nonexistent_domain", age=50000)

    def test_critical_period_gating_all_domains(self):
        """Test that all expected domains are available."""
        gating = CriticalPeriodGating()

        # Verify critical domains are defined
        all_domains = gating.get_all_domains()
        expected_domains = {"phonology", "motor", "grammar", "semantics", "face_recognition"}

        # All expected domains should be present
        for domain in expected_domains:
            assert domain in all_domains

    def test_critical_period_metrics_collection(self):
        """Test that critical period status can be queried for metrics."""
        gating = CriticalPeriodGating()

        # At step 25k, phonology should be at peak
        status = gating.get_window_status("phonology", age=25000)

        # Verify status contains all expected metric fields
        assert "multiplier" in status
        assert "phase" in status
        assert "progress" in status

        # Phonology should be at peak with correct multiplier
        assert status["multiplier"] == pytest.approx(1.2, abs=0.01)
        assert status["phase"] == "peak"

    def test_critical_period_phase_transitions(self):
        """Test that phases transition correctly across age windows."""
        gating = CriticalPeriodGating()

        # Motor domain: early age should be peak
        early_status = gating.get_window_status("motor", age=0)
        assert early_status["phase"] == "peak"

        # Same phase later in window
        mid_status = gating.get_window_status("motor", age=10000)
        assert mid_status["phase"] == "peak"

        # After window closes, should transition to late
        late_status = gating.get_window_status("motor", age=200000)
        assert late_status["phase"] == "late"
        assert late_status["multiplier"] < early_status["multiplier"]

    def test_domain_mappings_in_stage_config(self):
        """Test that domain mappings work in StageConfig."""
        config = StageConfig(
            enable_critical_periods=True,
            domain_mappings={
                "motor_control": ["motor"],
                "reaching": ["motor", "face_recognition"],
                "phoneme_task": ["phonology"],
            },
        )

        assert config.enable_critical_periods is True
        assert config.domain_mappings["motor_control"] == ["motor"]
        assert config.domain_mappings["reaching"] == ["motor", "face_recognition"]
        assert config.domain_mappings["phoneme_task"] == ["phonology"]

    def test_critical_periods_can_be_disabled(self):
        """Test that critical periods can be disabled in config."""
        config = StageConfig(
            enable_critical_periods=False,
        )

        assert config.enable_critical_periods is False

    def test_critical_period_progress_tracking(self):
        """Test that progress through a critical period is tracked correctly."""
        gating = CriticalPeriodGating()

        # At start of window
        start_status = gating.get_window_status("phonology", age=0)
        assert start_status["progress"] == pytest.approx(0.0, abs=0.01)

        # Mid-window
        mid_status = gating.get_window_status("phonology", age=25000)
        assert 0.0 < mid_status["progress"] < 1.0

        # After window
        end_status = gating.get_window_status("phonology", age=50000)
        assert end_status["progress"] >= 1.0 or end_status["phase"] == "late"


class TestCriticalPeriodWindows:
    """Test critical period window configurations."""

    def test_phonology_window(self):
        """Test phonology critical period window."""
        gating = CriticalPeriodGating()

        # Peak phase (step 25k within 0-50k)
        status = gating.get_window_status("phonology", 25000)
        assert status["phase"] == "peak"
        assert status["multiplier"] == pytest.approx(1.2, abs=0.01)

        # Late phase (step 100k after 50k)
        status = gating.get_window_status("phonology", 100000)
        assert status["phase"] == "late"
        assert status["multiplier"] < 1.0

    def test_grammar_window(self):
        """Test grammar critical period window."""
        gating = CriticalPeriodGating()

        # Early phase (before 25k)
        status = gating.get_window_status("grammar", 10000)
        assert status["phase"] == "early"
        assert status["multiplier"] == pytest.approx(0.5, abs=0.01)

        # Peak phase (within 25k-150k)
        status = gating.get_window_status("grammar", 100000)
        assert status["phase"] == "peak"
        assert status["multiplier"] == pytest.approx(1.2, abs=0.01)

    def test_motor_window_higher_peak(self):
        """Test motor has higher peak multiplier."""
        gating = CriticalPeriodGating()

        status = gating.get_window_status("motor", 30000)
        assert status["phase"] == "peak"
        assert status["multiplier"] == pytest.approx(1.25, abs=0.01)  # Higher than others

    def test_semantics_never_closes(self):
        """Test semantics window has extended duration."""
        gating = CriticalPeriodGating()

        # Should still be in peak even at 200k
        status = gating.get_window_status("semantics", 200000)
        assert status["phase"] == "peak"
        assert status["multiplier"] > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
