"""
Unit tests for critical periods integration into curriculum training.

Tests the integration of CriticalPeriodGating with CurriculumTrainer,
including domain mappings, learning rate modulation, and metrics collection.
"""

import pytest
from unittest.mock import Mock

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
        
        assert hasattr(trainer, 'critical_period_gating')
        assert isinstance(trainer.critical_period_gating, CriticalPeriodGating)
        assert hasattr(trainer, '_last_phase')
        assert isinstance(trainer._last_phase, dict)

    def test_stage_config_has_critical_period_fields(self):
        """Test that StageConfig includes critical period configuration."""
        config = StageConfig()
        
        assert hasattr(config, 'enable_critical_periods')
        assert hasattr(config, 'domain_mappings')
        assert config.enable_critical_periods is True  # Default enabled
        assert isinstance(config.domain_mappings, dict)

    def test_critical_period_modulation_phonology_peak(self):
        """Test that phonology domain gets peak multiplier during window."""
        brain = Mock()
        brain.set_plasticity_modulator = Mock()
        
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        trainer.global_step = 25000  # Mid-phonology window (0-50k)
        
        # Apply modulation for phonology task
        multipliers = trainer._apply_critical_period_modulation(
            task_name='phoneme_discrimination',
            domains=['phonology'],
            age=25000,
        )
        
        # Should be at peak (1.2x)
        assert 'phonology' in multipliers
        assert multipliers['phonology'] == pytest.approx(1.2, abs=0.01)
        
        # Brain should receive the modulator
        brain.set_plasticity_modulator.assert_called_once()
        call_args = brain.set_plasticity_modulator.call_args
        assert call_args[0][0] == pytest.approx(1.2, abs=0.01)

    def test_critical_period_modulation_phonology_late(self):
        """Test that phonology domain declines after window closes."""
        brain = Mock()
        brain.set_plasticity_modulator = Mock()
        
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        trainer.global_step = 200000  # Long after phonology window
        
        multipliers = trainer._apply_critical_period_modulation(
            task_name='phoneme_discrimination',
            domains=['phonology'],
            age=200000,
        )
        
        # Should be in declining phase (<0.5)
        assert 'phonology' in multipliers
        assert multipliers['phonology'] < 0.5
        
        # Brain should receive reduced modulator
        brain.set_plasticity_modulator.assert_called_once()
        call_args = brain.set_plasticity_modulator.call_args
        assert call_args[0][0] < 0.5

    def test_critical_period_modulation_motor_peak(self):
        """Test that motor domain gets peak multiplier during window."""
        brain = Mock()
        brain.set_plasticity_modulator = Mock()
        
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        trainer.global_step = 30000  # Within motor window (0-75k)
        
        multipliers = trainer._apply_critical_period_modulation(
            task_name='motor_control',
            domains=['motor'],
            age=30000,
        )
        
        # Motor has higher peak (1.25x)
        assert 'motor' in multipliers
        assert multipliers['motor'] == pytest.approx(1.25, abs=0.01)

    def test_critical_period_modulation_multiple_domains(self):
        """Test averaging when task has multiple domains."""
        brain = Mock()
        brain.set_plasticity_modulator = Mock()
        
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        trainer.global_step = 100000  # Grammar peak, semantics starting
        
        multipliers = trainer._apply_critical_period_modulation(
            task_name='grammar_semantic_task',
            domains=['grammar', 'semantics'],
            age=100000,
        )
        
        # Should have both domains
        assert 'grammar' in multipliers
        assert 'semantics' in multipliers
        
        # Average should be applied to brain
        brain.set_plasticity_modulator.assert_called_once()
        call_args = brain.set_plasticity_modulator.call_args
        avg_multiplier = (multipliers['grammar'] + multipliers['semantics']) / 2
        assert call_args[0][0] == pytest.approx(avg_multiplier, abs=0.01)

    def test_critical_period_modulation_unknown_domain(self):
        """Test that unknown domains are handled gracefully."""
        brain = Mock()
        brain.set_plasticity_modulator = Mock()
        
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        
        # Should not raise exception
        multipliers = trainer._apply_critical_period_modulation(
            task_name='unknown_task',
            domains=['nonexistent_domain'],
            age=50000,
        )
        
        # Should return empty dict
        assert multipliers == {}
        
        # Brain should not be called
        brain.set_plasticity_modulator.assert_not_called()

    def test_critical_period_modulation_empty_domains(self):
        """Test that empty domain list is handled."""
        brain = Mock()
        brain.set_plasticity_modulator = Mock()
        
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        
        multipliers = trainer._apply_critical_period_modulation(
            task_name='some_task',
            domains=[],
            age=50000,
        )
        
        assert multipliers == {}
        brain.set_plasticity_modulator.assert_not_called()

    def test_critical_period_metrics_collection(self):
        """Test that critical period metrics are collected."""
        brain = Mock()
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        trainer.global_step = 25000
        
        metrics = trainer._collect_metrics()
        
        # Should have metrics for all domains
        all_domains = trainer.critical_period_gating.get_all_domains()
        for domain in all_domains:
            assert f'critical_period/{domain}_multiplier' in metrics
            assert f'critical_period/{domain}_progress' in metrics
            assert f'critical_period/{domain}_phase' in metrics
        
        # Phonology should be at peak at step 25k
        assert metrics['critical_period/phonology_multiplier'] == pytest.approx(1.2, abs=0.01)
        assert metrics['critical_period/phonology_phase'] == 1.0  # peak

    def test_critical_period_phase_transitions_logged(self, capsys):
        """Test that phase transitions are logged."""
        brain = Mock()
        brain.set_plasticity_modulator = Mock()
        
        trainer = CurriculumTrainer(brain=brain, verbose=True)
        
        # First call at early phase
        trainer.global_step = 0
        trainer._apply_critical_period_modulation(
            task_name='motor_control',
            domains=['motor'],
            age=0,
        )
        
        captured = capsys.readouterr()
        assert 'motor' in captured.out
        assert 'peak' in captured.out.lower()
        
        # Should have recorded phase
        assert trainer._last_phase['motor'] == 'peak'
        
        # Second call in same phase (no log)
        trainer._apply_critical_period_modulation(
            task_name='motor_control',
            domains=['motor'],
            age=10000,
        )
        
        captured = capsys.readouterr()
        # Should not log again (same phase)
        assert 'entering' not in captured.out

    def test_domain_mappings_in_stage_config(self):
        """Test that domain mappings work in StageConfig."""
        config = StageConfig(
            enable_critical_periods=True,
            domain_mappings={
                'motor_control': ['motor'],
                'reaching': ['motor', 'face_recognition'],
                'phoneme_task': ['phonology'],
            },
        )
        
        assert config.enable_critical_periods is True
        assert config.domain_mappings['motor_control'] == ['motor']
        assert config.domain_mappings['reaching'] == ['motor', 'face_recognition']
        assert config.domain_mappings['phoneme_task'] == ['phonology']

    def test_critical_periods_can_be_disabled(self):
        """Test that critical periods can be disabled in config."""
        config = StageConfig(
            enable_critical_periods=False,
        )
        
        assert config.enable_critical_periods is False

    def test_brain_state_fallback(self):
        """Test fallback to brain.state if set_plasticity_modulator not available."""
        brain = Mock()
        brain.state = Mock()
        del brain.set_plasticity_modulator  # Remove method
        
        trainer = CurriculumTrainer(brain=brain, verbose=False)
        trainer.global_step = 25000
        
        trainer._apply_critical_period_modulation(
            task_name='phoneme_task',
            domains=['phonology'],
            age=25000,
        )
        
        # Should set via state instead
        assert brain.state.plasticity_modulator == pytest.approx(1.2, abs=0.01)


class TestCriticalPeriodWindows:
    """Test critical period window configurations."""

    def test_phonology_window(self):
        """Test phonology critical period window."""
        gating = CriticalPeriodGating()
        
        # Peak phase (step 25k within 0-50k)
        status = gating.get_window_status('phonology', 25000)
        assert status['phase'] == 'peak'
        assert status['multiplier'] == pytest.approx(1.2, abs=0.01)
        
        # Late phase (step 100k after 50k)
        status = gating.get_window_status('phonology', 100000)
        assert status['phase'] == 'late'
        assert status['multiplier'] < 1.0

    def test_grammar_window(self):
        """Test grammar critical period window."""
        gating = CriticalPeriodGating()
        
        # Early phase (before 25k)
        status = gating.get_window_status('grammar', 10000)
        assert status['phase'] == 'early'
        assert status['multiplier'] == pytest.approx(0.5, abs=0.01)
        
        # Peak phase (within 25k-150k)
        status = gating.get_window_status('grammar', 100000)
        assert status['phase'] == 'peak'
        assert status['multiplier'] == pytest.approx(1.2, abs=0.01)

    def test_motor_window_higher_peak(self):
        """Test motor has higher peak multiplier."""
        gating = CriticalPeriodGating()
        
        status = gating.get_window_status('motor', 30000)
        assert status['phase'] == 'peak'
        assert status['multiplier'] == pytest.approx(1.25, abs=0.01)  # Higher than others

    def test_semantics_never_closes(self):
        """Test semantics window has extended duration."""
        gating = CriticalPeriodGating()
        
        # Should still be in peak even at 200k
        status = gating.get_window_status('semantics', 200000)
        assert status['phase'] == 'peak'
        assert status['multiplier'] > 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
