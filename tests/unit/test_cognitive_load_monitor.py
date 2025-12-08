"""
Unit tests for CognitiveLoadMonitor.

Tests the cognitive load monitoring system that prevents mechanism
overload during curriculum stage transitions.
"""

import pytest
from thalia.training import (
    CognitiveLoadMonitor,
    MechanismPriority,
    CurriculumStage,
)


class TestCognitiveLoadMonitor:
    """Test CognitiveLoadMonitor functionality."""

    def test_initialization(self):
        """Test monitor initialization with default and custom thresholds."""
        # Default threshold
        monitor = CognitiveLoadMonitor()
        assert monitor.load_threshold == 0.9
        assert monitor.calculate_load() == 0.0
        assert not monitor.is_overloaded()

        # Custom threshold
        monitor = CognitiveLoadMonitor(load_threshold=0.8)
        assert monitor.load_threshold == 0.8

    def test_invalid_threshold(self):
        """Test that invalid thresholds raise errors."""
        with pytest.raises(ValueError):
            CognitiveLoadMonitor(load_threshold=0.0)

        with pytest.raises(ValueError):
            CognitiveLoadMonitor(load_threshold=1.5)

        with pytest.raises(ValueError):
            CognitiveLoadMonitor(load_threshold=-0.5)

    def test_add_mechanism(self):
        """Test adding mechanisms and calculating load."""
        monitor = CognitiveLoadMonitor()

        # Add single mechanism
        monitor.add_mechanism('visual', cost=0.2, priority=MechanismPriority.CRITICAL)
        assert monitor.calculate_load() == 0.2
        assert len(monitor.active_mechanisms) == 1

        # Add another mechanism
        monitor.add_mechanism('working_memory', cost=0.3, priority=MechanismPriority.HIGH)
        assert monitor.calculate_load() == 0.5
        assert len(monitor.active_mechanisms) == 2

    def test_invalid_mechanism_cost(self):
        """Test that invalid mechanism costs raise errors."""
        monitor = CognitiveLoadMonitor()

        with pytest.raises(ValueError):
            monitor.add_mechanism('invalid', cost=-0.1)

        with pytest.raises(ValueError):
            monitor.add_mechanism('invalid', cost=1.5)

    def test_remove_mechanism(self):
        """Test removing mechanisms."""
        monitor = CognitiveLoadMonitor()
        monitor.add_mechanism('visual', cost=0.2)
        monitor.add_mechanism('working_memory', cost=0.3)

        # Remove existing mechanism
        assert monitor.remove_mechanism('visual')
        assert monitor.calculate_load() == 0.3
        assert len(monitor.active_mechanisms) == 1

        # Try to remove non-existent mechanism
        assert not monitor.remove_mechanism('nonexistent')

    def test_deactivate_mechanism(self):
        """Test deactivating mechanisms."""
        monitor = CognitiveLoadMonitor()
        monitor.add_mechanism('visual', cost=0.2, can_deactivate=True)
        monitor.add_mechanism('auditory', cost=0.1, can_deactivate=False)

        # Deactivate deactivatable mechanism
        assert monitor.deactivate_mechanism('visual')
        assert monitor.calculate_load() == 0.1
        assert len(monitor.active_mechanisms) == 1
        assert len(monitor.deactivated_mechanisms) == 1

        # Try to deactivate non-deactivatable mechanism
        assert not monitor.deactivate_mechanism('auditory')
        assert monitor.calculate_load() == 0.1

    def test_reactivate_mechanism(self):
        """Test reactivating deactivated mechanisms."""
        monitor = CognitiveLoadMonitor()
        monitor.add_mechanism('visual', cost=0.2)
        monitor.deactivate_mechanism('visual')

        assert monitor.calculate_load() == 0.0
        assert len(monitor.deactivated_mechanisms) == 1

        # Reactivate
        assert monitor.reactivate_mechanism('visual')
        assert monitor.calculate_load() == 0.2
        assert len(monitor.active_mechanisms) == 1
        assert len(monitor.deactivated_mechanisms) == 0

        # Try to reactivate non-existent mechanism
        assert not monitor.reactivate_mechanism('nonexistent')

    def test_overload_detection(self):
        """Test overload detection."""
        monitor = CognitiveLoadMonitor(load_threshold=0.9)

        # Not overloaded
        monitor.add_mechanism('m1', cost=0.5)
        assert not monitor.is_overloaded()
        assert abs(monitor.get_headroom() - 0.4) < 1e-10

        # Exactly at threshold
        monitor.add_mechanism('m2', cost=0.4)
        assert not monitor.is_overloaded()
        assert abs(monitor.get_headroom() - 0.0) < 1e-10

        # Overloaded
        monitor.add_mechanism('m3', cost=0.2)
        assert monitor.is_overloaded()
        assert abs(monitor.get_headroom() - (-0.2)) < 1e-10

    def test_suggest_deactivation(self):
        """Test deactivation suggestions based on priority."""
        monitor = CognitiveLoadMonitor(load_threshold=0.9)

        # Not overloaded - no suggestion
        monitor.add_mechanism('m1', cost=0.5)
        assert monitor.suggest_deactivation() is None

        # Make it overloaded with various priorities
        monitor.add_mechanism('critical', cost=0.2, priority=MechanismPriority.CRITICAL, can_deactivate=False)
        monitor.add_mechanism('high', cost=0.15, priority=MechanismPriority.HIGH)
        monitor.add_mechanism('medium', cost=0.10, priority=MechanismPriority.MEDIUM)
        monitor.add_mechanism('low', cost=0.05, priority=MechanismPriority.LOW)

        # Should suggest LOW priority first (deactivate least important first)
        suggestion = monitor.suggest_deactivation()
        assert suggestion == 'low'

    def test_suggest_deactivation_priority_order(self):
        """Test that suggestions follow priority order."""
        monitor = CognitiveLoadMonitor(load_threshold=0.4)  # Lower threshold to stay overloaded longer
        
        # Add mechanisms with different priorities
        monitor.add_mechanism('critical', cost=0.2, priority=MechanismPriority.CRITICAL, can_deactivate=False)
        monitor.add_mechanism('high', cost=0.15, priority=MechanismPriority.HIGH)
        monitor.add_mechanism('medium', cost=0.12, priority=MechanismPriority.MEDIUM)
        monitor.add_mechanism('low', cost=0.10, priority=MechanismPriority.LOW)
        
        # Total load: 0.57, threshold: 0.4 -> overloaded
        assert monitor.is_overloaded()
        suggestion = monitor.suggest_deactivation()
        assert suggestion == 'low'
        
        # Deactivate LOW: 0.57 - 0.10 = 0.47 -> still overloaded
        monitor.deactivate_mechanism('low')
        assert monitor.is_overloaded()
        suggestion = monitor.suggest_deactivation()
        assert suggestion == 'medium'
        
        # Deactivate MEDIUM: 0.47 - 0.12 = 0.35 -> not overloaded anymore
        monitor.deactivate_mechanism('medium')
        assert not monitor.is_overloaded()
    
    def test_suggest_multiple_deactivations(self):
        """Test suggesting multiple deactivations to reach target."""
        monitor = CognitiveLoadMonitor(load_threshold=0.9)

        # Create overloaded system
        monitor.add_mechanism('critical', cost=0.3, priority=MechanismPriority.CRITICAL, can_deactivate=False)
        monitor.add_mechanism('high1', cost=0.25, priority=MechanismPriority.HIGH)
        monitor.add_mechanism('high2', cost=0.20, priority=MechanismPriority.HIGH)
        monitor.add_mechanism('medium', cost=0.15, priority=MechanismPriority.MEDIUM)
        monitor.add_mechanism('low', cost=0.10, priority=MechanismPriority.LOW)

        # Get suggestions to reach 0.6 load
        suggestions = monitor.suggest_multiple_deactivations(target_load=0.6)

        # Should suggest LOW first, then MEDIUM (total 0.25 reduction)
        assert len(suggestions) >= 1
        assert suggestions[0] == 'low'

    def test_load_breakdown_by_priority(self):
        """Test load breakdown by priority level."""
        monitor = CognitiveLoadMonitor()

        monitor.add_mechanism('c1', cost=0.1, priority=MechanismPriority.CRITICAL)
        monitor.add_mechanism('c2', cost=0.15, priority=MechanismPriority.CRITICAL)
        monitor.add_mechanism('h1', cost=0.2, priority=MechanismPriority.HIGH)
        monitor.add_mechanism('m1', cost=0.12, priority=MechanismPriority.MEDIUM)

        breakdown = monitor.get_load_by_priority()
        assert breakdown[MechanismPriority.CRITICAL] == 0.25
        assert breakdown[MechanismPriority.HIGH] == 0.2
        assert breakdown[MechanismPriority.MEDIUM] == 0.12
        assert breakdown[MechanismPriority.LOW] == 0.0

    def test_load_breakdown_by_stage(self):
        """Test load breakdown by curriculum stage."""
        monitor = CognitiveLoadMonitor()

        monitor.add_mechanism('s1', cost=0.2, stage_introduced=CurriculumStage.SENSORIMOTOR)
        monitor.add_mechanism('s2', cost=0.15, stage_introduced=CurriculumStage.SENSORIMOTOR)
        monitor.add_mechanism('p1', cost=0.25, stage_introduced=CurriculumStage.PHONOLOGY)
        monitor.add_mechanism('t1', cost=0.10, stage_introduced=CurriculumStage.TODDLER)

        breakdown = monitor.get_load_by_stage()
        assert breakdown[CurriculumStage.SENSORIMOTOR] == 0.35
        assert breakdown[CurriculumStage.PHONOLOGY] == 0.25
        assert breakdown[CurriculumStage.TODDLER] == 0.10

    def test_load_statistics(self):
        """Test load statistics tracking."""
        monitor = CognitiveLoadMonitor()

        # Add mechanisms to change load over time
        monitor.add_mechanism('m1', cost=0.3)
        stats1 = monitor.get_load_statistics()
        assert stats1['current'] == 0.3

        monitor.add_mechanism('m2', cost=0.4)
        stats2 = monitor.get_load_statistics()
        assert stats2['current'] == 0.7
        assert stats2['min'] == 0.3
        assert stats2['max'] == 0.7

        monitor.remove_mechanism('m2')
        stats3 = monitor.get_load_statistics()
        assert stats3['current'] == 0.3
        assert stats3['min'] == 0.3
        assert stats3['max'] == 0.7

    def test_status_report(self):
        """Test status report generation."""
        monitor = CognitiveLoadMonitor(load_threshold=0.9)
        monitor.add_mechanism('visual', cost=0.5, priority=MechanismPriority.CRITICAL, can_deactivate=False)
        monitor.add_mechanism('working_memory', cost=0.6, priority=MechanismPriority.HIGH)

        report = monitor.get_status_report()

        # Check that report contains key information
        assert 'Current Load' in report
        assert 'Headroom' in report
        assert 'OVERLOADED' in report  # Should be overloaded (1.1 > 0.9)
        assert 'visual' in report
        assert 'working_memory' in report
        assert 'Suggestion' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
