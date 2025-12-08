"""
Tests for Critical Period Gating.

Tests the time-windowed plasticity modulation system that implements
biologically-inspired critical periods for different learning domains.
"""

import pytest
import math
from thalia.learning.critical_periods import (
    CriticalPeriodGating,
    CriticalPeriodConfig,
    CriticalPeriodWindow,
)


class TestCriticalPeriodWindow:
    """Test CriticalPeriodWindow dataclass."""
    
    def test_default_window(self):
        """Test default window parameters."""
        window = CriticalPeriodWindow(start_step=0, end_step=50000)
        
        assert window.start_step == 0
        assert window.end_step == 50000
        assert window.peak_multiplier == 1.2
        assert window.early_multiplier == 0.5
        assert window.late_floor == 0.2
        assert window.decay_rate == 20000.0
    
    def test_custom_window(self):
        """Test custom window parameters."""
        window = CriticalPeriodWindow(
            start_step=10000,
            end_step=100000,
            peak_multiplier=1.5,
            early_multiplier=0.3,
            late_floor=0.1,
            decay_rate=15000.0,
        )
        
        assert window.start_step == 10000
        assert window.end_step == 100000
        assert window.peak_multiplier == 1.5
        assert window.early_multiplier == 0.3
        assert window.late_floor == 0.1
        assert window.decay_rate == 15000.0


class TestCriticalPeriodConfig:
    """Test CriticalPeriodConfig defaults."""
    
    def test_default_config(self):
        """Test default configuration has all expected domains."""
        config = CriticalPeriodConfig()
        
        # Check all default domains exist
        assert hasattr(config, 'phonology')
        assert hasattr(config, 'grammar')
        assert hasattr(config, 'semantics')
        assert hasattr(config, 'face_recognition')
        assert hasattr(config, 'motor')
    
    def test_phonology_window(self):
        """Test phonology window matches curriculum (0-50k)."""
        config = CriticalPeriodConfig()
        assert config.phonology.start_step == 0
        assert config.phonology.end_step == 50000
    
    def test_grammar_window(self):
        """Test grammar window matches curriculum (25k-150k)."""
        config = CriticalPeriodConfig()
        assert config.grammar.start_step == 25000
        assert config.grammar.end_step == 150000


class TestCriticalPeriodGating:
    """Test CriticalPeriodGating functionality."""
    
    def test_initialization(self):
        """Test gating module initializes correctly."""
        gating = CriticalPeriodGating()
        
        domains = gating.get_all_domains()
        assert 'phonology' in domains
        assert 'grammar' in domains
        assert 'semantics' in domains
        assert 'motor' in domains
        assert 'face_recognition' in domains
    
    def test_add_custom_domain(self):
        """Test adding custom critical period domain."""
        gating = CriticalPeriodGating()
        
        gating.add_domain(
            'music',
            start=5000,
            end=60000,
            peak_multiplier=1.3,
        )
        
        domains = gating.get_all_domains()
        assert 'music' in domains
        
        # Test it works
        lr = gating.gate_learning(0.001, 'music', 30000)
        assert lr == pytest.approx(0.0013)  # 1.3x multiplier
    
    def test_unknown_domain_raises(self):
        """Test unknown domain raises ValueError."""
        gating = CriticalPeriodGating()
        
        with pytest.raises(ValueError, match="Unknown domain"):
            gating.gate_learning(0.001, 'unknown_domain', 10000)
    
    # Test Phase 1: Early (before window)
    def test_early_phase(self):
        """Test learning rate modulation before critical period."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Phonology window starts at step 0, so can't test "before"
        # Use grammar window (starts at 25k)
        lr_early = gating.gate_learning(base_lr, 'grammar', age=10000)
        
        # Should be early_multiplier (0.5x)
        assert lr_early == pytest.approx(0.0005)
    
    # Test Phase 2: Peak (during window)
    def test_peak_phase(self):
        """Test learning rate modulation during critical period."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Phonology window: 0-50k, test at 25k (middle)
        lr_peak = gating.gate_learning(base_lr, 'phonology', age=25000)
        
        # Should be peak_multiplier (1.2x)
        assert lr_peak == pytest.approx(0.0012)
    
    def test_peak_phase_start(self):
        """Test learning rate at start of critical period."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Grammar window starts at 25k
        lr_start = gating.gate_learning(base_lr, 'grammar', age=25000)
        
        # Should be peak_multiplier (1.2x)
        assert lr_start == pytest.approx(0.0012)
    
    def test_peak_phase_end(self):
        """Test learning rate at end of critical period."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Phonology window ends at 50k
        lr_end = gating.gate_learning(base_lr, 'phonology', age=50000)
        
        # Should still be peak_multiplier (1.2x) at boundary
        assert lr_end == pytest.approx(0.0012)
    
    # Test Phase 3: Late (after window)
    def test_late_phase_immediate(self):
        """Test learning rate just after critical period closes."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Phonology window ends at 50k, test at 50001
        lr_late = gating.gate_learning(base_lr, 'phonology', age=50001)
        
        # Should be declining from peak (1.2) toward floor (0.2)
        # At step 1 past window, decay_factor ≈ 1.0 (almost peak)
        assert lr_late < 0.0012  # Below peak
        assert lr_late > 0.0002  # Above floor
    
    def test_late_phase_far_past(self):
        """Test learning rate far after critical period closes."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Phonology window ends at 50k, test at 150k (100k past)
        lr_late = gating.gate_learning(base_lr, 'phonology', age=150000)
        
        # Should be near floor (0.2x)
        # With decay_rate=20k, at 100k steps past: exp(100k/20k) = exp(5) → large
        # decay_factor ≈ 0, so multiplier ≈ late_floor
        assert lr_late == pytest.approx(0.0002, abs=0.00005)  # Near floor
    
    def test_sigmoid_decay_curve(self):
        """Test sigmoid decay follows expected curve."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Phonology window ends at 50k
        # Test decay at several points
        ages = [50000, 60000, 70000, 90000, 120000]
        lrs = [gating.gate_learning(base_lr, 'phonology', age) for age in ages]
        
        # Learning rates should be monotonically decreasing
        for i in range(len(lrs) - 1):
            assert lrs[i] > lrs[i + 1], f"LR should decrease: {lrs[i]} > {lrs[i+1]}"
        
        # First should be close to peak
        assert lrs[0] == pytest.approx(0.0012)  # At boundary
        
        # Last should be close to floor
        assert lrs[-1] < 0.0003  # Near floor
    
    def test_is_in_peak(self):
        """Test is_in_peak() method."""
        gating = CriticalPeriodGating()
        
        # Phonology window: 0-50k
        assert gating.is_in_peak('phonology', 0) is True
        assert gating.is_in_peak('phonology', 25000) is True
        assert gating.is_in_peak('phonology', 50000) is True
        assert gating.is_in_peak('phonology', 50001) is False
        
        # Grammar window: 25k-150k
        assert gating.is_in_peak('grammar', 10000) is False
        assert gating.is_in_peak('grammar', 25000) is True
        assert gating.is_in_peak('grammar', 100000) is True
        assert gating.is_in_peak('grammar', 150001) is False
    
    def test_get_optimal_age(self):
        """Test get_optimal_age() method."""
        gating = CriticalPeriodGating()
        
        start, end = gating.get_optimal_age('phonology')
        assert start == 0
        assert end == 50000
        
        start, end = gating.get_optimal_age('grammar')
        assert start == 25000
        assert end == 150000
    
    def test_get_window_status(self):
        """Test get_window_status() method."""
        gating = CriticalPeriodGating()
        
        # Test early phase
        status = gating.get_window_status('grammar', 10000)
        assert status['phase'] == 'early'
        assert status['multiplier'] == pytest.approx(0.5)
        assert status['progress'] == 0.0
        assert status['steps_remaining'] == 140000  # 150k - 10k
        
        # Test peak phase
        status = gating.get_window_status('phonology', 25000)
        assert status['phase'] == 'peak'
        assert status['multiplier'] == pytest.approx(1.2)
        assert status['progress'] == 0.5  # Halfway through (0-50k)
        assert status['steps_remaining'] == 25000  # 50k - 25k
        
        # Test late phase
        status = gating.get_window_status('phonology', 100000)
        assert status['phase'] == 'late'
        assert status['multiplier'] < 0.5  # Declining
        assert status['progress'] == 1.0
        assert status['steps_remaining'] is None
    
    def test_multiple_domains_independent(self):
        """Test multiple domains can have independent windows."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # At age 30k:
        # - Phonology (0-50k): peak (1.2x)
        # - Grammar (25k-150k): peak (1.2x)
        # - Semantics (50k-300k): early (0.6x)
        
        age = 30000
        phon_lr = gating.gate_learning(base_lr, 'phonology', age)
        gram_lr = gating.gate_learning(base_lr, 'grammar', age)
        sem_lr = gating.gate_learning(base_lr, 'semantics', age)
        
        assert phon_lr == pytest.approx(0.0012)  # Peak
        assert gram_lr == pytest.approx(0.0012)  # Peak
        assert sem_lr == pytest.approx(0.0006)  # Early
    
    def test_overlapping_windows(self):
        """Test overlapping critical periods work correctly."""
        gating = CriticalPeriodGating()
        
        # At age 75k:
        # - Phonology (0-50k): late (declining)
        # - Grammar (25k-150k): peak (1.2x)
        # - Semantics (50k-300k): peak (1.15x)
        
        age = 75000
        
        phon_status = gating.get_window_status('phonology', age)
        gram_status = gating.get_window_status('grammar', age)
        sem_status = gating.get_window_status('semantics', age)
        
        assert phon_status['phase'] == 'late'
        assert gram_status['phase'] == 'peak'
        assert sem_status['phase'] == 'peak'
    
    def test_motor_high_peak(self):
        """Test motor skills have higher peak multiplier."""
        gating = CriticalPeriodGating()
        
        # Motor has peak_multiplier=1.25 (higher than others)
        motor_lr = gating.gate_learning(0.001, 'motor', 25000)
        phon_lr = gating.gate_learning(0.001, 'phonology', 25000)
        
        assert motor_lr == pytest.approx(0.00125)  # 1.25x
        assert phon_lr == pytest.approx(0.0012)   # 1.2x
        assert motor_lr > phon_lr
    
    def test_semantics_higher_floor(self):
        """Test semantics has higher late_floor (easier to learn late)."""
        config = CriticalPeriodConfig()
        
        # Semantics floor: 0.3
        # Phonology floor: 0.2
        assert config.semantics.late_floor == 0.3
        assert config.phonology.late_floor == 0.2
        
        gating = CriticalPeriodGating(config)
        
        # Far past window (400k)
        sem_lr = gating.gate_learning(0.001, 'semantics', 400000)
        phon_lr = gating.gate_learning(0.001, 'phonology', 400000)
        
        # Both near floor, but semantics higher
        assert sem_lr > phon_lr
        assert sem_lr == pytest.approx(0.0003, abs=0.00002)
        assert phon_lr == pytest.approx(0.0002, abs=0.00002)


class TestCriticalPeriodIntegration:
    """Test critical periods in realistic training scenarios."""
    
    def test_stage_0_phonology_training(self):
        """Test phonology learning during Stage 0 (Week 4-12, 0-50k steps)."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Stage 0 spans phonology critical period (0-50k)
        # Should get peak learning throughout
        ages = [10000, 20000, 30000, 40000, 50000]
        lrs = [gating.gate_learning(base_lr, 'phonology', age) for age in ages]
        
        # All should be at peak (1.2x)
        for lr in lrs:
            assert lr == pytest.approx(0.0012)
    
    def test_stage_1_grammar_opening(self):
        """Test grammar window opening during Stage 1 (Week 12-20, 50k-100k steps)."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Grammar window opens at 25k, Stage 1 starts at 50k
        # Should be in peak throughout Stage 1
        ages = [50000, 60000, 75000, 90000, 100000]
        lrs = [gating.gate_learning(base_lr, 'grammar', age) for age in ages]
        
        # All should be at peak (1.2x)
        for lr in lrs:
            assert lr == pytest.approx(0.0012)
    
    def test_late_phonology_learning(self):
        """Test phonology learning in later stages (after critical period)."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Stage 3 (Week 28-46, ~150k-230k steps)
        # Phonology window closed at 50k
        age = 200000
        lr = gating.gate_learning(base_lr, 'phonology', age)
        
        # Should be near floor (hard to learn)
        assert lr < 0.0003
        assert lr > 0.0001
    
    def test_curriculum_realistic_progression(self):
        """Test realistic curriculum progression through stages."""
        gating = CriticalPeriodGating()
        base_lr = 0.001
        
        # Simulate training progression
        stages = [
            ('Stage 0', 25000, ['phonology']),
            ('Stage 1', 75000, ['grammar', 'phonology']),
            ('Stage 2', 125000, ['grammar', 'semantics']),
            ('Stage 3', 175000, ['grammar', 'semantics']),
        ]
        
        for stage_name, age, domains in stages:
            for domain in domains:
                lr = gating.gate_learning(base_lr, domain, age)
                status = gating.get_window_status(domain, age)
                
                print(f"{stage_name} ({age}): {domain} = {lr:.6f} ({status['phase']})")
        
        # Basic sanity checks
        # Stage 0: phonology peak
        assert gating.is_in_peak('phonology', 25000)
        
        # Stage 1: grammar peak, phonology closing
        assert gating.is_in_peak('grammar', 75000)
        assert not gating.is_in_peak('phonology', 75000)
        
        # Stage 2: grammar peak, semantics peak
        assert gating.is_in_peak('grammar', 125000)
        assert gating.is_in_peak('semantics', 125000)
