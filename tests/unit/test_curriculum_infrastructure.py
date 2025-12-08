"""
Unit tests for curriculum training infrastructure.

Tests all 6 curriculum components:
1. InterleavedCurriculumSampler - Multinomial task sampling
2. SpacedRepetitionScheduler - Leitner expanding intervals
3. TestingPhaseProtocol - Retrieval practice
4. ProductiveFailurePhase - Intentional difficulty
5. CurriculumDifficultyCalibrator - Zone of proximal development
6. StageTransitionProtocol - Gradual ramps
"""

import pytest
from collections import Counter

from thalia.training.curriculum import (
    InterleavedCurriculumSampler,
    InterleavedCurriculumSamplerConfig,
    SpacedRepetitionScheduler,
    SpacedRepetitionSchedulerConfig,
    TestingPhaseProtocol,
    TestingPhaseConfig,
    ProductiveFailurePhase,
    ProductiveFailureConfig,
    CurriculumDifficultyCalibrator,
    DifficultyCalibratorConfig,
    StageTransitionProtocol,
    StageTransitionConfig,
    TransitionWeekConfig,
)


# ============================================================================
# 1. Interleaved Curriculum Sampler Tests
# ============================================================================

class TestInterleavedCurriculumSampler:
    """Test multinomial task sampling for interleaved practice."""
    
    def test_sampler_basic(self):
        """Test basic sampling functionality."""
        sampler = InterleavedCurriculumSampler()
        weights = {0: 0.05, 1: 0.10, 2: 0.15, 4: 0.70}
        
        # Sample should return one of the stages
        stage = sampler.sample_next_task(weights)
        assert stage in [0, 1, 2, 4]
    
    def test_sampler_distribution(self):
        """Test that sampling follows expected distribution."""
        sampler = InterleavedCurriculumSampler(
            InterleavedCurriculumSamplerConfig(seed=42)
        )
        weights = {0: 0.05, 1: 0.10, 2: 0.15, 4: 0.70}
        
        # Sample 10000 times
        samples = [sampler.sample_next_task(weights) for _ in range(10000)]
        counts = Counter(samples)
        
        # Check distribution (within 2% tolerance)
        assert 0.03 <= counts[0] / 10000 <= 0.07  # ~5%
        assert 0.08 <= counts[1] / 10000 <= 0.12  # ~10%
        assert 0.13 <= counts[2] / 10000 <= 0.17  # ~15%
        assert 0.68 <= counts[4] / 10000 <= 0.72  # ~70%
    
    def test_sampler_interleaving(self):
        """Test that samples are interleaved (not blocked)."""
        sampler = InterleavedCurriculumSampler(
            InterleavedCurriculumSamplerConfig(seed=123)
        )
        weights = {0: 0.10, 1: 0.90}
        
        # Sample 100 times
        samples = [sampler.sample_next_task(weights) for _ in range(100)]
        
        # Check that we DON'T get long runs (blocked practice)
        max_run_length = 0
        current_run = 1
        
        for i in range(1, len(samples)):
            if samples[i] == samples[i-1]:
                current_run += 1
            else:
                max_run_length = max(max_run_length, current_run)
                current_run = 1
        
        # With 90% weight on stage 1, we expect some runs but not the full 90
        # Max run should be < 30 (significantly less than 90)
        assert max_run_length < 30
    
    def test_sampler_empty_weights(self):
        """Test error handling for empty weights."""
        sampler = InterleavedCurriculumSampler()
        
        with pytest.raises(ValueError, match="stage_weights cannot be empty"):
            sampler.sample_next_task({})
    
    def test_sampler_invalid_weights(self):
        """Test error handling for weights not summing to 1.0."""
        sampler = InterleavedCurriculumSampler()
        
        with pytest.raises(ValueError, match="Weights should sum to ~1.0"):
            sampler.sample_next_task({0: 0.5, 1: 0.3})  # Sums to 0.8
    
    def test_sampler_reproducibility(self):
        """Test that seed makes sampling reproducible."""
        # Create sampler 1 and get samples
        sampler1 = InterleavedCurriculumSampler(
            InterleavedCurriculumSamplerConfig(seed=42)
        )
        weights = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        samples1 = [sampler1.sample_next_task(weights) for _ in range(100)]
        
        # Create sampler 2 (reseed) and get samples
        sampler2 = InterleavedCurriculumSampler(
            InterleavedCurriculumSamplerConfig(seed=42)
        )
        samples2 = [sampler2.sample_next_task(weights) for _ in range(100)]
        
        # Should produce identical sequences
        assert samples1 == samples2


# ============================================================================
# 2. Spaced Repetition Scheduler Tests
# ============================================================================

class TestSpacedRepetitionScheduler:
    """Test Leitner-style spaced repetition scheduling."""
    
    def test_scheduler_basic_interval(self):
        """Test basic interval calculation."""
        scheduler = SpacedRepetitionScheduler()
        
        # Stage 1, moderate performance, no reviews yet
        interval = scheduler.calculate_interval(
            stage=1,
            performance=0.88,
            review_count=0,
        )
        
        # Should use base interval for stage 1 (25000)
        assert interval == 25000
    
    def test_scheduler_high_performance_expansion(self):
        """Test that high performance expands intervals."""
        scheduler = SpacedRepetitionScheduler()
        
        # High performance (>92%) after 2 reviews
        interval = scheduler.calculate_interval(
            stage=1,
            performance=0.95,
            review_count=2,
        )
        
        # Should expand: 25000 * 1.5^2 = 56250
        assert interval == 56250
    
    def test_scheduler_low_performance_reset(self):
        """Test that low performance resets interval."""
        scheduler = SpacedRepetitionScheduler()
        
        # Low performance (<85%)
        interval = scheduler.calculate_interval(
            stage=1,
            performance=0.78,
            review_count=5,
        )
        
        # Should reset to short interval (10000)
        assert interval == 10000
    
    def test_scheduler_should_review(self):
        """Test review scheduling logic."""
        scheduler = SpacedRepetitionScheduler()
        
        # Last reviewed at step 10000, now at 40000
        # Stage 1 base interval: 25000
        # 40000 - 10000 = 30000 > 25000 → Should review
        should_review, interval = scheduler.should_review_stage(
            stage=1,
            last_review_step=10000,
            current_step=40000,
            performance=0.88,
            review_count=0,
        )
        
        assert should_review is True
        assert interval == 25000
    
    def test_scheduler_not_yet_due(self):
        """Test when review is not yet due."""
        scheduler = SpacedRepetitionScheduler()
        
        # Last reviewed at step 10000, now at 20000
        # Stage 1 interval: 25000
        # 20000 - 10000 = 10000 < 25000 → Not due
        should_review, interval = scheduler.should_review_stage(
            stage=1,
            last_review_step=10000,
            current_step=20000,
            performance=0.88,
            review_count=0,
        )
        
        assert should_review is False
        assert interval == 25000
    
    def test_scheduler_stage_specific_intervals(self):
        """Test that different stages have different base intervals."""
        scheduler = SpacedRepetitionScheduler()
        
        interval_0 = scheduler.calculate_interval(0, 0.88, 0)
        interval_1 = scheduler.calculate_interval(1, 0.88, 0)
        interval_2 = scheduler.calculate_interval(2, 0.88, 0)
        
        # Stage 0 should have longest (foundation preservation)
        assert interval_0 == 50000
        assert interval_1 == 25000
        assert interval_2 == 15000
    
    def test_scheduler_review_schedule(self):
        """Test getting review distribution for multiple stages."""
        scheduler = SpacedRepetitionScheduler()
        
        stages = [0, 1, 2]
        stage_history = {0: 0, 1: 50000, 2: 80000}
        current_step = 100000
        stage_performance = {0: 0.95, 1: 0.88, 2: 0.82}
        stage_review_counts = {0: 3, 1: 1, 2: 0}
        
        weights = scheduler.get_review_schedule(
            stages=stages,
            stage_history=stage_history,
            current_step=current_step,
            stage_performance=stage_performance,
            stage_review_counts=stage_review_counts,
        )
        
        # All stages should be in distribution
        assert set(weights.keys()) == {0, 1, 2}
        
        # Weights should sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01


# ============================================================================
# 3. Testing Phase Protocol Tests
# ============================================================================

class TestTestingPhaseProtocol:
    """Test retrieval practice (testing effect) protocol."""
    
    def test_protocol_test_frequency(self):
        """Test that testing occurs at expected frequency."""
        protocol = TestingPhaseProtocol(
            TestingPhaseConfig(test_frequency=0.15)
        )
        
        # Count test steps in first 1000 steps
        test_steps = [
            step for step in range(1000)
            if protocol.should_test(step)
        ]
        
        # Should be ~15% (within 5% tolerance)
        test_ratio = len(test_steps) / 1000
        assert 0.10 <= test_ratio <= 0.20
    
    def test_protocol_deterministic(self):
        """Test that testing schedule is deterministic."""
        protocol1 = TestingPhaseProtocol(
            TestingPhaseConfig(test_frequency=0.20)
        )
        protocol2 = TestingPhaseProtocol(
            TestingPhaseConfig(test_frequency=0.20)
        )
        
        steps1 = [protocol1.should_test(s) for s in range(100)]
        steps2 = [protocol2.should_test(s) for s in range(100)]
        
        assert steps1 == steps2
    
    def test_protocol_log_results(self):
        """Test logging of test results."""
        protocol = TestingPhaseProtocol()
        
        protocol.log_test_result(100, True)
        protocol.log_test_result(200, False)
        protocol.log_test_result(300, True)
        
        accuracy = protocol.get_test_accuracy()
        assert accuracy == 2/3  # 2 correct out of 3
    
    def test_protocol_accuracy_window(self):
        """Test accuracy calculation with window."""
        protocol = TestingPhaseProtocol()
        
        # Log old results
        for step in range(0, 1000, 100):
            protocol.log_test_result(step, False)
        
        # Log recent results (all correct)
        for step in range(9000, 10000, 100):
            protocol.log_test_result(step, True)
        
        # Recent accuracy should be 100%
        recent_acc = protocol.get_test_accuracy(last_n_steps=1500)
        assert recent_acc == 1.0
        
        # Overall accuracy should be ~50%
        overall_acc = protocol.get_test_accuracy()
        assert 0.45 <= overall_acc <= 0.55
    
    def test_protocol_empty_history(self):
        """Test behavior with no test history."""
        protocol = TestingPhaseProtocol()
        
        accuracy = protocol.get_test_accuracy()
        assert accuracy == 0.0


# ============================================================================
# 4. Productive Failure Phase Tests
# ============================================================================

class TestProductiveFailurePhase:
    """Test intentional difficulty (productive failure) protocol."""
    
    def test_failure_phase_detection(self):
        """Test detection of failure phase."""
        pf = ProductiveFailurePhase(
            ProductiveFailureConfig(duration_steps=5000)
        )
        
        stage_start = 100000
        
        # Within failure phase
        assert pf.is_in_failure_phase(stage_start, 102000) is True
        
        # After failure phase
        assert pf.is_in_failure_phase(stage_start, 106000) is False
    
    def test_failure_phase_exact_boundary(self):
        """Test exact boundary of failure phase."""
        pf = ProductiveFailurePhase(
            ProductiveFailureConfig(duration_steps=5000)
        )
        
        stage_start = 0
        
        # Last step of failure phase
        assert pf.is_in_failure_phase(stage_start, 4999) is True
        
        # First step after failure phase
        assert pf.is_in_failure_phase(stage_start, 5000) is False
    
    def test_failure_phase_target_rate(self):
        """Test target success rate retrieval."""
        pf = ProductiveFailurePhase(
            ProductiveFailureConfig(target_success_rate=0.20)
        )
        
        assert pf.get_target_success_rate() == 0.20


# ============================================================================
# 5. Curriculum Difficulty Calibrator Tests
# ============================================================================

class TestCurriculumDifficultyCalibrator:
    """Test Zone of Proximal Development maintenance."""
    
    def test_calibrator_too_easy(self):
        """Test difficulty increase when too easy."""
        calibrator = CurriculumDifficultyCalibrator(
            DifficultyCalibratorConfig(
                too_easy_threshold=0.90,
                adjustment_rate=0.05,
            )
        )
        
        # Success rate 92% (too easy)
        new_difficulty = calibrator.calibrate(
            success_rate=0.92,
            current_difficulty=0.5,
        )
        
        # Should increase by 0.05
        assert new_difficulty == 0.55
    
    def test_calibrator_too_hard(self):
        """Test difficulty decrease when too hard."""
        calibrator = CurriculumDifficultyCalibrator(
            DifficultyCalibratorConfig(
                too_hard_threshold=0.60,
                adjustment_rate=0.05,
            )
        )
        
        # Success rate 58% (too hard)
        new_difficulty = calibrator.calibrate(
            success_rate=0.58,
            current_difficulty=0.7,
        )
        
        # Should decrease by 0.05 (within floating point tolerance)
        assert abs(new_difficulty - 0.65) < 0.001
    
    def test_calibrator_just_right(self):
        """Test no adjustment when in target zone."""
        calibrator = CurriculumDifficultyCalibrator(
            DifficultyCalibratorConfig(
                too_easy_threshold=0.90,
                too_hard_threshold=0.60,
            )
        )
        
        # Success rate 75% (just right)
        new_difficulty = calibrator.calibrate(
            success_rate=0.75,
            current_difficulty=0.6,
        )
        
        # Should maintain
        assert new_difficulty == 0.6
    
    def test_calibrator_boundary_conditions(self):
        """Test that difficulty stays in [0, 1] range."""
        calibrator = CurriculumDifficultyCalibrator(
            DifficultyCalibratorConfig(adjustment_rate=0.1)
        )
        
        # Can't go above 1.0
        new_difficulty = calibrator.calibrate(
            success_rate=0.95,
            current_difficulty=0.98,
        )
        assert new_difficulty == 1.0
        
        # Can't go below 0.0
        new_difficulty = calibrator.calibrate(
            success_rate=0.50,
            current_difficulty=0.02,
        )
        assert new_difficulty == 0.0


# ============================================================================
# 6. Stage Transition Protocol Tests
# ============================================================================

class TestStageTransitionProtocol:
    """Test gradual difficulty ramps during stage transitions."""
    
    def test_transition_week_1_config(self):
        """Test Week 1 configuration (easiest)."""
        protocol = StageTransitionProtocol()
        
        config = protocol.get_transition_config(
            old_stage=1,
            new_stage=2,
            weeks_since_transition=0,
        )
        
        # Week 1: Easy difficulty, high old stage review
        assert config.difficulty == 0.3
        assert config.old_stage_ratio == 0.70
    
    def test_transition_week_2_config(self):
        """Test Week 2 configuration (moderate)."""
        protocol = StageTransitionProtocol()
        
        config = protocol.get_transition_config(
            old_stage=1,
            new_stage=2,
            weeks_since_transition=1,
        )
        
        # Week 2: Moderate difficulty, moderate review
        assert config.difficulty == 0.5
        assert config.old_stage_ratio == 0.50
    
    def test_transition_week_3_config(self):
        """Test Week 3 configuration (harder)."""
        protocol = StageTransitionProtocol()
        
        config = protocol.get_transition_config(
            old_stage=1,
            new_stage=2,
            weeks_since_transition=2,
        )
        
        # Week 3: Harder difficulty, normal review
        assert config.difficulty == 0.7
        assert config.old_stage_ratio == 0.30
    
    def test_transition_week_4_config(self):
        """Test Week 4+ configuration (full difficulty)."""
        protocol = StageTransitionProtocol()
        
        config = protocol.get_transition_config(
            old_stage=1,
            new_stage=2,
            weeks_since_transition=3,
        )
        
        # Week 4+: Full difficulty, normal review
        assert config.difficulty == 1.0
        assert config.old_stage_ratio == 0.30
    
    def test_transition_completion_check(self):
        """Test transition completion detection."""
        protocol = StageTransitionProtocol()
        
        assert protocol.is_transition_complete(3) is False
        assert protocol.is_transition_complete(4) is True
        assert protocol.is_transition_complete(10) is True
    
    def test_transition_difficulty_progression(self):
        """Test that difficulty increases monotonically."""
        protocol = StageTransitionProtocol()
        
        difficulties = []
        for week in range(5):
            config = protocol.get_transition_config(0, 1, week)
            difficulties.append(config.difficulty)
        
        # Should be monotonically increasing
        assert difficulties == sorted(difficulties)
        
        # Should span from 0.3 to 1.0
        assert difficulties[0] == 0.3
        assert difficulties[-1] == 1.0
    
    def test_transition_review_ratio_decreases(self):
        """Test that old stage review ratio decreases over time."""
        protocol = StageTransitionProtocol()
        
        ratios = []
        for week in range(4):
            config = protocol.get_transition_config(0, 1, week)
            ratios.append(config.old_stage_ratio)
        
        # Should decrease (more focus on new stage)
        assert ratios[0] > ratios[1] > ratios[2]
        
        # Week 3+ should stabilize at 0.30
        assert ratios[2] == ratios[3] == 0.30


# ============================================================================
# Integration Tests
# ============================================================================

class TestCurriculumIntegration:
    """Test realistic curriculum scenarios."""
    
    def test_interleaved_spaced_repetition_combo(self):
        """Test combining interleaved practice with spaced repetition."""
        sampler = InterleavedCurriculumSampler(
            InterleavedCurriculumSamplerConfig(seed=42)
        )
        scheduler = SpacedRepetitionScheduler()
        
        # Get review schedule
        stages = [0, 1, 2]
        stage_history = {0: 0, 1: 50000, 2: 80000}
        current_step = 100000
        stage_performance = {0: 0.95, 1: 0.88, 2: 0.82}
        stage_review_counts = {0: 3, 1: 1, 2: 0}
        
        review_weights = scheduler.get_review_schedule(
            stages=stages,
            stage_history=stage_history,
            current_step=current_step,
            stage_performance=stage_performance,
            stage_review_counts=stage_review_counts,
        )
        
        # Sample from interleaved distribution
        if sum(review_weights.values()) > 0:
            stage = sampler.sample_next_task(review_weights)
            assert stage in stages
    
    def test_productive_failure_with_difficulty_calibration(self):
        """Test productive failure followed by difficulty calibration."""
        pf = ProductiveFailurePhase(
            ProductiveFailureConfig(
                target_success_rate=0.20,
                duration_steps=5000,
            )
        )
        calibrator = CurriculumDifficultyCalibrator()
        
        stage_start = 100000
        
        # During failure phase: target 20% success
        current_step = 102000
        if pf.is_in_failure_phase(stage_start, current_step):
            target_rate = pf.get_target_success_rate()
            assert target_rate == 0.20
        
        # After failure phase: calibrate to 75% success
        current_step = 106000
        if not pf.is_in_failure_phase(stage_start, current_step):
            # Simulate success rate climbing from 20% → 75%
            current_difficulty = 0.9  # Very hard
            success_rate = 0.30  # Still too hard
            
            new_difficulty = calibrator.calibrate(success_rate, current_difficulty)
            assert new_difficulty < current_difficulty  # Should decrease
    
    def test_stage_transition_with_testing(self):
        """Test stage transition with testing protocol."""
        transition = StageTransitionProtocol()
        tester = TestingPhaseProtocol(
            TestingPhaseConfig(test_frequency=0.15)
        )
        
        # Week 1 of transition
        config = transition.get_transition_config(
            old_stage=1,
            new_stage=2,
            weeks_since_transition=0,
        )
        
        # Easy difficulty during transition
        assert config.difficulty == 0.3
        
        # But still do testing (15% of steps)
        test_count = sum(
            1 for step in range(1000)
            if tester.should_test(step)
        )
        assert 100 < test_count < 200  # ~15%
    
    def test_full_curriculum_pipeline(self):
        """Test complete curriculum pipeline over multiple weeks."""
        # Initialize all components
        sampler = InterleavedCurriculumSampler(
            InterleavedCurriculumSamplerConfig(seed=42)
        )
        scheduler = SpacedRepetitionScheduler()
        tester = TestingPhaseProtocol()
        pf = ProductiveFailurePhase()
        calibrator = CurriculumDifficultyCalibrator()
        transition = StageTransitionProtocol()
        
        # Simulate stage transition
        stage_start = 0
        old_stage = 1
        new_stage = 2
        
        weeks_progress = []
        for week in range(5):
            # Get transition config
            week_config = transition.get_transition_config(
                old_stage, new_stage, week
            )
            
            # Check if in productive failure phase
            current_step = stage_start + (week * 7000)  # ~1 week
            in_failure = pf.is_in_failure_phase(stage_start, current_step)
            
            # Get spaced repetition weights
            review_weights = scheduler.get_review_schedule(
                stages=[old_stage, new_stage],
                stage_history={old_stage: 0, new_stage: stage_start},
                current_step=current_step,
                stage_performance={old_stage: 0.95, new_stage: 0.80},
                stage_review_counts={old_stage: 5, new_stage: 0},
            )
            
            weeks_progress.append({
                "week": week,
                "difficulty": week_config.difficulty,
                "old_ratio": week_config.old_stage_ratio,
                "in_failure": in_failure,
                "complete": transition.is_transition_complete(week),
            })
        
        # Verify progression
        assert weeks_progress[0]["difficulty"] < weeks_progress[-1]["difficulty"]
        assert weeks_progress[0]["old_ratio"] > weeks_progress[-1]["old_ratio"]
        assert weeks_progress[-1]["complete"] is True
