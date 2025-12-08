"""
Unit tests for enhanced consolidation system.

Tests 4 components:
1. MemoryPressureDetector - When to trigger consolidation
2. SleepStageController - NREM/REM cycling
3. ConsolidationMetrics - Transfer quality tracking
4. ConsolidationTrigger - High-level orchestration
"""

import pytest

from thalia.memory.consolidation import (
    MemoryPressureDetector,
    MemoryPressureConfig,
    SleepStageController,
    SleepStageConfig,
    SleepStage,
    ConsolidationMetrics,
    ConsolidationTrigger,
    ConsolidationTriggerConfig,
)


# ============================================================================
# 1. Memory Pressure Detector Tests
# ============================================================================

class TestMemoryPressureDetector:
    """Test memory pressure detection logic."""

    def test_detector_low_pressure(self):
        """Test low pressure conditions (normal operation)."""
        detector = MemoryPressureDetector()

        # Normal conditions: low activity, low overlap, good retrieval
        pressure = detector.calculate_pressure(
            hippocampus_activity=0.50,
            pattern_overlap=0.40,
            retrieval_success=0.95,
        )

        # Should have low pressure
        assert pressure < 0.5
        assert not detector.should_trigger_consolidation(pressure)
        assert detector.get_consolidation_urgency(pressure) in ["none", "low"]

    def test_detector_high_pressure(self):
        """Test high pressure conditions (need consolidation)."""
        detector = MemoryPressureDetector(
            MemoryPressureConfig(
                high_activity_threshold=0.70,
                high_overlap_threshold=0.60,
                low_retrieval_threshold=0.90,
                pressure_threshold=0.30,  # Lower threshold for testing
            )
        )

        # High load: high activity, high overlap, poor retrieval
        pressure = detector.calculate_pressure(
            hippocampus_activity=0.90,
            pattern_overlap=0.80,
            retrieval_success=0.78,
        )

        # Should have high pressure
        assert pressure > 0.30
        assert detector.should_trigger_consolidation(pressure)
        # Urgency can be low/moderate depending on exact value
        assert detector.get_consolidation_urgency(pressure) in ["low", "moderate", "high", "critical"]

    def test_detector_activity_component(self):
        """Test activity pressure component."""
        detector = MemoryPressureDetector(
            MemoryPressureConfig(
                high_activity_threshold=0.80,
                activity_weight=1.0,  # Only activity matters
                overlap_weight=0.0,
                retrieval_weight=0.0,
            )
        )

        # Below threshold: no pressure
        pressure_low = detector.calculate_pressure(
            hippocampus_activity=0.70,
            pattern_overlap=0.0,
            retrieval_success=1.0,
        )
        assert pressure_low == 0.0

        # At threshold: no pressure
        pressure_at = detector.calculate_pressure(
            hippocampus_activity=0.80,
            pattern_overlap=0.0,
            retrieval_success=1.0,
        )
        assert pressure_at == 0.0

        # Above threshold: pressure
        pressure_high = detector.calculate_pressure(
            hippocampus_activity=0.90,
            pattern_overlap=0.0,
            retrieval_success=1.0,
        )
        assert pressure_high > 0.0

    def test_detector_overlap_component(self):
        """Test pattern overlap pressure component."""
        detector = MemoryPressureDetector(
            MemoryPressureConfig(
                high_overlap_threshold=0.70,
                activity_weight=0.0,
                overlap_weight=1.0,  # Only overlap matters
                retrieval_weight=0.0,
            )
        )

        # Low overlap: no pressure
        pressure_low = detector.calculate_pressure(
            hippocampus_activity=0.0,
            pattern_overlap=0.50,
            retrieval_success=1.0,
        )
        assert pressure_low == 0.0

        # High overlap: pressure
        pressure_high = detector.calculate_pressure(
            hippocampus_activity=0.0,
            pattern_overlap=0.85,
            retrieval_success=1.0,
        )
        assert pressure_high > 0.0

    def test_detector_retrieval_component(self):
        """Test retrieval success pressure component."""
        detector = MemoryPressureDetector(
            MemoryPressureConfig(
                low_retrieval_threshold=0.85,
                activity_weight=0.0,
                overlap_weight=0.0,
                retrieval_weight=1.0,  # Only retrieval matters
            )
        )

        # Good retrieval: no pressure
        pressure_good = detector.calculate_pressure(
            hippocampus_activity=0.0,
            pattern_overlap=0.0,
            retrieval_success=0.95,
        )
        assert pressure_good == 0.0

        # Poor retrieval: pressure
        pressure_poor = detector.calculate_pressure(
            hippocampus_activity=0.0,
            pattern_overlap=0.0,
            retrieval_success=0.70,
        )
        assert pressure_poor > 0.0

    def test_detector_weighted_combination(self):
        """Test that pressure is weighted combination."""
        detector = MemoryPressureDetector(
            MemoryPressureConfig(
                activity_weight=0.5,
                overlap_weight=0.3,
                retrieval_weight=0.2,
            )
        )

        # Mixed conditions
        pressure = detector.calculate_pressure(
            hippocampus_activity=0.90,  # High
            pattern_overlap=0.50,       # Low
            retrieval_success=0.95,     # Good
        )

        # Should have moderate pressure (mostly from activity)
        assert 0.2 < pressure < 0.6

    def test_detector_urgency_levels(self):
        """Test all urgency levels."""
        detector = MemoryPressureDetector()

        assert detector.get_consolidation_urgency(0.10) == "none"
        assert detector.get_consolidation_urgency(0.35) == "low"
        assert detector.get_consolidation_urgency(0.60) == "moderate"
        assert detector.get_consolidation_urgency(0.80) == "high"
        assert detector.get_consolidation_urgency(0.95) == "critical"


# ============================================================================
# 2. Sleep Stage Controller Tests
# ============================================================================

class TestSleepStageController:
    """Test sleep stage alternation logic."""

    def test_controller_nrem_stage(self):
        """Test NREM stage detection."""
        controller = SleepStageController(
            SleepStageConfig(nrem_duration=5000, rem_duration=2000)
        )

        # Early in consolidation → NREM
        assert controller.get_current_stage(0) == SleepStage.NREM
        assert controller.get_current_stage(2500) == SleepStage.NREM
        assert controller.get_current_stage(4999) == SleepStage.NREM

    def test_controller_rem_stage(self):
        """Test REM stage detection."""
        controller = SleepStageController(
            SleepStageConfig(nrem_duration=5000, rem_duration=2000)
        )

        # After NREM → REM
        assert controller.get_current_stage(5000) == SleepStage.REM
        assert controller.get_current_stage(6000) == SleepStage.REM
        assert controller.get_current_stage(6999) == SleepStage.REM

    def test_controller_cycle_repeat(self):
        """Test that cycles repeat."""
        controller = SleepStageController(
            SleepStageConfig(nrem_duration=5000, rem_duration=2000)
        )

        # Second cycle should repeat pattern
        assert controller.get_current_stage(7000) == SleepStage.NREM  # Cycle 2 start
        assert controller.get_current_stage(12000) == SleepStage.REM  # Cycle 2 REM
        assert controller.get_current_stage(14000) == SleepStage.NREM  # Cycle 3 start

    def test_controller_replay_speed(self):
        """Test replay speed by stage."""
        controller = SleepStageController(
            SleepStageConfig(
                nrem_replay_speed=10.0,
                rem_replay_speed=20.0,
            )
        )

        # NREM: slower replay
        nrem_speed = controller.get_replay_speed(SleepStage.NREM)
        assert nrem_speed == 10.0

        # REM: faster replay
        rem_speed = controller.get_replay_speed(SleepStage.REM)
        assert rem_speed == 20.0
        assert rem_speed > nrem_speed

    def test_controller_cycle_completion(self):
        """Test cycle completion detection."""
        controller = SleepStageController(
            SleepStageConfig(nrem_duration=5000, rem_duration=2000)
        )

        # Not complete mid-cycle
        assert not controller.is_cycle_complete(3000)
        assert not controller.is_cycle_complete(6000)

        # Complete at cycle boundary
        assert controller.is_cycle_complete(7000)
        assert controller.is_cycle_complete(14000)

    def test_controller_stage_progress(self):
        """Test progress tracking within stages."""
        controller = SleepStageController(
            SleepStageConfig(nrem_duration=5000, rem_duration=2000)
        )

        # Progress through NREM
        assert controller.get_progress_in_stage(0) == 0.0  # Start
        assert controller.get_progress_in_stage(2500) == 0.5  # Halfway
        assert abs(controller.get_progress_in_stage(4999) - 1.0) < 0.01  # End

        # Progress through REM
        assert controller.get_progress_in_stage(5000) == 0.0  # Start
        assert controller.get_progress_in_stage(6000) == 0.5  # Halfway
        assert abs(controller.get_progress_in_stage(6999) - 1.0) < 0.01  # End


# ============================================================================
# 3. Consolidation Metrics Tests
# ============================================================================

class TestConsolidationMetrics:
    """Test consolidation quality tracking."""

    def test_metrics_empty_state(self):
        """Test metrics with no data."""
        metrics = ConsolidationMetrics()

        assert metrics.get_total_patterns_replayed() == 0
        assert metrics.get_consolidation_quality() == 0.0

    def test_metrics_log_transfer(self):
        """Test logging transfer events."""
        metrics = ConsolidationMetrics()

        metrics.log_transfer(
            step=100,
            patterns_replayed=50,
            cortical_learning_rate=0.03,
            retrieval_degradation=0.01,
            sleep_stage=SleepStage.NREM,
        )

        assert metrics.get_total_patterns_replayed() == 50

    def test_metrics_quality_high(self):
        """Test high quality consolidation."""
        metrics = ConsolidationMetrics()

        # Many transfers, good learning, low degradation
        for step in range(100):
            metrics.log_transfer(
                step=step,
                patterns_replayed=100,  # Many patterns
                cortical_learning_rate=0.05,  # Good learning
                retrieval_degradation=0.01,  # Low degradation
                sleep_stage=SleepStage.NREM,
            )

        quality = metrics.get_consolidation_quality()
        assert quality > 0.75  # High quality

    def test_metrics_quality_low(self):
        """Test low quality consolidation."""
        metrics = ConsolidationMetrics()

        # Few transfers, poor learning, high degradation
        for step in range(100):
            metrics.log_transfer(
                step=step,
                patterns_replayed=10,  # Few patterns
                cortical_learning_rate=0.005,  # Poor learning
                retrieval_degradation=0.08,  # High degradation
                sleep_stage=SleepStage.NREM,
            )

        quality = metrics.get_consolidation_quality()
        assert quality < 0.5  # Low quality

    def test_metrics_stage_statistics(self):
        """Test statistics broken down by stage."""
        metrics = ConsolidationMetrics()

        # NREM transfers
        for i in range(50):
            metrics.log_transfer(
                step=i,
                patterns_replayed=100,
                cortical_learning_rate=0.03,
                retrieval_degradation=0.01,
                sleep_stage=SleepStage.NREM,
            )

        # REM transfers
        for i in range(50, 70):
            metrics.log_transfer(
                step=i,
                patterns_replayed=80,
                cortical_learning_rate=0.04,
                retrieval_degradation=0.02,
                sleep_stage=SleepStage.REM,
            )

        stats = metrics.get_stage_statistics()

        # NREM stats
        assert stats[SleepStage.NREM]["count"] == 50
        assert stats[SleepStage.NREM]["patterns_replayed"] == 5000
        assert abs(stats[SleepStage.NREM]["avg_cortical_learning"] - 0.03) < 0.001

        # REM stats
        assert stats[SleepStage.REM]["count"] == 20
        assert stats[SleepStage.REM]["patterns_replayed"] == 1600
        assert abs(stats[SleepStage.REM]["avg_cortical_learning"] - 0.04) < 0.001

    def test_metrics_cumulative_patterns(self):
        """Test cumulative pattern counting."""
        metrics = ConsolidationMetrics()

        metrics.log_transfer(100, 50, 0.03, 0.01, SleepStage.NREM)
        assert metrics.get_total_patterns_replayed() == 50

        metrics.log_transfer(200, 75, 0.03, 0.01, SleepStage.NREM)
        assert metrics.get_total_patterns_replayed() == 125

        metrics.log_transfer(300, 25, 0.03, 0.01, SleepStage.REM)
        assert metrics.get_total_patterns_replayed() == 150


# ============================================================================
# 4. Consolidation Trigger Tests
# ============================================================================

class TestConsolidationTrigger:
    """Test high-level consolidation orchestration."""

    def test_trigger_first_consolidation(self):
        """Test that first consolidation can happen immediately."""
        trigger = ConsolidationTrigger(
            ConsolidationTriggerConfig(
                pressure_config=MemoryPressureConfig(
                    high_activity_threshold=0.70,
                    high_overlap_threshold=0.60,
                    pressure_threshold=0.30,
                )
            )
        )

        # High pressure should trigger
        should_consolidate, reason = trigger.should_start_consolidation(
            current_step=1000,
            hippocampus_activity=0.90,
            pattern_overlap=0.80,
            retrieval_success=0.75,
        )

        assert should_consolidate
        assert "pressure" in reason.lower()

    def test_trigger_minimum_interval(self):
        """Test minimum interval enforcement."""
        trigger = ConsolidationTrigger(
            ConsolidationTriggerConfig(min_consolidation_interval=50000)
        )

        # First consolidation at step 10000
        trigger.mark_consolidation_started(10000)

        # Try again at step 40000 (only 30k elapsed)
        should_consolidate, reason = trigger.should_start_consolidation(
            current_step=40000,
            hippocampus_activity=0.90,
            pattern_overlap=0.80,
            retrieval_success=0.75,
        )

        assert not should_consolidate
        assert "Too soon" in reason

    def test_trigger_interval_elapsed(self):
        """Test consolidation after interval elapsed."""
        trigger = ConsolidationTrigger(
            ConsolidationTriggerConfig(
                min_consolidation_interval=50000,
                pressure_config=MemoryPressureConfig(
                    high_activity_threshold=0.70,
                    high_overlap_threshold=0.60,
                    pressure_threshold=0.30,
                ),
            )
        )

        # First consolidation at step 10000
        trigger.mark_consolidation_started(10000)

        # Try again at step 70000 (60k elapsed > 50k minimum)
        should_consolidate, reason = trigger.should_start_consolidation(
            current_step=70000,
            hippocampus_activity=0.90,
            pattern_overlap=0.80,
            retrieval_success=0.75,
        )

        assert should_consolidate
        assert "pressure" in reason.lower()

    def test_trigger_low_pressure_blocks(self):
        """Test that low pressure blocks consolidation."""
        trigger = ConsolidationTrigger()

        # Low pressure conditions (even after interval)
        should_consolidate, reason = trigger.should_start_consolidation(
            current_step=100000,
            hippocampus_activity=0.50,
            pattern_overlap=0.40,
            retrieval_success=0.95,
        )

        assert not should_consolidate
        assert "Low pressure" in reason

    def test_trigger_can_consolidate_now(self):
        """Test can_consolidate_now helper."""
        trigger = ConsolidationTrigger(
            ConsolidationTriggerConfig(min_consolidation_interval=50000)
        )

        # No previous consolidation
        assert trigger.can_consolidate_now(0)
        assert trigger.can_consolidate_now(100000)

        # After marking consolidation
        trigger.mark_consolidation_started(10000)
        assert not trigger.can_consolidate_now(40000)  # Too soon
        assert trigger.can_consolidate_now(70000)  # Enough time

    def test_trigger_get_sleep_controller(self):
        """Test sleep controller access."""
        trigger = ConsolidationTrigger()

        controller = trigger.get_sleep_controller()
        assert isinstance(controller, SleepStageController)

        # Should be usable
        stage = controller.get_current_stage(1000)
        assert stage in [SleepStage.NREM, SleepStage.REM]


# ============================================================================
# Integration Tests
# ============================================================================

class TestConsolidationIntegration:
    """Test realistic consolidation scenarios."""

    def test_full_consolidation_cycle(self):
        """Test complete consolidation cycle."""
        # Setup
        trigger = ConsolidationTrigger()
        metrics = ConsolidationMetrics()

        # Training phase (pressure builds)
        training_step = 100000

        # Check if should consolidate
        should_consolidate, reason = trigger.should_start_consolidation(
            current_step=training_step,
            hippocampus_activity=0.88,
            pattern_overlap=0.75,
            retrieval_success=0.82,
        )

        if should_consolidate:
            # Mark consolidation started
            trigger.mark_consolidation_started(training_step)

            # Get sleep controller
            controller = trigger.get_sleep_controller()

            # Simulate consolidation (1 full cycle)
            for step in range(7000):
                stage = controller.get_current_stage(step)

                # Log transfer
                metrics.log_transfer(
                    step=step,
                    patterns_replayed=100 if stage == SleepStage.NREM else 80,
                    cortical_learning_rate=0.03,
                    retrieval_degradation=0.01,
                    sleep_stage=stage,
                )

            # Check quality
            quality = metrics.get_consolidation_quality()
            assert quality > 0.5  # Reasonable consolidation

            # Check that cycle completed
            assert controller.is_cycle_complete(7000)

    def test_multiple_consolidations(self):
        """Test multiple consolidation episodes."""
        trigger = ConsolidationTrigger(
            ConsolidationTriggerConfig(
                min_consolidation_interval=50000,
                pressure_config=MemoryPressureConfig(
                    high_activity_threshold=0.70,
                    high_overlap_threshold=0.60,
                    pressure_threshold=0.30,
                ),
            )
        )

        consolidations = []

        # Simulate training with periodic consolidations
        for step in range(0, 300000, 60000):
            should_consolidate, reason = trigger.should_start_consolidation(
                current_step=step,
                hippocampus_activity=0.88,
                pattern_overlap=0.75,
                retrieval_success=0.82,
            )

            if should_consolidate:
                trigger.mark_consolidation_started(step)
                consolidations.append(step)

        # Should have multiple consolidations
        assert len(consolidations) >= 3

        # Should be spaced appropriately
        for i in range(1, len(consolidations)):
            interval = consolidations[i] - consolidations[i-1]
            assert interval >= 50000

    def test_pressure_buildup_scenario(self):
        """Test gradual pressure buildup leading to consolidation."""
        detector = MemoryPressureDetector(
            MemoryPressureConfig(
                high_activity_threshold=0.60,
                high_overlap_threshold=0.50,
                low_retrieval_threshold=0.90,
                pressure_threshold=0.40,
            )
        )

        # Simulate gradual pressure increase
        pressures = []

        for activity in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
            pressure = detector.calculate_pressure(
                hippocampus_activity=activity,
                pattern_overlap=activity - 0.05,
                retrieval_success=1.0 - (activity - 0.50),
            )
            pressures.append(pressure)

        # Pressure should increase monotonically
        assert pressures == sorted(pressures)

        # Eventually triggers consolidation
        assert not detector.should_trigger_consolidation(pressures[0])  # Early
        assert detector.should_trigger_consolidation(pressures[-1])  # Late

    def test_sleep_stage_transitions(self):
        """Test realistic sleep stage transitions."""
        controller = SleepStageController(
            SleepStageConfig(nrem_duration=5000, rem_duration=2000)
        )

        stages = []
        for step in range(0, 14000, 100):
            stage = controller.get_current_stage(step)
            stages.append(stage)

        # Count transitions
        transitions = 0
        for i in range(1, len(stages)):
            if stages[i] != stages[i-1]:
                transitions += 1

        # Should have at least 3 transitions (N→R, R→N, N→R at minimum)
        # 14000 steps, 7000 per cycle = 2 cycles = 3-4 transitions
        assert transitions >= 3

        # Should start with NREM
        assert stages[0] == SleepStage.NREM
