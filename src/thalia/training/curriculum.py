"""Curriculum Training Infrastructure - Learning Science-Based Training Strategies.

This module implements sophisticated curriculum strategies from learning science
and cognitive psychology research, not just simple staged training.

**Advanced Curriculum Mechanics**:
===================================
1. **Interleaved practice**: Multinomial task sampling (mix tasks, not blocks)
2. **Spaced repetition**: Leitner expanding intervals (optimal review timing)
3. **Testing effect**: Retrieval practice without immediate feedback
4. **Productive failure**: Intentional difficulty before scaffolding
5. **Dynamic difficulty**: Zone of Proximal Development (70-80% success)
6. **Stage transitions**: Gradual difficulty ramps (no sudden jumps)

**Key Learning Science Principles**:
=====================================

1. **INTERLEAVED PRACTICE**:
   Mixing tasks from multiple stages in each session (not blocked):

   - **Better discrimination**: Forces comparison between concepts
   - **Context reloading**: Must switch mental models frequently
   - **Durable learning**: More resistant to forgetting

2. **SPACED REPETITION**:
   Expanding intervals between review sessions (Leitner algorithm):

   - **Optimal timing**: Review just before forgetting
   - **Exponential spacing**: Well-retained → longer intervals
   - **Reset for errors**: Forgotten material → back to short intervals

3. **TESTING EFFECT**:
   Frequent low-stakes testing WITHOUT immediate feedback:

   - **Testing beats re-study**: More effective than passive review
   - **Retrieval effort**: Strengthens memory traces
   - **Delayed feedback**: Encourages effortful retrieval

4. **PRODUCTIVE FAILURE**:
   Allow struggle (~20% success) BEFORE scaffolding:

   - **Activates prior knowledge**: Prepares mind for new information
   - **Makes instruction effective**: Subsequent teaching "clicks"
   - **Metacognitive awareness**: Learn what you don't know

5. **ZONE OF PROXIMAL DEVELOPMENT** (ZPD):
   Maintain optimal challenge (70-80% success rate):

   - **Too easy (>90%)**: No learning, boredom
   - **Too hard (<60%)**: Frustration, giving up
   - **Just right (70-80%)**: Optimal growth region

6. STAGE TRANSITIONS
   Gradual difficulty ramps between curriculum stages:
   - Prevents "cliff" transitions
   - Maintains old knowledge while learning new
   - Builds confidence through early success
   Evidence: Scaffolding fading principles
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class CurriculumStage(IntEnum):
    """Curriculum stages matching main training plan."""

    BOOTSTRAP = 0  # Developmental initialization (pre-curriculum)
    SENSORIMOTOR = 1  # TODO
    # PHONOLOGY = 2  # TODO
    # TODDLER = 3  # TODO
    # GRAMMAR = 4  # TODO
    # READING = 5  # TODO
    # ABSTRACT = 6  # TODO
    # YOUNG_ADULT = 7  # TODO
    # ADULT = 8  # TODO


# ============================================================================
# 1. Interleaved Curriculum Sampler
# ============================================================================


@dataclass
class InterleavedCurriculumSamplerConfig:
    """Configuration for interleaved curriculum sampling."""

    seed: Optional[int] = None  # Random seed for reproducibility


class InterleavedCurriculumSampler:
    """Sample tasks from multinomial distribution each step (interleaved practice).

    Interleaved practice mixes tasks from multiple stages within each session,
    rather than practicing one stage in blocks. This forces the brain to
    discriminate between contexts and leads to better long-term retention.
    """

    def __init__(self, config: Optional[InterleavedCurriculumSamplerConfig] = None):
        """Initialize sampler.

        Args:
            config: Configuration (seed for reproducibility)
        """
        self.config = config or InterleavedCurriculumSamplerConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)

    def sample_next_task(self, stage_weights: Dict[int, float]) -> int:
        """Sample next task stage from distribution.

        Args:
            stage_weights: Dictionary of {stage_id: weight}
                          Example: {0: 0.05, 1: 0.10, 2: 0.85}
                          Weights should sum to ~1.0

        Returns:
            stage_id: Sampled stage ID

        Raises:
            ValueError: If no stages provided or weights don't sum to ~1.0
        """
        if not stage_weights:
            raise ValueError("stage_weights cannot be empty")

        # Normalize weights (handle small deviations from 1.0)
        total = sum(stage_weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Weights should sum to ~1.0, got {total}. " f"Weights: {stage_weights}"
            )

        # Sample from multinomial
        stages = list(stage_weights.keys())
        weights = [stage_weights[s] for s in stages]

        return random.choices(stages, weights=weights, k=1)[0]


# ============================================================================
# 2. Spaced Repetition Scheduler
# ============================================================================


@dataclass
class SpacedRepetitionSchedulerConfig:
    """Configuration for spaced repetition scheduler."""

    # Base intervals by stage (conservative)
    base_intervals: Dict[int, int] = field(
        default_factory=lambda: {
            0: 50000,  # Stage 0: Very long (foundation preservation)
            1: 25000,  # Stage 1: Moderate
            2: 15000,  # Stage 2: Shorter (more review)
            3: 25000,  # Stage 3+: Moderate
        }
    )

    # Expansion factor for well-retained knowledge
    expansion_factor: float = 1.5

    # Reset interval for forgotten material
    reset_interval: int = 10000

    # Performance thresholds
    high_performance_threshold: float = 0.92
    low_performance_threshold: float = 0.85


class SpacedRepetitionScheduler:
    """Leitner-style spaced repetition with expanding intervals.

    Reviews are scheduled based on performance:
    - High performance (>92%): Expand interval (× 1.5)
    - Low performance (<85%): Reset to short interval (10k steps)
    - Moderate: Use base interval for stage

    This creates optimal "just before forgetting" timing.
    """

    def __init__(self, config: Optional[SpacedRepetitionSchedulerConfig] = None):
        """Initialize scheduler.

        Args:
            config: Configuration for intervals and thresholds
        """
        self.config = config or SpacedRepetitionSchedulerConfig()

    def calculate_interval(
        self,
        stage: int,
        performance: float,
        review_count: int,
    ) -> int:
        """Calculate optimal review interval for a stage.

        Args:
            stage: Stage ID (0, 1, 2, ...)
            performance: Current performance on stage (0-1)
            review_count: Number of times stage has been reviewed

        Returns:
            interval: Steps until next review
        """
        cfg = self.config

        # Get base interval for stage
        base_interval = cfg.base_intervals.get(stage, 25000)

        if performance > cfg.high_performance_threshold:
            # Well-retained: expand interval exponentially
            interval = int(base_interval * (cfg.expansion_factor**review_count))
        elif performance < cfg.low_performance_threshold:
            # Forgotten: reset to short interval
            interval = cfg.reset_interval
        else:
            # Moderate: use base interval
            interval = base_interval

        return interval

    def should_review_stage(
        self,
        stage: int,
        last_review_step: int,
        current_step: int,
        performance: float,
        review_count: int = 0,
    ) -> Tuple[bool, int]:
        """Check if stage should be reviewed now.

        Args:
            stage: Stage ID
            last_review_step: Step when stage was last reviewed
            current_step: Current training step
            performance: Current performance on stage (0-1)
            review_count: Number of times reviewed

        Returns:
            Tuple of (should_review, interval):
            - should_review: True if due for review
            - interval: Optimal interval for next review
        """
        interval = self.calculate_interval(stage, performance, review_count)
        steps_since_review = current_step - last_review_step

        should_review = steps_since_review >= interval

        return should_review, interval

    def get_review_schedule(
        self,
        stages: List[int],
        stage_history: Dict[int, int],
        current_step: int,
        stage_performance: Dict[int, float],
        stage_review_counts: Dict[int, int],
    ) -> Dict[int, float]:
        """Get review distribution weights for all stages.

        Useful for curriculum mixing: allocate review time proportionally.

        Args:
            stages: List of all stages
            stage_history: {stage_id: last_review_step}
            current_step: Current training step
            stage_performance: {stage_id: performance}
            stage_review_counts: {stage_id: review_count}

        Returns:
            Distribution weights {stage_id: weight} (sums to 1.0)
        """
        review_weights = {}

        for stage in stages:
            last_review = stage_history.get(stage, 0)
            perf = stage_performance.get(stage, 0.9)
            count = stage_review_counts.get(stage, 0)

            should_review, interval = self.should_review_stage(
                stage=stage,
                last_review_step=last_review,
                current_step=current_step,
                performance=perf,
                review_count=count,
            )

            if should_review:
                # Urgency: how overdue is the review?
                steps_overdue = (current_step - last_review) - interval
                urgency = max(0.1, min(1.0, steps_overdue / interval))
                review_weights[stage] = urgency
            else:
                review_weights[stage] = 0.0

        # Normalize
        total = sum(review_weights.values())
        if total > 0:
            review_weights = {s: w / total for s, w in review_weights.items()}

        return review_weights


# ============================================================================
# 3. Testing Phase Protocol
# ============================================================================


@dataclass
class TestingPhaseConfig:
    """Configuration for testing protocol."""

    test_frequency: float = 0.15  # 15% of steps are tests
    feedback_delay_steps: int = 100  # Delay before feedback


class TestingPhaseProtocol:
    """Frequent low-stakes testing without immediate feedback.

    Testing (retrieval practice) is more effective than re-studying for
    long-term retention. This protocol implements testing phases where:
    - No learning signal is provided
    - Accuracy is logged for later analysis
    - Feedback is delayed to allow consolidation
    """

    def __init__(self, config: Optional[TestingPhaseConfig] = None):
        """Initialize testing protocol.

        Args:
            config: Configuration for test frequency and feedback delay
        """
        self.config = config or TestingPhaseConfig()
        self._test_history: Dict[int, bool] = {}  # {step: correct}

    def should_test(self, step: int) -> bool:
        """Check if current step should be a test (no learning).

        Args:
            step: Current training step

        Returns:
            True if this should be a test step
        """
        # Deterministic based on step (for reproducibility)
        # Every Nth step is a test, where N = 1 / test_frequency
        test_interval = int(1.0 / self.config.test_frequency)
        return step % test_interval == 0

    def log_test_result(self, step: int, correct: bool) -> None:
        """Log result of a test.

        Args:
            step: Training step
            correct: Whether prediction was correct
        """
        self._test_history[step] = correct

    def get_test_accuracy(self, last_n_steps: Optional[int] = None) -> float:
        """Get accuracy on recent tests.

        Args:
            last_n_steps: If provided, only consider last N steps

        Returns:
            Accuracy (0-1), or 0.0 if no tests
        """
        if not self._test_history:
            return 0.0

        if last_n_steps is not None:
            max_step = max(self._test_history.keys())
            min_step = max_step - last_n_steps
            results = [v for k, v in self._test_history.items() if k >= min_step]
        else:
            results = list(self._test_history.values())

        if not results:
            return 0.0

        return sum(results) / len(results)


# ============================================================================
# 4. Productive Failure Phase
# ============================================================================


@dataclass
class ProductiveFailureConfig:
    """Configuration for productive failure phases."""

    target_success_rate: float = 0.20  # Intentionally low (struggle)
    duration_steps: int = 5000  # Duration of failure phase
    min_attempts: int = 100  # Minimum attempts before ending early


class ProductiveFailurePhase:
    """Intentional difficulty before scaffolding (productive failure).

    Allow struggle (~20% success) before providing teaching/scaffolding.
    This activates prior knowledge and makes subsequent instruction more effective.

    Phases:
    1. Productive Failure (duration_steps): ~20% success, no scaffolding
    2. Instruction: Teaching/scaffolding introduced
    3. Practice: Normal training with support
    """

    def __init__(self, config: Optional[ProductiveFailureConfig] = None):
        """Initialize productive failure protocol.

        Args:
            config: Configuration for failure phase
        """
        self.config = config or ProductiveFailureConfig()

    def is_in_failure_phase(
        self,
        stage_start_step: int,
        current_step: int,
    ) -> bool:
        """Check if currently in productive failure phase.

        Args:
            stage_start_step: Step when stage began
            current_step: Current training step

        Returns:
            True if in failure phase (should struggle)
        """
        steps_since_start = current_step - stage_start_step
        return 0 <= steps_since_start < self.config.duration_steps

    def get_target_success_rate(self) -> float:
        """Get target success rate for failure phase.

        Returns:
            Target success rate (typically ~0.20)
        """
        return self.config.target_success_rate


# ============================================================================
# 5. Curriculum Difficulty Calibrator
# ============================================================================


@dataclass
class DifficultyCalibratorConfig:
    """Configuration for difficulty calibration."""

    target_success_rate: float = 0.75  # Optimal challenge (ZPD)
    adjustment_rate: float = 0.05  # How much to adjust per step
    # Performance bands
    too_easy_threshold: float = 0.90
    too_hard_threshold: float = 0.60


class CurriculumDifficultyCalibrator:
    """Adjust task difficulty to maintain Zone of Proximal Development.

    Dynamically adjusts difficulty to maintain ~75% success rate:
    - Too easy (>90%): Increase difficulty
    - Too hard (<60%): Decrease difficulty
    - Just right (60-90%): Maintain
    """

    def __init__(self, config: Optional[DifficultyCalibratorConfig] = None):
        """Initialize calibrator.

        Args:
            config: Configuration for target rate and adjustment
        """
        self.config = config or DifficultyCalibratorConfig()

    def calibrate(
        self,
        success_rate: float,
        current_difficulty: float,
    ) -> float:
        """Adjust difficulty based on success rate.

        Args:
            success_rate: Current success rate (0-1)
            current_difficulty: Current difficulty level (0-1)

        Returns:
            new_difficulty: Adjusted difficulty level (0-1)
        """
        cfg = self.config

        if success_rate > cfg.too_easy_threshold:
            # Too easy → increase difficulty
            new_difficulty = min(1.0, current_difficulty + cfg.adjustment_rate)
        elif success_rate < cfg.too_hard_threshold:
            # Too hard → decrease difficulty
            new_difficulty = max(0.0, current_difficulty - cfg.adjustment_rate)
        else:
            # Just right → maintain
            new_difficulty = current_difficulty

        return new_difficulty


# ============================================================================
# 6. Stage Transition Protocol
# ============================================================================


@dataclass
class TransitionWeekConfig:
    """Configuration for a week during stage transition."""

    difficulty: float  # Task difficulty (0-1)
    old_stage_ratio: float  # Fraction of old stage review (0-1)


@dataclass
class StageTransitionConfig:
    """Configuration for stage transitions."""

    # 4-week difficulty ramp
    week_1_difficulty: float = 0.3  # Very easy intro
    week_2_difficulty: float = 0.5  # Easy
    week_3_difficulty: float = 0.7  # Moderate
    week_4_difficulty: float = 1.0  # Full difficulty

    # 3-week mixing schedule (old stage review)
    week_1_old_ratio: float = 0.70  # High review (70% old)
    week_2_old_ratio: float = 0.50  # Moderate review (50% old)
    week_3_old_ratio: float = 0.30  # Normal review (30% old)


class StageTransitionProtocol:
    """Gradual difficulty ramps during stage transitions.

    Implements smooth transitions between curriculum stages:
    1. Extended consolidation (before transition)
    2. Milestone evaluation (go/no-go decision)
    3. Gradual difficulty ramp (4 weeks)
    4. High initial review (70% → 50% → 30% over 3 weeks)

    Prevents "cliff" transitions that cause failures.
    """

    def __init__(self, config: Optional[StageTransitionConfig] = None):
        """Initialize transition protocol.

        Args:
            config: Configuration for ramps and mixing
        """
        self.config = config or StageTransitionConfig()

    def get_transition_config(
        self,
        old_stage: int,
        new_stage: int,
        weeks_since_transition: int,
    ) -> TransitionWeekConfig:
        """Get configuration for current week of transition.

        Args:
            old_stage: Previous stage ID
            new_stage: New stage ID
            weeks_since_transition: Weeks since transition began (0-indexed)

        Returns:
            TransitionWeekConfig with difficulty and mixing schedule
        """
        cfg = self.config

        # Difficulty ramp (4 weeks)
        if weeks_since_transition == 0:
            difficulty = cfg.week_1_difficulty
        elif weeks_since_transition == 1:
            difficulty = cfg.week_2_difficulty
        elif weeks_since_transition == 2:
            difficulty = cfg.week_3_difficulty
        else:  # Week 3+
            difficulty = cfg.week_4_difficulty

        # Mixing schedule (3 weeks)
        if weeks_since_transition == 0:
            old_ratio = cfg.week_1_old_ratio
        elif weeks_since_transition == 1:
            old_ratio = cfg.week_2_old_ratio
        else:  # Week 2+
            old_ratio = cfg.week_3_old_ratio

        return TransitionWeekConfig(
            difficulty=difficulty,
            old_stage_ratio=old_ratio,
        )

    def is_transition_complete(self, weeks_since_transition: int) -> bool:
        """Check if transition period is complete.

        Args:
            weeks_since_transition: Weeks since transition began

        Returns:
            True if transition complete (>= 4 weeks)
        """
        return weeks_since_transition >= 4
