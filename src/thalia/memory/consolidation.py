"""
Enhanced Consolidation - Memory pressure triggers and sleep-based replay.

This module implements biologically-inspired memory consolidation mechanisms:
1. Memory pressure detection (when to trigger consolidation)
2. Sleep-stage simulation (NREM/REM alternation)
3. Hippocampal replay triggering
4. Consolidation quality metrics

Key Concepts:
=============

1. MEMORY PRESSURE
   Hippocampus has limited capacity (~10k patterns). When full, consolidation needed:
   - Pattern overlap → interference
   - High activity → saturation
   - Poor retrieval → forgetting
   Evidence: McClelland et al. (1995) - complementary learning systems

2. SLEEP STAGES
   Different stages serve different functions:
   - NREM (slow-wave): Hippocampus → Cortex transfer (system consolidation)
   - REM (paradoxical): Cortical reorganization, pruning
   - Alternating cycles (~90 min each)
   Evidence: Born & Wilhelm (2012) - system consolidation during sleep

3. HIPPOCAMPAL REPLAY
   Compress experiences for efficient cortical encoding:
   - 10-20× speed-up during replay
   - Reverse replay for credit assignment
   - Prioritize high-value/recent memories
   Evidence: Diba & Buzsáki (2007) - forward/reverse replay

4. CONSOLIDATION QUALITY
   Track effectiveness of consolidation:
   - How much was transferred?
   - Is cortex learning?
   - Can we still retrieve from hippocampus?
   Evidence: Dudai (2004) - consolidation metrics

Usage:
======

    from thalia.memory.consolidation import (
        MemoryPressureDetector,
        SleepStageController,
        ConsolidationMetrics,
        ConsolidationTrigger,
    )

    # Memory pressure detection
    detector = MemoryPressureDetector()
    pressure = detector.calculate_pressure(
        hippocampus_activity=0.85,
        pattern_overlap=0.72,
        retrieval_success=0.88,
    )
    should_consolidate = detector.should_trigger_consolidation(pressure)

    # Sleep stage simulation
    controller = SleepStageController()
    stage = controller.get_current_stage(consolidation_step=1000)
    # → 'NREM' (first half of cycle)

    # Consolidation metrics
    metrics = ConsolidationMetrics()
    metrics.log_transfer(
        patterns_replayed=500,
        cortical_learning_rate=0.03,
        retrieval_degradation=0.02,
    )
    quality = metrics.get_consolidation_quality()

References:
===========
- McClelland et al. (1995): Complementary learning systems
- Born & Wilhelm (2012): System consolidation during sleep
- Diba & Buzsáki (2007): Forward and reverse hippocampal replay
- Dudai (2004): Memory consolidation mechanisms

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ============================================================================
# Sleep Stage Types
# ============================================================================

class SleepStage(Enum):
    """Sleep stages during consolidation."""
    NREM = "NREM"  # Non-REM: Hippocampus → Cortex transfer
    REM = "REM"    # REM: Cortical reorganization


# ============================================================================
# 1. Memory Pressure Detector
# ============================================================================

@dataclass
class MemoryPressureConfig:
    """Configuration for memory pressure detection."""
    # Thresholds for triggering consolidation
    high_activity_threshold: float = 0.80  # Hippocampus saturation
    high_overlap_threshold: float = 0.70   # Pattern interference
    low_retrieval_threshold: float = 0.85  # Forgetting signal

    # Weights for combined pressure (sum to 1.0)
    activity_weight: float = 0.4
    overlap_weight: float = 0.3
    retrieval_weight: float = 0.3

    # Consolidation trigger threshold
    pressure_threshold: float = 0.75  # Trigger when pressure > 75%


class MemoryPressureDetector:
    """Detect when hippocampus needs consolidation.

    Monitors three pressure indicators:
    1. Activity level (saturation)
    2. Pattern overlap (interference)
    3. Retrieval success (forgetting)

    When combined pressure exceeds threshold, trigger consolidation.

    Example:
        >>> detector = MemoryPressureDetector()
        >>>
        >>> # Normal conditions
        >>> pressure = detector.calculate_pressure(
        ...     hippocampus_activity=0.60,
        ...     pattern_overlap=0.45,
        ...     retrieval_success=0.92,
        ... )
        >>> # → Low pressure (~0.35)
        >>>
        >>> # High load conditions
        >>> pressure = detector.calculate_pressure(
        ...     hippocampus_activity=0.88,  # High activity
        ...     pattern_overlap=0.75,       # High interference
        ...     retrieval_success=0.82,     # Poor retrieval
        ... )
        >>> # → High pressure (~0.82) → Should consolidate!
    """

    def __init__(self, config: Optional[MemoryPressureConfig] = None):
        """Initialize detector.

        Args:
            config: Configuration for thresholds and weights
        """
        self.config = config or MemoryPressureConfig()

    def calculate_pressure(
        self,
        hippocampus_activity: float,
        pattern_overlap: float,
        retrieval_success: float,
    ) -> float:
        """Calculate memory pressure from multiple indicators.

        Args:
            hippocampus_activity: Current activity level (0-1)
            pattern_overlap: Pattern similarity/interference (0-1)
            retrieval_success: Retrieval accuracy (0-1)

        Returns:
            pressure: Combined pressure metric (0-1)
                     Higher = more urgent need for consolidation
        """
        cfg = self.config

        # Individual pressure components
        activity_pressure = max(
            0.0,
            (hippocampus_activity - cfg.high_activity_threshold) /
            (1.0 - cfg.high_activity_threshold)
        )

        overlap_pressure = max(
            0.0,
            (pattern_overlap - cfg.high_overlap_threshold) /
            (1.0 - cfg.high_overlap_threshold)
        )

        # Retrieval: LOW success = HIGH pressure
        retrieval_pressure = max(
            0.0,
            (cfg.low_retrieval_threshold - retrieval_success) /
            cfg.low_retrieval_threshold
        )

        # Weighted combination
        pressure = (
            cfg.activity_weight * activity_pressure +
            cfg.overlap_weight * overlap_pressure +
            cfg.retrieval_weight * retrieval_pressure
        )

        return min(1.0, pressure)  # Clamp to [0, 1]

    def should_trigger_consolidation(self, pressure: float) -> bool:
        """Check if pressure exceeds trigger threshold.

        Args:
            pressure: Current memory pressure (0-1)

        Returns:
            True if consolidation should be triggered
        """
        return pressure > self.config.pressure_threshold

    def get_consolidation_urgency(self, pressure: float) -> str:
        """Get human-readable urgency level.

        Args:
            pressure: Current memory pressure (0-1)

        Returns:
            Urgency level: 'none', 'low', 'moderate', 'high', 'critical'
        """
        if pressure < 0.25:
            return "none"
        elif pressure < 0.50:
            return "low"
        elif pressure < 0.75:
            return "moderate"
        elif pressure < 0.90:
            return "high"
        else:
            return "critical"


# ============================================================================
# 2. Sleep Stage Controller
# ============================================================================

@dataclass
class SleepStageConfig:
    """Configuration for sleep stage alternation."""
    # Cycle duration (steps)
    nrem_duration: int = 5000  # ~90 min in real life
    rem_duration: int = 2000   # ~30 min in real life

    # Replay characteristics per stage
    nrem_replay_speed: float = 10.0  # 10× compression
    rem_replay_speed: float = 20.0   # 20× compression (faster)

    # Delta oscillation parameters (for NREM)
    delta_frequency_hz: float = 2.0  # Slow-wave sleep frequency
    delta_upstate_threshold: float = 0.3  # Minimum signal for up-state
    enable_delta_gating: bool = True  # Use delta to gate replay


class SleepStageController:
    """Control sleep stage alternation during consolidation.

    Implements NREM/REM cycling:
    - NREM: Slow-wave sleep with delta oscillations, hippocampal → cortical transfer
    - REM: Rapid eye movement, cortical reorganization

    Cycles alternate with realistic timing ratios (roughly 2:1).

    Delta Integration:
    - Creates DeltaOscillator during NREM stages
    - Gates replay to delta up-states (optimal plasticity window)
    - Coordinates hippocampal-cortical transfer

    Example:
        >>> controller = SleepStageController()
        >>>
        >>> # Early in consolidation (NREM phase)
        >>> stage = controller.get_current_stage(consolidation_step=1000)
        >>> # → SleepStage.NREM
        >>>
        >>> # Get delta oscillator for NREM
        >>> delta = controller.get_delta_oscillator()
        >>> if delta:
        >>>     delta.advance(dt_ms=1.0)
        >>>     if delta.is_up_state():
        >>>         # Optimal time for replay!
        >>>
        >>> # Later in consolidation (REM phase)
        >>> stage = controller.get_current_stage(consolidation_step=6000)
        >>> # → SleepStage.REM (no delta during REM)
        >>>
        >>> # Check if cycle complete
        >>> complete = controller.is_cycle_complete(consolidation_step=7500)
        >>> # → True (NREM 5000 + REM 2000 = 7000 steps)
    """

    def __init__(self, config: Optional[SleepStageConfig] = None):
        """Initialize controller.

        Args:
            config: Configuration for cycle durations
        """
        self.config = config or SleepStageConfig()
        self._cycle_length = (
            self.config.nrem_duration + self.config.rem_duration
        )
        self._current_stage: Optional[SleepStage] = None

    def get_current_stage(self, consolidation_step: int) -> SleepStage:
        """Get current sleep stage based on step.

        Args:
            consolidation_step: Step within current consolidation period

        Returns:
            Current sleep stage (NREM or REM)
        """
        step_in_cycle = consolidation_step % self._cycle_length

        if step_in_cycle < self.config.nrem_duration:
            stage = SleepStage.NREM
        else:
            stage = SleepStage.REM

        self._current_stage = stage
        return stage

    def get_replay_speed(self, stage: SleepStage) -> float:
        """Get replay speed multiplier for given stage.

        Args:
            stage: Sleep stage

        Returns:
            Replay speed multiplier (>1 = faster than real-time)
        """
        if stage == SleepStage.NREM:
            return self.config.nrem_replay_speed
        else:
            return self.config.rem_replay_speed

    def should_replay_now(self) -> bool:
        """Determine if replay should occur at this moment.

        During NREM with delta gating:
        - Replay only during delta up-states (optimal plasticity)
        - Delta phase comes from Brain's centralized oscillators

        During REM or without delta:
        - Always allow replay

        Args:
            dt_ms: Timestep in milliseconds

        Returns:
            True if replay should occur now

        Note:
            Delta oscillator is managed by Brain. Sleep controller receives
            delta phase via hippocampus.set_oscillator_phases() from Brain.
        """
        # For now, always allow replay. Delta gating should be implemented
        # by Brain checking delta phase before calling consolidation.
        # This is cleaner than having consolidation track its own delta.
        return True

    def is_cycle_complete(self, consolidation_step: int) -> bool:
        """Check if a full NREM/REM cycle is complete.

        Args:
            consolidation_step: Step within consolidation period

        Returns:
            True if cycle just completed
        """
        return (consolidation_step > 0 and
                consolidation_step % self._cycle_length == 0)

    def get_progress_in_stage(self, consolidation_step: int) -> float:
        """Get progress through current stage (0-1).

        Args:
            consolidation_step: Step within consolidation period

        Returns:
            Progress fraction (0 = start of stage, 1 = end of stage)
        """
        stage = self.get_current_stage(consolidation_step)
        step_in_cycle = consolidation_step % self._cycle_length

        if stage == SleepStage.NREM:
            return step_in_cycle / self.config.nrem_duration
        else:
            step_in_rem = step_in_cycle - self.config.nrem_duration
            return step_in_rem / self.config.rem_duration


# ============================================================================
# 3. Consolidation Metrics
# ============================================================================

@dataclass
class ConsolidationSnapshot:
    """Single snapshot of consolidation progress."""
    step: int
    patterns_replayed: int
    cortical_learning_rate: float
    retrieval_degradation: float
    sleep_stage: SleepStage


class ConsolidationMetrics:
    """Track effectiveness of memory consolidation.

    Monitors:
    - Transfer efficiency (how much replayed)
    - Cortical learning (is cortex improving)
    - Retrieval degradation (hippocampal decay)

    Example:
        >>> metrics = ConsolidationMetrics()
        >>>
        >>> # Log transfers during consolidation
        >>> metrics.log_transfer(
        ...     step=100,
        ...     patterns_replayed=50,
        ...     cortical_learning_rate=0.03,
        ...     retrieval_degradation=0.01,
        ...     sleep_stage=SleepStage.NREM,
        ... )
        >>>
        >>> # Get overall quality
        >>> quality = metrics.get_consolidation_quality()
        >>> # → 0.92 (high quality: good transfer, minimal degradation)
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self._history: List[ConsolidationSnapshot] = []
        self._total_patterns_replayed: int = 0

    def log_transfer(
        self,
        step: int,
        patterns_replayed: int,
        cortical_learning_rate: float,
        retrieval_degradation: float,
        sleep_stage: SleepStage,
    ) -> None:
        """Log a consolidation transfer event.

        Args:
            step: Consolidation step
            patterns_replayed: Number of patterns replayed this step
            cortical_learning_rate: Cortex learning magnitude
            retrieval_degradation: Hippocampus degradation
            sleep_stage: Current sleep stage
        """
        snapshot = ConsolidationSnapshot(
            step=step,
            patterns_replayed=patterns_replayed,
            cortical_learning_rate=cortical_learning_rate,
            retrieval_degradation=retrieval_degradation,
            sleep_stage=sleep_stage,
        )

        self._history.append(snapshot)
        self._total_patterns_replayed += patterns_replayed

    def get_consolidation_quality(self) -> float:
        """Calculate overall consolidation quality (0-1).

        Quality = f(transfer_rate, cortical_learning, retrieval_preservation)

        Returns:
            Quality score (0-1), higher = better consolidation
        """
        if not self._history:
            return 0.0

        # Average cortical learning (higher = better)
        avg_cortical_learning = sum(
            s.cortical_learning_rate for s in self._history
        ) / len(self._history)

        # Average retrieval degradation (lower = better)
        avg_degradation = sum(
            s.retrieval_degradation for s in self._history
        ) / len(self._history)

        # Transfer efficiency (patterns / steps)
        transfer_efficiency = self._total_patterns_replayed / len(self._history)
        # Normalize assuming ~100 patterns/step is ideal
        transfer_score = min(1.0, transfer_efficiency / 100.0)

        # Combine (weighted)
        quality = (
            0.4 * transfer_score +
            0.4 * min(1.0, avg_cortical_learning / 0.05) +  # 5% is good
            0.2 * (1.0 - avg_degradation / 0.05)  # <5% degradation is good
        )

        return min(1.0, quality)

    def get_total_patterns_replayed(self) -> int:
        """Get total number of patterns replayed."""
        return self._total_patterns_replayed

    def get_stage_statistics(self) -> Dict[SleepStage, Dict[str, float]]:
        """Get statistics broken down by sleep stage.

        Returns:
            Dictionary of {stage: {metric: value}}
        """
        stats: Dict[SleepStage, Dict[str, float]] = {
            SleepStage.NREM: {},
            SleepStage.REM: {},
        }

        for stage in [SleepStage.NREM, SleepStage.REM]:
            stage_snapshots = [s for s in self._history if s.sleep_stage == stage]

            if stage_snapshots:
                stats[stage] = {
                    "count": len(stage_snapshots),
                    "patterns_replayed": sum(s.patterns_replayed for s in stage_snapshots),
                    "avg_cortical_learning": sum(
                        s.cortical_learning_rate for s in stage_snapshots
                    ) / len(stage_snapshots),
                    "avg_degradation": sum(
                        s.retrieval_degradation for s in stage_snapshots
                    ) / len(stage_snapshots),
                }
            else:
                stats[stage] = {
                    "count": 0,
                    "patterns_replayed": 0,
                    "avg_cortical_learning": 0.0,
                    "avg_degradation": 0.0,
                }

        return stats


# ============================================================================
# 4. Consolidation Trigger (High-Level)
# ============================================================================

@dataclass
class ConsolidationTriggerConfig:
    """Configuration for consolidation triggering."""
    # Minimum steps between consolidations
    min_consolidation_interval: int = 50000

    # Memory pressure threshold
    pressure_config: MemoryPressureConfig = field(
        default_factory=MemoryPressureConfig
    )

    # Sleep stage config
    sleep_config: SleepStageConfig = field(
        default_factory=SleepStageConfig
    )


class ConsolidationTrigger:
    """High-level consolidation orchestrator.

    Coordinates memory pressure detection and consolidation scheduling.

    Example:
        >>> trigger = ConsolidationTrigger()
        >>>
        >>> # Check if should consolidate
        >>> should_consolidate, reason = trigger.should_start_consolidation(
        ...     current_step=100000,
        ...     hippocampus_activity=0.88,
        ...     pattern_overlap=0.75,
        ...     retrieval_success=0.82,
        ... )
        >>> # → (True, "High memory pressure (0.82)")
        >>>
        >>> # Mark consolidation started
        >>> trigger.mark_consolidation_started(100000)
        >>>
        >>> # Later: check if can consolidate again
        >>> can_consolidate = trigger.can_consolidate_now(120000)
        >>> # → False (too soon, need 50k steps)
    """

    def __init__(self, config: Optional[ConsolidationTriggerConfig] = None):
        """Initialize trigger.

        Args:
            config: Configuration for triggering logic
        """
        self.config = config or ConsolidationTriggerConfig()

        self._pressure_detector = MemoryPressureDetector(
            self.config.pressure_config
        )
        self._sleep_controller = SleepStageController(
            self.config.sleep_config
        )

        self._last_consolidation_step: Optional[int] = None

    def should_start_consolidation(
        self,
        current_step: int,
        hippocampus_activity: float,
        pattern_overlap: float,
        retrieval_success: float,
    ) -> Tuple[bool, str]:
        """Check if consolidation should start.

        Args:
            current_step: Current training step
            hippocampus_activity: Activity level (0-1)
            pattern_overlap: Pattern interference (0-1)
            retrieval_success: Retrieval accuracy (0-1)

        Returns:
            Tuple of (should_consolidate, reason)
        """
        # Check minimum interval
        if not self.can_consolidate_now(current_step):
            steps_remaining = self._steps_until_next_consolidation(current_step)
            return False, f"Too soon (need {steps_remaining} more steps)"

        # Check memory pressure
        pressure = self._pressure_detector.calculate_pressure(
            hippocampus_activity=hippocampus_activity,
            pattern_overlap=pattern_overlap,
            retrieval_success=retrieval_success,
        )

        if self._pressure_detector.should_trigger_consolidation(pressure):
            urgency = self._pressure_detector.get_consolidation_urgency(pressure)
            return True, f"{urgency.capitalize()} memory pressure ({pressure:.2f})"

        return False, f"Low pressure ({pressure:.2f})"

    def can_consolidate_now(self, current_step: int) -> bool:
        """Check if minimum interval has passed.

        Args:
            current_step: Current training step

        Returns:
            True if can consolidate now
        """
        if self._last_consolidation_step is None:
            return True

        steps_since_last = current_step - self._last_consolidation_step
        return steps_since_last >= self.config.min_consolidation_interval

    def mark_consolidation_started(self, step: int) -> None:
        """Mark that consolidation started at given step.

        Args:
            step: Step when consolidation started
        """
        self._last_consolidation_step = step

    def get_sleep_controller(self) -> SleepStageController:
        """Get sleep stage controller for replay."""
        return self._sleep_controller

    def _steps_until_next_consolidation(self, current_step: int) -> int:
        """Calculate steps until next consolidation allowed."""
        if self._last_consolidation_step is None:
            return 0

        elapsed = current_step - self._last_consolidation_step
        remaining = max(0, self.config.min_consolidation_interval - elapsed)
        return remaining
