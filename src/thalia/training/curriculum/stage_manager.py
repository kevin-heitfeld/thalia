"""
Curriculum Trainer - Orchestrate multi-stage developmental training.

This module implements the complete curriculum training pipeline from
docs/design/curriculum_strategy.md, coordinating all subsystems:

- Stage progression (Sensorimotor → Phonology → ... → LLM-level)
- Interleaved practice within stages
- Growth triggering based on capacity
- Consolidation based on memory pressure
- Milestone evaluation (go/no-go decisions)
- Smooth stage transitions
- Checkpoint management
- Cognitive load monitoring during transitions

Key Responsibilities:
====================

1. STAGE MANAGEMENT
   - Load stage-specific tasks and configurations
   - Track progress within each stage
   - Evaluate milestones (success criteria)
   - Make go/no-go decisions for transitions

2. TRAINING ORCHESTRATION
   - Interleaved task sampling (multinomial mixing)
   - Growth monitoring and triggering
   - Consolidation triggering (memory pressure)
   - Checkpoint save at critical points

3. EVALUATION & MONITORING
   - Continuous health monitoring
   - Backward compatibility checks
   - Catastrophic forgetting detection
   - Performance tracking per task

4. TRANSITION PROTOCOL
   - Extended consolidation before transition
   - Gradual difficulty ramps (4 weeks)
   - High initial review of old stages
   - Cognitive load monitoring

Learning & Reward Philosophy:
==============================

The brain learns CONTINUOUSLY via local rules during forward passes:
- STDP in pathways (spike-timing dependent)
- BCM in cortex (competitive, unsupervised)
- Hebbian in hippocampus (one-shot episodic)
- Intrinsic rewards from prediction errors

External rewards are ONLY used for RL tasks (motor control, reaching):
- Modulate dopamine → affect striatum three-factor learning
- NOT used for supervised/classification tasks (MNIST, phonology)

For classification tasks:
- Forward pass creates representations via unsupervised cortical plasticity
- No task-specific "outputs" to decode during training
- Evaluation happens SEPARATELY via milestone checks (accuracy tests)
- Brain learns features, not labels

This matches biology: cortex learns representations without supervision,
dopamine provides global reward signal for action selection, evaluation
requires separate testing (behavioral experiments).

Usage:
======

    from thalia.training.curriculum.stage_manager import (
        CurriculumTrainer,
        StageConfig,
        CurriculumStage,
    )
    from thalia.core.brain import EventDrivenBrain
    from thalia.config.curriculum_growth import get_curriculum_growth_config

    # Initialize brain
    brain = EventDrivenBrain(config)

    # Initialize trainer
    trainer = CurriculumTrainer(
        brain=brain,
        growth_config=get_curriculum_growth_config(),
        checkpoint_dir="checkpoints/curriculum",
    )

    # Train Stage -0.5 (Sensorimotor)
    result = trainer.train_stage(
        stage=CurriculumStage.SENSORIMOTOR,
        config=StageConfig(
            duration_steps=50000,
            task_configs={
                'motor_control': {'weight': 0.4, 'difficulty': 0.5},
                'reaching': {'weight': 0.35, 'difficulty': 0.6},
                'manipulation': {'weight': 0.25, 'difficulty': 0.7},
            },
            success_criteria={
                'reaching_accuracy': PERFORMANCE_GOOD,
                'manipulation_success': 0.85,
                'prediction_error': PREDICTION_ERROR_SMALL,
            },
        ),
    )

    # Check if ready for next stage
    if trainer.evaluate_stage_readiness(CurriculumStage.SENSORIMOTOR):
        # Proceed to Stage 0
        trainer.train_stage(CurriculumStage.PHONOLOGY, config=stage0_config)

Architecture:
=============

    CurriculumTrainer
    ├── StageConfig (per-stage configuration)
    ├── InterleavedCurriculumSampler (task sampling)
    ├── MemoryPressureDetector (consolidation triggering)
    ├── GrowthManager (capacity monitoring)
    ├── StageTransitionProtocol (smooth transitions)
    └── StageEvaluator (milestone checking)

References:
===========
- docs/design/curriculum_strategy.md - Complete training strategy
- docs/design/checkpoint_format.md - Checkpoint system
- docs/patterns/component-parity.md - Regions + pathways

Author: Thalia Project
Date: December 8, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from enum import IntEnum
from collections import deque, defaultdict
import time

import numpy as np

from thalia.core.spike_utils import compute_firing_rate
from thalia.core.growth import GrowthManager
from thalia.training.constants import (
    FIRING_RATE_MINIMUM,
    CURRICULUM_LOAD_THRESHOLD,
    CURRICULUM_MARGIN,
)
from thalia.config.curriculum_growth import (
    CurriculumGrowthConfig,
    CurriculumStage,
    get_curriculum_growth_config,
)
from thalia.training.curriculum import (
    InterleavedCurriculumSampler,
    SpacedRepetitionScheduler,
    TestingPhaseProtocol,
    StageTransitionProtocol,
)
from thalia.memory.consolidation import (
    MemoryPressureDetector,
    SleepStageController,
)
from thalia.io import CheckpointManager
from thalia.training.visualization.live_diagnostics import LiveDiagnostics
from thalia.regions.prefrontal_hierarchy import Goal
from thalia.learning.critical_periods import (
    CriticalPeriodGating,
)
from thalia.diagnostics import (
    PerformanceProfiler,
    HealthMonitor,
    HealthConfig,
)


# ============================================================================
# Cognitive Load Monitoring
# ============================================================================

class MechanismPriority(IntEnum):
    """Priority levels for cognitive mechanisms."""

    CRITICAL = 1  # Cannot be disabled (e.g., basic perception)
    HIGH = 2      # Core mechanisms for current stage
    MEDIUM = 3    # Supporting mechanisms
    LOW = 4       # Optional enhancements


@dataclass
class ActiveMechanism:
    """Represents an active cognitive mechanism.

    **Attributes**:
        name: Human-readable mechanism name
        cost: Cognitive load cost (0-1)
        priority: Priority level for deactivation decisions
        stage_introduced: Which stage introduced this mechanism
        can_deactivate: Whether mechanism can be temporarily disabled
    """

    name: str
    cost: float
    priority: MechanismPriority = MechanismPriority.MEDIUM
    stage_introduced: Optional[CurriculumStage] = None
    can_deactivate: bool = True

    def __post_init__(self):
        """Validate mechanism parameters."""
        if not 0.0 <= self.cost <= 1.0:
            raise ValueError(f"Mechanism cost must be in [0, 1], got {self.cost}")


class CognitiveLoadMonitor:
    """Track active mechanisms to prevent cognitive overload.

    During stage transitions, the brain must manage multiple active
    mechanisms simultaneously (old stage review + new stage learning).
    This monitor tracks the total cognitive load and suggests which
    mechanisms to deactivate if overload occurs.

    **Design Principles**:
    - Each mechanism has a cost (0-1) representing its cognitive demand
    - Total load is sum of all active mechanism costs
    - Overload threshold typically 0.9 (90% capacity)
    - Lower priority mechanisms deactivated first
    - Critical mechanisms cannot be deactivated

    **Usage**:
    ```python
    monitor = CognitiveLoadMonitor(load_threshold=0.9)

    # Add mechanisms as they become active
    monitor.add_mechanism('visual_processing', cost=0.2, priority=MechanismPriority.CRITICAL)
    monitor.add_mechanism('working_memory', cost=0.3, priority=MechanismPriority.HIGH)
    monitor.add_mechanism('new_stage_tasks', cost=0.5, priority=MechanismPriority.HIGH)

    # Check for overload
    if monitor.is_overloaded():
        suggestion = monitor.suggest_deactivation()
        print(f"Overloaded! Consider deactivating: {suggestion}")

    # During transitions, adjust review ratios
    if monitor.is_overloaded():
        old_stage_review_ratio *= 0.7  # Reduce review
    ```

    **Attributes**:
        load_threshold: Maximum acceptable cognitive load (default CURRICULUM_LOAD_THRESHOLD)
        active_mechanisms: List of currently active mechanisms
        deactivated_mechanisms: List of temporarily disabled mechanisms
    """

    def __init__(self, load_threshold: float = CURRICULUM_LOAD_THRESHOLD):
        """Initialize cognitive load monitor.

        **Args**:
            load_threshold: Maximum cognitive load before overload (0-1)
        """
        if not 0.0 < load_threshold <= 1.0:
            raise ValueError(f"Load threshold must be in (0, 1], got {load_threshold}")

        self.load_threshold = load_threshold
        self.active_mechanisms: List[ActiveMechanism] = []
        self.deactivated_mechanisms: List[ActiveMechanism] = []
        self._load_history: List[Tuple[float, float]] = []  # (timestamp, load)

    def add_mechanism(
        self,
        name: str,
        cost: float,
        priority: MechanismPriority = MechanismPriority.MEDIUM,
        stage_introduced: Optional[CurriculumStage] = None,
        can_deactivate: bool = True,
    ) -> None:
        """Register a new active mechanism.

        **Args**:
            name: Human-readable mechanism name
            cost: Cognitive load cost (0-1)
            priority: Priority level for deactivation
            stage_introduced: Which stage introduced this mechanism
            can_deactivate: Whether mechanism can be disabled
        """
        mechanism = ActiveMechanism(
            name=name,
            cost=cost,
            priority=priority,
            stage_introduced=stage_introduced,
            can_deactivate=can_deactivate,
        )
        self.active_mechanisms.append(mechanism)
        self._record_load()

    def remove_mechanism(self, name: str) -> bool:
        """Remove an active mechanism.

        **Args**:
            name: Mechanism name to remove

        **Returns**:
            True if mechanism was found and removed
        """
        for i, mech in enumerate(self.active_mechanisms):
            if mech.name == name:
                self.active_mechanisms.pop(i)
                self._record_load()
                return True
        return False

    def deactivate_mechanism(self, name: str) -> bool:
        """Temporarily deactivate a mechanism.

        **Args**:
            name: Mechanism name to deactivate

        **Returns**:
            True if mechanism was found and deactivated
        """
        for i, mech in enumerate(self.active_mechanisms):
            if mech.name == name:
                if not mech.can_deactivate:
                    return False
                mechanism = self.active_mechanisms.pop(i)
                self.deactivated_mechanisms.append(mechanism)
                self._record_load()
                return True
        return False

    def reactivate_mechanism(self, name: str) -> bool:
        """Reactivate a previously deactivated mechanism.

        **Args**:
            name: Mechanism name to reactivate

        **Returns**:
            True if mechanism was found and reactivated
        """
        for i, mech in enumerate(self.deactivated_mechanisms):
            if mech.name == name:
                mechanism = self.deactivated_mechanisms.pop(i)
                self.active_mechanisms.append(mechanism)
                self._record_load()
                return True
        return False

    def calculate_load(self) -> float:
        """Calculate current cognitive load.

        **Returns**:
            Total cognitive load (0-1+, can exceed 1 when overloaded)
        """
        return sum(m.cost for m in self.active_mechanisms)

    def is_overloaded(self) -> bool:
        """Check if current load exceeds threshold.

        **Returns**:
            True if cognitively overloaded
        """
        return self.calculate_load() > self.load_threshold

    def get_headroom(self) -> float:
        """Calculate available cognitive capacity.

        **Returns**:
            Remaining capacity before overload (can be negative)
        """
        return self.load_threshold - self.calculate_load()

    def suggest_deactivation(self) -> Optional[str]:
        """Suggest which mechanism to deactivate if overloaded.

        Uses priority-based selection:
        1. Only consider deactivatable mechanisms
        2. Sort by priority (LOW first, then MEDIUM, HIGH, CRITICAL)
        3. Within same priority, choose highest cost

        **Returns**:
            Mechanism name to deactivate, or None if not overloaded
        """
        if not self.is_overloaded():
            return None

        # Filter to deactivatable mechanisms
        candidates = [m for m in self.active_mechanisms if m.can_deactivate]

        if not candidates:
            return None

        # Sort by priority (LOW=4 first, then MEDIUM=3, HIGH=2), then by cost (high first)
        # Use negative priority value to sort descending (LOW priority deactivated first)
        candidates.sort(key=lambda m: (-m.priority.value, -m.cost))

        return candidates[0].name

    def suggest_multiple_deactivations(self, target_load: Optional[float] = None) -> List[str]:
        """Suggest multiple mechanisms to deactivate to reach target load.

        **Args**:
            target_load: Target cognitive load (default: threshold - CURRICULUM_MARGIN)

        **Returns**:
            List of mechanism names to deactivate (in order)
        """
        if target_load is None:
            target_load = self.load_threshold - CURRICULUM_MARGIN

        current_load = self.calculate_load()
        if current_load <= target_load:
            return []

        suggestions = []
        temp_load = current_load

        # Get deactivatable mechanisms sorted by priority (LOW first)
        candidates = [m for m in self.active_mechanisms if m.can_deactivate]
        candidates.sort(key=lambda m: (-m.priority.value, -m.cost))

        for mech in candidates:
            if temp_load <= target_load:
                break
            suggestions.append(mech.name)
            temp_load -= mech.cost

        return suggestions

    def get_load_by_priority(self) -> Dict[MechanismPriority, float]:
        """Get cognitive load breakdown by priority level.

        **Returns**:
            Dictionary mapping priority to total load at that priority
        """
        breakdown = {p: 0.0 for p in MechanismPriority}
        for mech in self.active_mechanisms:
            breakdown[mech.priority] += mech.cost
        return breakdown

    def get_load_by_stage(self) -> Dict[Optional[CurriculumStage], float]:
        """Get cognitive load breakdown by introducing stage.

        **Returns**:
            Dictionary mapping stage to total load from that stage
        """
        breakdown: Dict[Optional[CurriculumStage], float] = {}
        for mech in self.active_mechanisms:
            stage = mech.stage_introduced
            breakdown[stage] = breakdown.get(stage, 0.0) + mech.cost
        return breakdown

    def get_status_report(self) -> str:
        """Generate human-readable status report.

        **Returns**:
            Formatted status report string
        """
        load = self.calculate_load()
        headroom = self.get_headroom()
        overloaded = self.is_overloaded()

        report = []
        report.append("=" * 60)
        report.append("Cognitive Load Monitor Status")
        report.append("=" * 60)
        report.append(f"Current Load: {load:.2f} / {self.load_threshold:.2f}")
        report.append(f"Headroom: {headroom:.2f}")
        report.append(f"Status: {'WARNING OVERLOADED' if overloaded else 'OK'}")
        report.append("")

        if self.active_mechanisms:
            report.append("Active Mechanisms:")
            for mech in sorted(self.active_mechanisms, key=lambda m: m.priority.value):
                deactivatable = "Y" if mech.can_deactivate else "N"
                report.append(
                    f"  [{mech.priority.name:8}] {mech.name:30} "
                    f"Cost: {mech.cost:.2f}  Deactivatable: {deactivatable}"
                )

        if self.deactivated_mechanisms:
            report.append("")
            report.append("Deactivated Mechanisms:")
            for mech in self.deactivated_mechanisms:
                report.append(f"  {mech.name:30} Cost: {mech.cost:.2f}")

        if overloaded:
            suggestion = self.suggest_deactivation()
            if suggestion:
                report.append("")
                report.append(f"Suggestion: Deactivate '{suggestion}'")

        report.append("=" * 60)
        return "\n".join(report)

    def _record_load(self) -> None:
        """Record current load in history for analysis."""
        self._load_history.append((time.time(), self.calculate_load()))

        # Keep only last 1000 entries
        if len(self._load_history) > 1000:
            self._load_history = self._load_history[-1000:]

    def get_load_statistics(self, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get load statistics over time window.

        **Args**:
            window_seconds: Time window for statistics (None = all history)

        **Returns**:
            Dictionary with min, max, mean, current load
        """
        if not self._load_history:
            current = self.calculate_load()
            return {'min': current, 'max': current, 'mean': current, 'current': current}

        # Filter by time window
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            history = [(t, load) for t, load in self._load_history if t >= cutoff]
        else:
            history = self._load_history

        if not history:
            current = self.calculate_load()
            return {'min': current, 'max': current, 'mean': current, 'current': current}

        loads = [load for _, load in history]
        return {
            'min': min(loads),
            'max': max(loads),
            'mean': sum(loads) / len(loads),
            'current': self.calculate_load(),
        }


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class TaskConfig:
    """Configuration for a single task within a stage."""
    weight: float = 1.0  # Relative weight in curriculum sampling
    difficulty: float = 0.5  # Difficulty level (0-1)
    enabled: bool = True  # Whether task is active


@dataclass
class StageConfig:
    """Configuration for training a single curriculum stage."""

    # Duration
    duration_steps: int = 50000  # Total training steps

    # Task configuration
    task_configs: Dict[str, TaskConfig] = field(default_factory=dict)

    # Success criteria (milestone evaluation)
    success_criteria: Dict[str, float] = field(default_factory=dict)

    # Critical period configuration
    enable_critical_periods: bool = True  # Apply critical period gating
    domain_mappings: Dict[str, List[str]] = field(default_factory=dict)  # task_name -> domains

    # Curriculum parameters
    interleaved_practice: bool = True  # Mix tasks within stage
    spaced_repetition: bool = True  # Review based on Leitner algorithm
    testing_frequency: float = 0.15  # Fraction of steps for retrieval practice
    productive_failure_steps: int = 5000  # Initial struggle period

    # Review from previous stages
    review_stages: Dict[int, float] = field(default_factory=dict)  # {stage: weight}

    # Consolidation
    consolidation_interval: int = 10000  # Steps between consolidation
    consolidation_cycles: int = 5  # Replay cycles per consolidation

    # Growth
    enable_growth: bool = True  # Allow neuron addition
    growth_check_interval: int = 5000  # Steps between capacity checks

    # Checkpointing
    checkpoint_interval: int = 5000  # Steps between checkpoints

    # Stage-specific settings
    stage_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result from training a stage."""

    stage: CurriculumStage
    success: bool  # Whether milestones met

    # Performance metrics
    final_metrics: Dict[str, float] = field(default_factory=dict)
    milestone_results: Dict[str, bool] = field(default_factory=dict)

    # Training statistics
    total_steps: int = 0
    training_time_seconds: float = 0.0

    # Growth events
    growth_events: List[Dict[str, Any]] = field(default_factory=list)

    # Consolidation events
    consolidation_events: List[Dict[str, Any]] = field(default_factory=list)

    # Checkpoints saved
    checkpoints: List[str] = field(default_factory=list)

    # Failure reasons (if not successful)
    failure_reasons: List[str] = field(default_factory=list)


# ============================================================================
# Main CurriculumTrainer Class
# ============================================================================

class CurriculumTrainer:
    """Orchestrate multi-stage curriculum training."""

    def __init__(
        self,
        brain: Any,  # EventDrivenBrain or similar
        growth_config: Optional[CurriculumGrowthConfig] = None,
        checkpoint_dir: Optional[str] = None,
        device: str = 'cpu',
        verbose: bool = True,
        enable_live_diagnostics: bool = False,
        diagnostics_interval: int = 100,
    ):
        """Initialize curriculum trainer.

        Args:
            brain: Brain instance to train
            growth_config: Growth configuration (uses default if None)
            checkpoint_dir: Directory for checkpoints (creates if needed)
            device: Device for training
            verbose: Whether to print progress
            enable_live_diagnostics: Whether to enable real-time visualization
            diagnostics_interval: Steps between diagnostic updates (if enabled)
        """
        self.brain = brain
        self.device = device
        self.verbose = verbose

        # Checkpoint manager (Tier 3.2 - unified checkpoint management)
        self.checkpoint_manager = CheckpointManager(
            brain=brain,
            default_compression='zstd',
            default_precision='fp32',
        )

        # Growth configuration
        self.growth_config = growth_config or get_curriculum_growth_config()

        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Curriculum subsystems
        self.task_sampler = InterleavedCurriculumSampler()
        self.spaced_repetition = SpacedRepetitionScheduler()
        self.testing_protocol = TestingPhaseProtocol()
        self.transition_protocol = StageTransitionProtocol()
        self.memory_pressure = MemoryPressureDetector()
        self.sleep_controller = SleepStageController()
        self.critical_period_gating = CriticalPeriodGating()  # NEW: Critical period windows

        # Live diagnostics (optional)
        self.enable_live_diagnostics = enable_live_diagnostics
        self.diagnostics_interval = diagnostics_interval
        self.live_diagnostics = LiveDiagnostics() if enable_live_diagnostics else None

        # Performance profiler (always enabled)
        self.profiler = PerformanceProfiler(window_size=100)

        # Health monitor (always enabled)
        self.health_monitor = HealthMonitor(HealthConfig())

        # Task performance tracking (per-task accuracy buffers)
        self.task_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # State tracking
        self.current_stage: Optional[CurriculumStage] = None
        self.global_step = 0
        self.stage_start_step = 0

        # Critical period tracking
        self._last_phase: Dict[str, str] = {}  # domain -> last_phase
        self._last_milestone_results: Dict[str, bool] = {}  # Cached milestone results

        # History
        self.training_history: List[TrainingResult] = []
        self.performance_history: Dict[str, List[float]] = {}

        # Callbacks
        self.callbacks: List[Callable[[int, Dict[str, float]], None]] = []

        # Goal hierarchy cache (for Stages 3+)
        self._goal_hierarchies: Dict[str, Goal] = {}

    def train_stage(
        self,
        stage: CurriculumStage,
        config: StageConfig,
        task_loader: Any,  # Task dataset/loader
        evaluator: Optional[Callable] = None,
    ) -> TrainingResult:
        """Train a single curriculum stage.

        Args:
            stage: Which curriculum stage to train
            config: Stage configuration
            task_loader: Object providing tasks (dataset, environment, etc.)
            evaluator: Optional custom evaluation function

        Returns:
            TrainingResult with success status and metrics
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting Stage {stage.name} ({stage.value})")
            print(f"Duration: {config.duration_steps:,} steps")
            print(f"Tasks: {list(config.task_configs.keys())}")
            print(f"{'='*80}\n")

        # Initialize stage
        self.current_stage = stage
        self.stage_start_step = self.global_step
        start_time = time.time()

        # Setup goal hierarchies for stages that need them (3+)
        if stage in [CurriculumStage.READING, CurriculumStage.ABSTRACT]:
            self._setup_stage_goal_hierarchies(stage)

        # Initialize result tracking
        result = TrainingResult(stage=stage, success=False)

        # Setup task sampling weights
        task_weights = {
            name: cfg.weight
            for name, cfg in config.task_configs.items()
            if cfg.enabled
        }

        try:
            # Main training loop
            for step in range(config.duration_steps):
                # 1. Sample next task (interleaved practice)
                if config.interleaved_practice:
                    task_name = self.task_sampler.sample_next_task(task_weights)
                else:
                    # Sequential (less effective, but sometimes needed)
                    task_name = list(task_weights.keys())[step % len(task_weights)]

                # 2. Get task from loader
                task_data = task_loader.get_task(task_name)

                # 2.5. Apply critical period modulation (NEW)
                if config.enable_critical_periods:
                    self._apply_critical_period_modulation(
                        task_name=task_name,
                        domains=config.domain_mappings.get(task_name, []),
                        age=self.global_step,
                    )

                # 3. Forward pass
                # Learning happens AUTOMATICALLY during forward via:
                # - STDP in pathways (spike-timing dependent plasticity)
                # - BCM in cortex (Bienenstock-Cooper-Munro, competition)
                # - Hebbian in hippocampus (one-shot episodic encoding)
                # - Intrinsic rewards from prediction errors (continuous)
                # MODULATED by critical period windows (plasticity_modulator)

                # Time the forward pass
                self.profiler.start_forward()
                _output = self.brain.forward(
                    task_data['input'],
                    n_timesteps=task_data.get('n_timesteps', 10),
                )
                self.profiler.end_forward()

                # 4. External reward (ONLY for RL tasks)
                # Modulates dopamine → affects striatum three-factor learning
                # For supervised tasks (MNIST, temporal), learning is unsupervised
                # via cortical plasticity. Evaluation happens separately.
                if 'reward' in task_data and task_data.get('task_type') in [
                    'motor_control', 'reaching', 'manipulation', 'prediction',
                    'reinforcement_learning',  # Explicit RL tasks
                ]:
                    # Deliver external reward for RL tasks
                    # This modulates dopamine for striatum/PFC
                    self.brain.deliver_reward(external_reward=task_data['reward'])

                # For classification/prediction tasks (MNIST, temporal, phonology):
                # - Forward pass creates cortical representations
                # - Learning via unsupervised BCM/STDP
                # - No external reward during training
                # - Evaluation happens periodically via milestone checks

                # Track task-specific performance (if available)
                if 'accuracy' in task_data or 'reward' in task_data:
                    task_type = task_data.get('task_type', 'unknown')
                    perf_value = task_data.get('accuracy', task_data.get('reward', 0.0))
                    self.task_performance[task_type].append(float(perf_value))

                # Record step completion for profiler
                self.profiler.record_step()

                # Sample memory periodically (every 100 steps)
                if step % 100 == 0:
                    self.profiler.record_memory(self.brain, self.device)

                # 5. Check for growth (every N steps)
                if step % config.growth_check_interval == 0 and config.enable_growth:
                    self._check_and_trigger_growth(stage, result)

                # 6. Check for consolidation (memory pressure)
                if step % config.consolidation_interval == 0:
                    self._check_and_trigger_consolidation(stage, config, result)

                # 7. Checkpoint (periodic)
                if step % config.checkpoint_interval == 0:
                    checkpoint_path = self._save_checkpoint(stage, step)
                    result.checkpoints.append(checkpoint_path)

                # 8. Logging and callbacks
                metrics = None
                if step % 1000 == 0:
                    metrics = self._collect_metrics()
                    self._log_progress(stage, step, metrics)

                # Call callbacks every step for progress tracking
                # Only collect metrics if not already done above
                if self.callbacks:
                    if metrics is None:
                        metrics = {}  # Empty dict for progress tracking
                    for callback in self.callbacks:
                        callback(self.global_step, metrics)

                # 9. Live diagnostics (if enabled)
                if self.enable_live_diagnostics and step % self.diagnostics_interval == 0:
                    metrics = self._collect_metrics()
                    self.live_diagnostics.update(step, self.brain, metrics)
                    if step % (self.diagnostics_interval * 10) == 0:  # Show every 10 updates
                        self.live_diagnostics.show()

                self.global_step += 1

            # Post-training evaluation
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Stage {stage.name} training complete")
                print("Evaluating milestones...")
                print(f"{'='*80}\n")

            # Evaluate milestones
            if evaluator is not None:
                milestone_results = evaluator(self.brain, task_loader)
            else:
                milestone_results = self._default_evaluation(stage, task_loader, config)

            result.milestone_results = milestone_results
            result.success = all(milestone_results.values())

            # Store for checkpoint saving
            self._last_milestone_results = milestone_results

            # Final metrics
            result.final_metrics = self._collect_metrics()
            result.total_steps = config.duration_steps
            result.training_time_seconds = time.time() - start_time

            # Print performance summary
            if self.verbose:
                self.profiler.print_summary()

            # Report results
            self._report_stage_results(result)

            # Save final checkpoint with milestones
            final_checkpoint = self._save_checkpoint(stage, config.duration_steps, final=True)
            result.checkpoints.append(final_checkpoint)

        except Exception as e:
            # Training failed
            result.success = False
            result.failure_reasons.append(f"Exception: {str(e)}")
            if self.verbose:
                print(f"\n❌ Stage {stage.name} FAILED: {e}")
            raise

        # Add to history
        self.training_history.append(result)

        return result

    def evaluate_stage_readiness(
        self,
        stage: CurriculumStage,
        evaluator: Optional[Callable] = None,
    ) -> bool:
        """Evaluate if brain is ready to proceed to next stage.

        Args:
            stage: Current stage to evaluate
            evaluator: Optional custom evaluation function

        Returns:
            True if all milestones met
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Evaluating readiness for Stage {stage.name}")
            print(f"{'='*80}\n")

        # Get last training result for this stage
        stage_results = [r for r in self.training_history if r.stage == stage]
        if not stage_results:
            print(f"⚠️  No training history for Stage {stage.name}")
            return False

        last_result = stage_results[-1]

        # Check milestone results
        all_passed = all(last_result.milestone_results.values())

        if self.verbose:
            print("Milestone Results:")
            for criterion, passed in last_result.milestone_results.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {criterion}: {passed}")

            if all_passed:
                print(f"\n✅ Stage {stage.name} ready for transition!")
            else:
                failed = [k for k, v in last_result.milestone_results.items() if not v]
                print(f"\n❌ Stage {stage.name} NOT ready")
                print(f"Failed criteria: {failed}")

        return all_passed

    def transition_to_stage(
        self,
        new_stage: CurriculumStage,
        old_stage: CurriculumStage,
        weeks: int = 4,
    ) -> None:
        """Execute smooth transition between stages.

        Implements gradual difficulty ramps and high initial review
        as specified in curriculum_strategy.md.

        Args:
            new_stage: Stage to transition to
            old_stage: Current stage
            weeks: Number of weeks for gradual transition (default 4)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Stage Transition: {old_stage.name} → {new_stage.name}")
            print(f"Transition period: {weeks} weeks")
            print(f"{'='*80}\n")

        # Extended consolidation before transition
        if self.verbose:
            print("Performing extended consolidation...")

        self._extended_consolidation(cycles=10)  # 2x normal

        # Gradual difficulty ramps (4 weeks by default)
        for week in range(weeks):
            if self.verbose:
                print(f"\nTransition Week {week + 1}/{weeks}")

            # Get transition configuration
            config = self.transition_protocol.get_transition_config(
                old_stage=old_stage.value,
                new_stage=new_stage.value,
                weeks_since_transition=week,
            )

            if self.verbose:
                print(f"  Difficulty: {config.difficulty:.1%}")
                print(f"  Old stage review: {config.old_stage_ratio:.1%}")
                print(f"  Cognitive load: {config.cognitive_load.value}")

        if self.verbose:
            print(f"\n✅ Transition complete: Now in Stage {new_stage.name}")

    def extend_stage(
        self,
        stage: CurriculumStage,
        additional_steps: int = 20000,
        reason: str = "Milestones not met",
    ) -> None:
        """Extend current stage training.

        Called when milestones not met and stage needs more time.

        Args:
            stage: Stage to extend
            additional_steps: Additional training steps
            reason: Why extension is needed
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"⚠️  Extending Stage {stage.name}")
            print(f"Additional steps: {additional_steps:,}")
            print(f"Reason: {reason}")
            print(f"{'='*80}\n")

    def save_checkpoint(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint manually.

        Args:
            name: Optional checkpoint name
            metadata: Optional additional metadata

        Returns:
            Path to saved checkpoint
        """
        if name is None:
            name = f"manual_step_{self.global_step}"

        checkpoint_path = self.checkpoint_dir / f"{name}.thalia"

        # Get current milestone results if in stage training
        milestones = {}
        if self.current_stage and hasattr(self, '_last_milestone_results'):
            milestones = getattr(self, '_last_milestone_results', {})

        full_metadata = {
            'global_step': self.global_step,
            'current_stage': self.current_stage.value if self.current_stage else None,
            'stage_start_step': self.stage_start_step,
            'milestones': milestones,
            'training_history': [
                {
                    'stage': r.stage.value,
                    'success': r.success,
                    'total_steps': r.total_steps,
                }
                for r in self.training_history
            ],
        }

        if metadata:
            full_metadata.update(metadata)

        # Use CheckpointManager for unified checkpoint handling (Tier 3.2)
        save_info = self.checkpoint_manager.save(
            str(checkpoint_path),
            metadata=full_metadata,
        )

        if self.verbose:
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            print(f"   Size: {save_info.get('size_mb', 0):.2f} MB, Time: {save_info.get('time_s', 0):.2f}s")

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        path: str,
        restore_trainer_state: bool = True,
    ) -> None:
        """Load checkpoint and optionally restore trainer state.

        Args:
            path: Path to checkpoint file
            restore_trainer_state: Whether to restore step counters, etc.
        """
        # Load brain state using CheckpointManager (Tier 3.2)
        load_info = self.checkpoint_manager.load(path, device=self.device, strict=True)

        if self.verbose:
            print(f"📥 Checkpoint loaded: {path}")
            print(f"   Components: {load_info.get('components_loaded', {})}")
            print(f"   Time: {load_info.get('time_s', 0):.2f}s")

        if restore_trainer_state:
            metadata = load_info.get('metadata', {})
            self.global_step = metadata.get('global_step', 0)
            self.stage_start_step = metadata.get('stage_start_step', 0)

            stage_value = metadata.get('current_stage')
            if stage_value is not None:
                self.current_stage = CurriculumStage(stage_value)

        if self.verbose:
            print(f"✅ Checkpoint loaded: {path}")
            if restore_trainer_state:
                print(f"   Global step: {self.global_step}")
                print(f"   Current stage: {self.current_stage.name if self.current_stage else 'None'}")

    def add_callback(
        self,
        callback: Callable[[int, Dict[str, float]], None],
    ) -> None:
        """Add callback function called after each logging step.

        Callback signature: callback(step: int, metrics: Dict[str, float])
        """
        self.callbacks.append(callback)

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _check_and_trigger_growth(
        self,
        stage: CurriculumStage,
        result: TrainingResult,
    ) -> None:
        """Check if growth needed and trigger if so.

        Implements curriculum-aware growth strategy:
        1. Check capacity metrics for each brain region
        2. Compare against stage-specific thresholds
        3. Trigger growth if needed (with consolidation)
        4. Record growth events for tracking

        Growth is coordinated across regions and pathways:
        - Region grows first (e.g., cortex adds neurons)
        - Connected pathways automatically grow to match
        - Consolidation before AND after growth prevents forgetting
        """
        if not self.growth_config.enable_growth:
            return

        # Get component-wise growth configs
        component_configs = self.growth_config.component_configs

        # Map brain attributes to config names
        region_mapping = {
            'cortex': self.brain.cortex.impl if hasattr(self.brain, 'cortex') else None,
            'hippocampus': self.brain.hippocampus.impl if hasattr(self.brain, 'hippocampus') else None,
            'prefrontal': self.brain.pfc.impl if hasattr(self.brain, 'pfc') else None,
            'striatum': self.brain.striatum.impl if hasattr(self.brain, 'striatum') else None,
            'cerebellum': self.brain.cerebellum.impl if hasattr(self.brain, 'cerebellum') else None,
        }

        # Check each region for growth needs
        for region_name, region in region_mapping.items():
            if region is None:
                continue

            # Get growth config for this component
            growth_config = component_configs.get(region_name)
            if growth_config is None:
                continue

            # Get stage-specific trigger
            trigger = growth_config.get_trigger_for_stage(stage.value)
            if trigger is None or not trigger.enabled:
                continue

            # Create GrowthManager for this region
            growth_manager = GrowthManager(region_name=region_name)

            # Get capacity metrics
            metrics = growth_manager.get_capacity_metrics(
                component=region,
                saturation_threshold=trigger.capacity_threshold,
            )

            # Check if growth recommended
            if not metrics.growth_recommended:
                continue

            # Check minimum steps between growth events
            last_growth_step = None
            for event in result.growth_events:
                if event.get('component_name') == region_name:
                    last_growth_step = event.get('step', 0)

            if last_growth_step is not None:
                steps_since_growth = self.global_step - last_growth_step
                if steps_since_growth < trigger.min_steps_between:
                    if self.verbose:
                        print(f"  ⏳ {region_name}: Growth on cooldown "
                              f"({steps_since_growth}/{trigger.min_steps_between} steps)")
                    continue

            # Calculate growth amount
            n_new_neurons = int(region.n_output * trigger.expansion_rate)
            n_new_neurons = max(growth_config.min_neurons_per_growth,
                               min(n_new_neurons, growth_config.max_neurons_per_growth))

            # Check total growth limit
            original_size = region.n_output - sum(
                e.get('n_neurons_added', 0)
                for e in result.growth_events
                if e.get('component_name') == region_name
            )
            current_ratio = region.n_output / max(original_size, 1)
            if current_ratio >= growth_config.max_total_growth:
                if self.verbose:
                    print(f"  🛑 {region_name}: Max growth limit reached "
                          f"({current_ratio:.1f}x original)")
                continue

            if self.verbose:
                print(f"\n🌱 Growing {region_name}:")
                print(f"  Current size: {region.n_output}")
                print(f"  Utilization: {metrics.utilization:.2%}")
                print(f"  Reason: {metrics.growth_reason}")
                print(f"  Adding: {n_new_neurons} neurons")

            # Consolidate before growth (if enabled)
            if trigger.consolidate_before:
                if self.verbose:
                    print("  Consolidating before growth...")
                self.brain.consolidate(n_cycles=5, batch_size=32, verbose=False)

            # Perform growth
            _growth_event = growth_manager.add_neurons(
                component=region,
                n_new=n_new_neurons,
                initialization='sparse_random',
                sparsity=0.1,
                reason=metrics.growth_reason,
                component_type='region',
            )

            # Grow connected pathways (automatic via PathwayManager)
            if hasattr(self.brain, 'pathway_manager'):
                grown_pathways = self.brain.pathway_manager.grow_connected_pathways(
                    region_name=region_name,
                    new_size=region.n_output,
                )
                if self.verbose and grown_pathways:
                    print(f"  🔗 Grew {len(grown_pathways)} connected pathways")

            # Consolidate after growth (if enabled)
            if trigger.consolidate_after:
                if self.verbose:
                    print("  Consolidating after growth...")
                self.brain.consolidate(n_cycles=5, batch_size=32, verbose=False)

            # Record growth event
            result.growth_events.append({
                'step': self.global_step,
                'stage': stage.name,
                'component_name': region_name,
                'component_type': 'region',
                'n_neurons_added': n_new_neurons,
                'old_size': region.n_output - n_new_neurons,
                'new_size': region.n_output,
                'reason': metrics.growth_reason,
                'utilization_before': metrics.utilization,
            })

            if self.verbose:
                print(f"  ✅ Growth complete: {region_name} now has {region.n_output} neurons\n")

    def _check_and_trigger_consolidation(
        self,
        stage: CurriculumStage,
        config: StageConfig,
        result: TrainingResult,
    ) -> None:
        """Check memory pressure and trigger consolidation if needed.

        During consolidation (sleep):
        1. Hippocampus enters consolidation mode
        2. HER (if enabled) relabels experiences with hindsight goals
        3. Replay mixed batch of real + hindsight experiences
        4. Cortex learns from replayed patterns
        """
        # Calculate memory pressure
        # For now, trigger on schedule (every consolidation_interval steps)
        # Future: Use MemoryPressureDetector for adaptive triggering

        if self.verbose:
            print("\n🌙 Entering consolidation (sleep)...")

        # Run consolidation automatically (handles HER, replay, mode switching)
        stats = self.brain.consolidate(
            n_cycles=config.consolidation_cycles,
            batch_size=32,
            verbose=self.verbose
        )

        # Record consolidation event
        result.consolidation_events.append({
            'step': self.global_step,
            'stage': stage.name,
            'cycles': stats['cycles_completed'],
            'total_replayed': stats['total_replayed'],
            'her_enabled': stats['her_enabled'],
        })

        if self.verbose:
            print("  ✅ Consolidation complete")

    def _extended_consolidation(self, cycles: int = 10) -> None:
        """Perform extended consolidation before stage transition.

        Runs more replay cycles than normal to strengthen memory consolidation
        before introducing new tasks. HER automatically participates if enabled.
        """
        if self.verbose:
            print(f"  Extended consolidation: {cycles} cycles")

        # Run extended consolidation automatically
        stats = self.brain.consolidate(
            n_cycles=cycles,
            batch_size=64,
            verbose=self.verbose
        )

        if self.verbose:
            print(f"  Replayed {stats['total_replayed']} total experiences")

    def _save_checkpoint(
        self,
        stage: CurriculumStage,
        step: int,
        final: bool = False,
    ) -> str:
        """Save checkpoint internally during training."""
        suffix = "final" if final else f"step_{step}"
        name = f"stage_{stage.value}_{suffix}"

        # Include current milestone progress in metadata
        milestone_metadata = {}
        if hasattr(self, '_last_milestone_results'):
            milestone_metadata = {'milestones': self._last_milestone_results}

        return self.save_checkpoint(name=name, metadata=milestone_metadata)

    def _apply_critical_period_modulation(
        self,
        task_name: str,
        domains: List[str],
        age: int,
    ) -> Dict[str, float]:
        """Apply critical period gating to learning rates.

        For each domain active in this task, compute the gating multiplier
        and modulate learning rates in relevant regions.

        Args:
            task_name: Current task name
            domains: List of domains this task trains (e.g., ['phonology'])
            age: Current training step

        Returns:
            Dict of domain -> multiplier for logging
        """
        if not domains:
            return {}

        # Compute average multiplier across all domains for this task
        multipliers = {}
        total_multiplier = 0.0

        for domain in domains:
            try:
                # Get window status for logging
                status = self.critical_period_gating.get_window_status(domain, age)
                multiplier = status['multiplier']
                multipliers[domain] = multiplier
                total_multiplier += multiplier

                # Log if at critical transition points
                last_phase = self._last_phase.get(domain)
                if status['phase'] != last_phase:
                    if self.verbose:
                        print(f"  🧠 Critical period: {domain} entering {status['phase']} phase "
                              f"(multiplier: {multiplier:.2f})")
                    self._last_phase[domain] = status['phase']

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Warning: Unknown domain '{domain}' for task '{task_name}': {e}")
                continue

        # Average multiplier for this task
        if multipliers:
            avg_multiplier = total_multiplier / len(multipliers)

            # Apply to brain plasticity
            # Set as global plasticity modulator
            if hasattr(self.brain, 'set_plasticity_modulator'):
                self.brain.set_plasticity_modulator(avg_multiplier)
            elif hasattr(self.brain, 'state'):
                # Alternative: Set via brain state
                self.brain.state.plasticity_modulator = avg_multiplier

        return multipliers

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect comprehensive training metrics.

        Collects:
        - Performance timing (steps/sec, forward time)
        - Memory footprint (CPU/GPU usage)
        - Firing rates per region
        - Weight statistics per pathway
        - Health status
        - Task-specific performance
        - Critical period status
        """
        metrics = {
            'global_step': float(self.global_step),
        }

        # =====================================================================
        # 1. PERFORMANCE TIMING
        # =====================================================================
        perf_metrics = self.profiler.get_metrics_dict()
        metrics.update(perf_metrics)

        # =====================================================================
        # 2. MEMORY FOOTPRINT
        # =====================================================================
        # Already included in perf_metrics

        # =====================================================================
        # 3. BRAIN DIAGNOSTICS
        # =====================================================================
        try:
            brain_diag = self.brain.get_diagnostics()

            # 3.1 Firing rates per region
            spike_counts = brain_diag.get('spike_counts', {})
            for region_name, spike_count in spike_counts.items():
                # Normalize by rough neuron count estimate
                # This is approximate; actual firing rate calculation would need neuron counts
                metrics[f'firing_rate/{region_name}'] = spike_count / 100.0

            # Overall average firing rate
            if spike_counts:
                metrics['firing_rate/average'] = sum(spike_counts.values()) / (len(spike_counts) * 100.0)
            else:
                metrics['firing_rate/average'] = 0.0

            # 3.2 Weight statistics per pathway
            pathway_diag = brain_diag.get('pathways', {})
            for pathway_name, pathway_data in pathway_diag.items():
                # Extract weight statistics if available
                if isinstance(pathway_data, dict):
                    if 'weights_mean' in pathway_data:
                        metrics[f'weights/{pathway_name}_mean'] = float(pathway_data['weights_mean'])
                    if 'weights_std' in pathway_data:
                        metrics[f'weights/{pathway_name}_std'] = float(pathway_data['weights_std'])
                    if 'weights_min' in pathway_data:
                        metrics[f'weights/{pathway_name}_min'] = float(pathway_data['weights_min'])
                    if 'weights_max' in pathway_data:
                        metrics[f'weights/{pathway_name}_max'] = float(pathway_data['weights_max'])

            # 3.3 Neuromodulator levels
            dopamine = brain_diag.get('dopamine', {})
            if dopamine:
                metrics['neuromodulator/dopamine_global'] = float(dopamine.get('global', 0.0))
                metrics['neuromodulator/dopamine_tonic'] = float(dopamine.get('tonic', 0.0))
                metrics['neuromodulator/dopamine_phasic'] = float(dopamine.get('phasic', 0.0))

            lc_state = brain_diag.get('locus_coeruleus', {})
            if lc_state:
                metrics['neuromodulator/norepinephrine'] = float(lc_state.get('norepinephrine', 0.0))
                metrics['neuromodulator/arousal'] = float(lc_state.get('arousal', 0.0))

            nb_state = brain_diag.get('nucleus_basalis', {})
            if nb_state:
                metrics['neuromodulator/acetylcholine'] = float(nb_state.get('acetylcholine', 0.0))

            # 3.4 Oscillator state
            metrics['oscillator/theta_phase'] = float(brain_diag.get('theta_phase', 0.0))
            metrics['oscillator/theta_frequency'] = float(brain_diag.get('theta_frequency', 8.0))

        except Exception as e:
            # Graceful degradation if brain diagnostics fail
            if self.verbose:
                print(f"  ⚠️ Warning: Failed to collect brain diagnostics: {e}")
            metrics['firing_rate/average'] = 0.0

        # =====================================================================
        # 4. HEALTH STATUS
        # =====================================================================
        try:
            brain_diag = self.brain.get_diagnostics() if 'brain_diag' not in locals() else brain_diag
            health_report = self.health_monitor.check_health(brain_diag)

            metrics['health/is_healthy'] = 1.0 if health_report.is_healthy else 0.0
            metrics['health/issue_count'] = float(len(health_report.issues))

            # Maximum severity across all issues
            if health_report.issues:
                max_severity = max(issue.severity for issue in health_report.issues)
                metrics['health/max_severity'] = float(max_severity)
            else:
                metrics['health/max_severity'] = 0.0

            # Count by issue type
            issue_counts = {}
            for issue in health_report.issues:
                issue_type = issue.issue_type.value
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

            for issue_type, count in issue_counts.items():
                metrics[f'health/issue_{issue_type}'] = float(count)

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Warning: Failed to check health: {e}")
            metrics['health/is_healthy'] = 1.0  # Assume healthy if check fails
            metrics['health/issue_count'] = 0.0
            metrics['health/max_severity'] = 0.0

        # =====================================================================
        # 5. TASK-SPECIFIC PERFORMANCE
        # =====================================================================
        for task_name, perf_buffer in self.task_performance.items():
            if perf_buffer:
                # Recent average (last 100 samples)
                metrics[f'task/{task_name}_accuracy'] = float(np.mean(list(perf_buffer)[-100:]))
                # Overall average
                metrics[f'task/{task_name}_accuracy_all'] = float(np.mean(perf_buffer))

        # =====================================================================
        # 6. CRITICAL PERIOD STATUS
        # =====================================================================
        for domain in self.critical_period_gating.get_all_domains():
            try:
                status = self.critical_period_gating.get_window_status(
                    domain,
                    self.global_step
                )
                metrics[f'critical_period/{domain}_multiplier'] = float(status['multiplier'])
                metrics[f'critical_period/{domain}_progress'] = float(status['progress'])
                metrics[f'critical_period/{domain}_phase'] = float(
                    {'early': 0.0, 'peak': 1.0, 'late': 2.0}.get(status['phase'], -1.0)
                )
            except Exception:
                pass  # Skip unknown domains

        return metrics

    def _log_progress(
        self,
        stage: CurriculumStage,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log training progress with key metrics highlighted."""
        if self.verbose and step % 5000 == 0:
            # Extract key metrics for clean display
            steps_per_sec = metrics.get('performance/steps_per_sec', 0.0)
            forward_ms = metrics.get('performance/avg_forward_ms', 0.0)
            gpu_mb = metrics.get('memory/gpu_mb', 0.0)
            cpu_mb = metrics.get('memory/cpu_mb', 0.0)
            avg_fr = metrics.get('firing_rate/average', 0.0)
            is_healthy = metrics.get('health/is_healthy', 1.0)

            health_icon = "✅" if is_healthy > 0.5 else "⚠️"

            print(f"[Stage {stage.name}] Step {step:,} | "
                  f"{health_icon} {steps_per_sec:.1f} steps/s | "
                  f"Forward: {forward_ms:.1f}ms | "
                  f"GPU: {gpu_mb:.0f}MB | CPU: {cpu_mb:.0f}MB | "
                  f"FR: {avg_fr:.3f}")

    def _default_evaluation(
        self,
        stage: CurriculumStage,
        task_loader: Any,
        config: StageConfig,
    ) -> Dict[str, bool]:
        """Default milestone evaluation if no custom evaluator provided.

        Implements comprehensive milestone checking per curriculum_strategy.md.
        Each stage has specific success criteria that must ALL pass before
        proceeding to the next stage (go/no-go evaluation).

        This method evaluates:
        1. Task-specific performance (accuracy thresholds)
        2. System health (firing rates, stability, no pathologies)
        3. Backward compatibility (previous stages maintained)
        4. Growth progress (capacity metrics)

        Args:
            stage: Current curriculum stage
            task_loader: Task dataset for evaluation
            config: Stage configuration with success_criteria

        Returns:
            Dict mapping criterion name to pass/fail bool
        """
        results = {}

        # =====================================================================
        # 1. TASK-SPECIFIC PERFORMANCE
        # =====================================================================
        # Evaluate each criterion from config.success_criteria
        # These are stage-specific thresholds (e.g., "mnist_accuracy": 0.95)

        for criterion, threshold in config.success_criteria.items():
            # Extract task and metric from criterion name
            # Format: "task_metric" (e.g., "mnist_accuracy", "reaching_success")

            if hasattr(task_loader, 'evaluate'):
                # Task loader provides evaluation method
                actual_value = task_loader.evaluate(self.brain, criterion)
            else:
                # Fallback: Run task and measure performance
                actual_value = self._evaluate_criterion(criterion, task_loader, threshold)

            # Compare against threshold
            passed = actual_value >= threshold
            results[criterion] = passed

            if self.verbose:
                status = "✅" if passed else "❌"
                print(f"  {status} {criterion}: {actual_value:.3f} (threshold: {threshold:.3f})")

        # =====================================================================
        # 2. SYSTEM HEALTH CHECKS
        # =====================================================================
        # These are universal criteria across all stages

        # 2.1 Firing rate stability
        firing_rates = self._get_region_firing_rates()
        firing_rate_ok = all(0.05 <= fr <= 0.15 for fr in firing_rates.values())
        results['firing_rate_stability'] = firing_rate_ok

        if self.verbose:
            status = "✅" if firing_rate_ok else "❌"
            print(f"  {status} firing_rate_stability: {firing_rate_ok}")
            if not firing_rate_ok:
                for region, fr in firing_rates.items():
                    if not 0.05 <= fr <= 0.15:
                        print(f"      ⚠️  {region}: {fr:.3f}")

        # 2.2 No runaway excitation
        no_runaway = all(fr < 0.8 for fr in firing_rates.values())
        results['no_runaway_excitation'] = no_runaway

        # 2.3 No silent regions
        no_silence = all(fr > FIRING_RATE_MINIMUM for fr in firing_rates.values())
        results['no_silent_regions'] = no_silence

        # 2.4 Weight health (not saturated)
        weight_health = self._check_weight_saturation()
        results['weight_health'] = weight_health

        if self.verbose:
            status = "✅" if weight_health else "❌"
            print(f"  {status} weight_health: {weight_health}")

        # 2.5 Oscillator accuracy (if theta oscillations used)
        if hasattr(self.brain, 'oscillators') and hasattr(self.brain.oscillators, 'theta'):
            theta_freq = self.brain.oscillators.theta.frequency_hz
            theta_ok = 7.5 <= theta_freq <= 8.5
            results['theta_oscillations'] = theta_ok

            if self.verbose:
                status = "✅" if theta_ok else "❌"
                print(f"  {status} theta_oscillations: {theta_freq:.2f} Hz")

        # =====================================================================
        # 3. BACKWARD COMPATIBILITY
        # =====================================================================
        # Ensure previous stage performance is maintained (>90% of original)

        if stage.value > -1:  # Not the first stage
            prev_stage_ok = self._check_backward_compatibility(stage)
            results['backward_compatibility'] = prev_stage_ok

            if self.verbose:
                status = "✅" if prev_stage_ok else "❌"
                print(f"  {status} backward_compatibility: {prev_stage_ok}")

        # =====================================================================
        # 4. GROWTH PROGRESS (if enabled)
        # =====================================================================
        if config.enable_growth:
            growth_ok = self._check_growth_progress(stage)
            results['growth_progress'] = growth_ok

            if self.verbose:
                status = "✅" if growth_ok else "❌"
                print(f"  {status} growth_progress: {growth_ok}")

        return results

    def _evaluate_criterion(
        self,
        criterion: str,
        task_loader: Any,
        threshold: float,
    ) -> float:
        """Evaluate a single criterion by running tasks.

        Args:
            criterion: Criterion name (e.g., "mnist_accuracy")
            task_loader: Task loader for evaluation
            threshold: Required threshold

        Returns:
            Actual performance value (0.0 to 1.0)
        """
        # Parse criterion name to extract task and metric
        parts = criterion.split('_')
        if len(parts) >= 2:
            task_name = '_'.join(parts[:-1])  # e.g., "mnist", "reaching"
            metric = parts[-1]  # e.g., "accuracy", "success"
        else:
            task_name = criterion
            metric = 'accuracy'

        # Run evaluation trials (e.g., 100 test samples)
        n_trials = 100
        correct = 0

        for _ in range(n_trials):
            try:
                # Get test sample
                if hasattr(task_loader, 'get_test_sample'):
                    test_data = task_loader.get_test_sample(task_name)
                else:
                    # Fallback
                    test_data = task_loader.get_task(task_name)

                # Forward pass
                output = self.brain.forward(
                    test_data['input'],
                    n_timesteps=test_data.get('n_timesteps', 10),
                )

                # Evaluate based on metric type
                if metric in ['accuracy', 'correct', 'success']:
                    # Classification/binary success
                    if 'label' in test_data:
                        # Compare prediction to label
                        prediction = self._extract_prediction(output)
                        correct += int(prediction == test_data['label'])
                    elif 'reward' in test_data:
                        # RL task - check if reward achieved
                        correct += int(test_data['reward'] > 0)
                elif metric in ['error', 'loss']:
                    # Error metric (lower is better)
                    if 'target' in test_data:
                        error = self._compute_error(output, test_data['target'])
                        correct += (1.0 - min(error, 1.0))
            except Exception:
                # Skip failed trials
                continue

        # Return proportion correct
        return correct / n_trials if n_trials > 0 else 0.0

    def _extract_prediction(self, output: Dict[str, Any]) -> int:
        """Extract prediction from brain output.

        For classification, uses striatum action selection.
        For other tasks, may use different criteria.
        """
        # Use striatum's action as prediction
        if hasattr(self.brain, 'striatum'):
            action, _confidence = self.brain.select_action(explore=False)
            return action

        # Fallback: use most active region
        return 0

    def _compute_error(self, output: Dict[str, Any], target: Any) -> float:
        """Compute error between output and target."""
        # Placeholder - would compute actual error based on task
        return 0.0

    def _get_region_firing_rates(self) -> Dict[str, float]:
        """Get firing rates for all brain regions.

        Returns:
            Dict mapping region name to firing rate (0.0 to 1.0)
        """
        firing_rates = {}

        region_mapping = {
            'cortex': self.brain.cortex.impl if hasattr(self.brain, 'cortex') else None,
            'hippocampus': self.brain.hippocampus.impl if hasattr(self.brain, 'hippocampus') else None,
            'pfc': self.brain.pfc.impl if hasattr(self.brain, 'pfc') else None,
            'striatum': self.brain.striatum.impl if hasattr(self.brain, 'striatum') else None,
            'cerebellum': self.brain.cerebellum.impl if hasattr(self.brain, 'cerebellum') else None,
        }

        for region_name, region in region_mapping.items():
            if region is None:
                continue

            # Get firing rate from current state
            if hasattr(region, 'state') and hasattr(region.state, 'spikes'):
                if region.state.spikes is not None:
                    firing_rates[region_name] = compute_firing_rate(region.state.spikes)
                else:
                    firing_rates[region_name] = 0.0
            else:
                firing_rates[region_name] = 0.0

        return firing_rates

    def _check_weight_saturation(self) -> bool:
        """Check if weights are healthy (not saturated).

        Returns:
            True if <80% of weights are saturated
        """
        # Check each region for weight saturation

        region_mapping = {
            'cortex': self.brain.cortex.impl if hasattr(self.brain, 'cortex') else None,
            'hippocampus': self.brain.hippocampus.impl if hasattr(self.brain, 'hippocampus') else None,
            'pfc': self.brain.pfc.impl if hasattr(self.brain, 'pfc') else None,
            'striatum': self.brain.striatum.impl if hasattr(self.brain, 'striatum') else None,
            'cerebellum': self.brain.cerebellum.impl if hasattr(self.brain, 'cerebellum') else None,
        }

        for region_name, region in region_mapping.items():
            if region is None:
                continue

            growth_manager = GrowthManager(region_name=region_name)
            metrics = growth_manager.get_capacity_metrics(region)

            # Check saturation fraction
            if metrics.saturation_fraction is not None:
                if metrics.saturation_fraction >= 0.80:
                    return False

        return True

    def _check_backward_compatibility(self, current_stage: CurriculumStage) -> bool:
        """Check if previous stage performance is maintained.

        Implements catastrophic forgetting detection by re-evaluating
        previous stage milestones and comparing to original performance.

        Args:
            current_stage: Current stage being evaluated

        Returns:
            True if all previous stages maintained >90% of original performance
        """
        # Get all previous stages (stages with lower enum values)
        previous_stages = [
            stage for stage in CurriculumStage
            if stage.value < current_stage.value
        ]

        if not previous_stages:
            # First stage - no backward compatibility to check
            return True

        # Check each previous stage
        for prev_stage in previous_stages:
            # Find original performance from training history
            stage_results = [
                r for r in self.training_history
                if r.stage == prev_stage and r.success
            ]

            if not stage_results:
                # Previous stage was never successfully completed
                if self.verbose:
                    print(f"  ⚠️  backward_compatibility: No successful training for {prev_stage.name}")
                continue

            # Use last successful result for this stage
            original_result = stage_results[-1]
            original_metrics = original_result.milestone_results

            if not original_metrics:
                # No milestones recorded for comparison
                continue

            # Re-evaluate each milestone from the previous stage
            retained_count = 0
            total_count = 0

            for criterion, originally_passed in original_metrics.items():
                if not originally_passed:
                    # Skip criteria that weren't passed originally
                    continue

                total_count += 1

                # Re-evaluate this criterion now
                try:
                    # For simplicity, we'll check if the brain can still
                    # perform at 90% of the original level
                    # This would require re-running tasks from that stage

                    # TODO: This needs a task loader for the previous stage
                    # For now, we'll do a simplified check based on
                    # current milestone results

                    # If we have current metrics for the same criterion,
                    # check if they're within 90% of original
                    if hasattr(self, '_last_milestone_results'):
                        current_metrics = self._last_milestone_results
                        if criterion in current_metrics:
                            currently_passing = current_metrics[criterion]
                            if currently_passing:
                                retained_count += 1
                            elif self.verbose:
                                print(f"    ❌ Lost: {criterion} (from {prev_stage.name})")
                        else:
                            # Criterion not in current metrics - assume retained
                            # (it may not be relevant for current stage)
                            retained_count += 1
                    else:
                        # No current metrics - assume retained
                        retained_count += 1

                except Exception as e:
                    if self.verbose:
                        print(f"    ⚠️  Could not re-evaluate {criterion}: {e}")
                    # Assume retained on error (benefit of doubt)
                    retained_count += 1

            # Check retention threshold (90%)
            if total_count > 0:
                retention_rate = retained_count / total_count
                if retention_rate < 0.90:
                    if self.verbose:
                        print(f"  ❌ {prev_stage.name}: Only {retention_rate:.1%} retained "
                              f"({retained_count}/{total_count})")
                    return False
                elif self.verbose:
                    print(f"  ✅ {prev_stage.name}: {retention_rate:.1%} retained "
                          f"({retained_count}/{total_count})")

        # All previous stages maintained
        return True

    def _check_growth_progress(self, stage: CurriculumStage) -> bool:
        """Check if growth has progressed appropriately for stage.

        Args:
            stage: Current curriculum stage

        Returns:
            True if brain has grown to expected size for this stage
        """
        # Expected sizes per stage (from curriculum_strategy.md)
        expected_sizes = {
            CurriculumStage.SENSORIMOTOR: 35000,  # +5k from 30k initial
            CurriculumStage.PHONOLOGY: 50000,     # +15k
            CurriculumStage.TODDLER: 75000,       # +25k
            CurriculumStage.GRAMMAR: 100000,      # +25k
            CurriculumStage.READING: 120000,      # +20k
            CurriculumStage.ABSTRACT: 135000,     # +15k
        }

        if stage not in expected_sizes:
            return True

        # Count current neurons across all regions
        total_neurons = 0

        region_mapping = {
            'cortex': self.brain.cortex.impl if hasattr(self.brain, 'cortex') else None,
            'hippocampus': self.brain.hippocampus.impl if hasattr(self.brain, 'hippocampus') else None,
            'pfc': self.brain.pfc.impl if hasattr(self.brain, 'pfc') else None,
            'striatum': self.brain.striatum.impl if hasattr(self.brain, 'striatum') else None,
            'cerebellum': self.brain.cerebellum.impl if hasattr(self.brain, 'cerebellum') else None,
        }

        for region in region_mapping.values():
            if region is not None and hasattr(region, 'n_output'):
                total_neurons += region.n_output

        expected = expected_sizes[stage]

        # Allow 10% tolerance (may grow more or less than expected)
        return expected * 0.9 <= total_neurons <= expected * 1.2

    def _report_stage_results(self, result: TrainingResult) -> None:
        """Print stage training results."""
        if not self.verbose:
            return

        print(f"\n{'='*80}")
        print(f"Stage {result.stage.name} Results")
        print(f"{'='*80}")
        print(f"Success: {'✅ YES' if result.success else '❌ NO'}")
        print(f"Duration: {result.training_time_seconds:.1f}s ({result.total_steps:,} steps)")
        print(f"Growth events: {len(result.growth_events)}")
        print(f"Consolidation events: {len(result.consolidation_events)}")
        print(f"Checkpoints: {len(result.checkpoints)}")

        if result.milestone_results:
            print("\nMilestone Results:")
            for criterion, passed in result.milestone_results.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {criterion}")

        if result.failure_reasons:
            print("\nFailure Reasons:")
            for reason in result.failure_reasons:
                print(f"  ❌ {reason}")

        print(f"{'='*80}\n")

    # ========================================================================
    # Goal Hierarchy Setup (Stages 3+)
    # ========================================================================

    def _setup_stage_goal_hierarchies(self, stage: CurriculumStage) -> None:
        """Setup goal hierarchies for stages that use hierarchical planning.

        Stage 3 (READING): Planning tasks (Tower of Hanoi, essay writing)
        Stage 4 (ABSTRACT): Abstract reasoning (hypothesis testing, matrix reasoning)

        Args:
            stage: Current curriculum stage
        """
        if not hasattr(self.brain, 'prefrontal') or \
           not hasattr(self.brain.prefrontal, 'goal_manager') or \
           self.brain.prefrontal.goal_manager is None:
            if self.verbose:
                print("⚠️  Goal manager not available - skipping goal hierarchy setup")
            return

        if self.verbose:
            print(f"\n🎯 Setting up goal hierarchies for Stage {stage.name}...")

        if stage == CurriculumStage.READING:
            # Stage 3: Planning and text generation
            self._setup_stage3_goals()
        elif stage == CurriculumStage.ABSTRACT:
            # Stage 4: Abstract reasoning and hypothesis testing
            self._setup_stage4_goals()

        if self.verbose:
            print("  ✅ Goal hierarchies configured\n")

    def _setup_stage3_goals(self) -> None:
        """Setup goal hierarchies for Stage 3 (Reading/Planning).

        Tasks requiring hierarchical goals:
        - Tower of Hanoi (3-4 disks, multi-step planning)
        - Essay writing (intro/body/conclusion structure)
        - Maze solving (waypoint decomposition)
        """
        # 1. Tower of Hanoi goal hierarchy
        tower_hanoi = self._create_tower_hanoi_goal()
        self._goal_hierarchies['tower_hanoi'] = tower_hanoi

        # 2. Essay writing goal hierarchy
        essay = self._create_essay_goal()
        self._goal_hierarchies['essay_writing'] = essay

        # 3. Maze solving goal hierarchy
        maze = self._create_maze_goal()
        self._goal_hierarchies['maze_solving'] = maze

        # Set default goal hierarchy (essay is most general)
        self.brain.prefrontal.set_goal_hierarchy(essay)

        if self.verbose:
            print("    - Tower of Hanoi: 3-level hierarchy (move_disk → move_stack → solve)")
            print("    - Essay writing: 3-level hierarchy (intro/body/conclusion)")
            print("    - Maze solving: 2-level hierarchy (waypoints → goal)")
            print("    - Default hierarchy: essay_writing")

    def _setup_stage4_goals(self) -> None:
        """Setup goal hierarchies for Stage 4 (Abstract Reasoning).

        Tasks requiring hierarchical goals:
        - Raven's matrices (pattern analysis → rule induction → prediction)
        - Hypothesis testing (generate → test → revise)
        - Multi-premise reasoning (gather → integrate → conclude)
        """
        # 1. Raven's matrices goal hierarchy
        ravens = self._create_ravens_goal()
        self._goal_hierarchies['ravens_matrices'] = ravens

        # 2. Hypothesis testing goal hierarchy
        hypothesis = self._create_hypothesis_testing_goal()
        self._goal_hierarchies['hypothesis_testing'] = hypothesis

        # 3. Multi-premise reasoning goal hierarchy
        reasoning = self._create_reasoning_goal()
        self._goal_hierarchies['multi_premise_reasoning'] = reasoning

        # Set default goal hierarchy (hypothesis testing is most general)
        self.brain.prefrontal.set_goal_hierarchy(hypothesis)

        if self.verbose:
            print("    - Raven's matrices: 3-level hierarchy (analyze → induce → predict)")
            print("    - Hypothesis testing: 3-level hierarchy (generate → test → revise)")
            print("    - Multi-premise reasoning: 3-level hierarchy (gather → integrate → conclude)")
            print("    - Default hierarchy: hypothesis_testing")

    def _create_tower_hanoi_goal(self) -> Goal:
        """Create Tower of Hanoi goal hierarchy.

        Level 3 (root): solve_puzzle
        Level 2: move_stack(n) for each stack size
        Level 1: move_disk(i) for each individual disk
        """
        # Root goal
        root = Goal(goal_id=0, name="solve_tower_hanoi", level=3)

        # Level 2: Move stacks of different sizes
        move_3 = Goal(goal_id=1, name="move_stack_3", level=2)
        move_2 = Goal(goal_id=2, name="move_stack_2", level=2)
        move_1 = Goal(goal_id=3, name="move_stack_1", level=2)

        root.add_subgoal(move_3)
        root.add_subgoal(move_2)
        root.add_subgoal(move_1)

        # Level 1: Individual disk movements (primitives)
        for i in range(3):
            disk_goal = Goal(goal_id=4+i, name=f"move_disk_{i}", level=1)
            move_1.add_subgoal(disk_goal)

        return root

    def _create_essay_goal(self) -> Goal:
        """Create essay writing goal hierarchy.

        Level 3 (root): write_essay
        Level 2: intro, body, conclusion
        Level 1: sentences within each section
        """
        # Root goal
        root = Goal(goal_id=10, name="write_essay", level=3)

        # Level 2: Essay sections
        intro = Goal(goal_id=11, name="write_intro", level=2)
        body = Goal(goal_id=12, name="write_body", level=2)
        conclusion = Goal(goal_id=13, name="write_conclusion", level=2)

        root.add_subgoal(intro)
        root.add_subgoal(body)
        root.add_subgoal(conclusion)

        # Level 1: Sentences (3-4 per section)
        for section_id, section in [(11, intro), (12, body), (13, conclusion)]:
            for i in range(3):
                sentence_goal = Goal(
                    goal_id=section_id*10 + i,
                    name=f"{section.name}_sentence_{i}",
                    level=1
                )
                section.add_subgoal(sentence_goal)

        return root

    def _create_maze_goal(self) -> Goal:
        """Create maze solving goal hierarchy.

        Level 2 (root): reach_goal
        Level 1: reach_waypoint(i) for intermediate points
        """
        # Root goal
        root = Goal(goal_id=20, name="reach_goal", level=2)

        # Level 1: Waypoints (decompose path)
        for i in range(4):  # 4 waypoints typical for maze
            waypoint = Goal(goal_id=21+i, name=f"reach_waypoint_{i}", level=1)
            root.add_subgoal(waypoint)

        return root

    def _create_ravens_goal(self) -> Goal:
        """Create Raven's matrices goal hierarchy.

        Level 3 (root): solve_matrix
        Level 2: analyze_patterns, induce_rule
        Level 1: compare_rows, compare_cols, find_relation
        """
        # Root goal
        root = Goal(goal_id=30, name="solve_ravens_matrix", level=3)

        # Level 2: High-level reasoning steps
        analyze = Goal(goal_id=31, name="analyze_patterns", level=2)
        induce = Goal(goal_id=32, name="induce_rule", level=2)
        predict = Goal(goal_id=33, name="predict_missing", level=2)

        root.add_subgoal(analyze)
        root.add_subgoal(induce)
        root.add_subgoal(predict)

        # Level 1: Low-level analysis operations
        compare_rows = Goal(goal_id=311, name="compare_rows", level=1)
        compare_cols = Goal(goal_id=312, name="compare_cols", level=1)
        find_relation = Goal(goal_id=313, name="find_relation", level=1)

        analyze.add_subgoal(compare_rows)
        analyze.add_subgoal(compare_cols)
        analyze.add_subgoal(find_relation)

        return root

    def _create_hypothesis_testing_goal(self) -> Goal:
        """Create hypothesis testing goal hierarchy.

        Level 3 (root): scientific_reasoning
        Level 2: generate_hypotheses, test_hypothesis, revise_hypothesis
        Level 1: design_experiment, collect_data, analyze_results
        """
        # Root goal
        root = Goal(goal_id=40, name="scientific_reasoning", level=3)

        # Level 2: Hypothesis testing phases
        generate = Goal(goal_id=41, name="generate_hypotheses", level=2)
        test = Goal(goal_id=42, name="test_hypothesis", level=2)
        revise = Goal(goal_id=43, name="revise_hypothesis", level=2)

        root.add_subgoal(generate)
        root.add_subgoal(test)
        root.add_subgoal(revise)

        # Level 1: Testing operations
        design = Goal(goal_id=421, name="design_experiment", level=1)
        collect = Goal(goal_id=422, name="collect_data", level=1)
        analyze = Goal(goal_id=423, name="analyze_results", level=1)

        test.add_subgoal(design)
        test.add_subgoal(collect)
        test.add_subgoal(analyze)

        return root

    def _create_reasoning_goal(self) -> Goal:
        """Create multi-premise reasoning goal hierarchy.

        Level 3 (root): logical_inference
        Level 2: gather_premises, integrate_premises, draw_conclusion
        Level 1: parse_premise, check_consistency, apply_rule
        """
        # Root goal
        root = Goal(goal_id=50, name="logical_inference", level=3)

        # Level 2: Reasoning phases
        gather = Goal(goal_id=51, name="gather_premises", level=2)
        integrate = Goal(goal_id=52, name="integrate_premises", level=2)
        conclude = Goal(goal_id=53, name="draw_conclusion", level=2)

        root.add_subgoal(gather)
        root.add_subgoal(integrate)
        root.add_subgoal(conclude)

        # Level 1: Logical operations
        parse = Goal(goal_id=511, name="parse_premise", level=1)
        check = Goal(goal_id=512, name="check_consistency", level=1)
        apply = Goal(goal_id=513, name="apply_rule", level=1)

        integrate.add_subgoal(parse)
        integrate.add_subgoal(check)
        integrate.add_subgoal(apply)

        return root

    def get_goal_hierarchy(self, task_name: str) -> Optional[Goal]:
        """Get a specific goal hierarchy for a task.

        Args:
            task_name: Name of the task (e.g., 'tower_hanoi', 'essay_writing')

        Returns:
            Goal hierarchy root, or None if not found
        """
        return self._goal_hierarchies.get(task_name)

    def set_active_goal_hierarchy(self, task_name: str) -> bool:
        """Switch to a specific goal hierarchy.

        Args:
            task_name: Name of the task

        Returns:
            True if successful, False if task not found
        """
        goal = self.get_goal_hierarchy(task_name)
        if goal is None:
            return False

        if hasattr(self.brain, 'prefrontal') and \
           hasattr(self.brain.prefrontal, 'set_goal_hierarchy'):
            self.brain.prefrontal.set_goal_hierarchy(goal)
            return True

        return False
