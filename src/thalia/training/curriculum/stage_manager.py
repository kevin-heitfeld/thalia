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
    from thalia.core.brain_builder import BrainBuilder
    from thalia.config import BrainConfig
    from thalia.config.curriculum_growth import get_curriculum_growth_config

    # Initialize brain
    brain_config = BrainConfig(device="cpu", dt_ms=1.0)
    brain = BrainBuilder.preset("default", brain_config)

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
                'prediction_error': 0.05,
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

import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from thalia.components.coding.spike_utils import compute_firing_rate
from thalia.config.curriculum_growth import (
    CurriculumGrowthConfig,
    CurriculumStage,
    get_attention_stage_for_curriculum,
    get_curriculum_growth_config,
)
from thalia.constants.regions import THALAMUS_ALPHA_SUPPRESSION
from thalia.constants.training import (
    CURRICULUM_LOAD_THRESHOLD,
    CURRICULUM_MARGIN,
    FIRING_RATE_MINIMUM,
    get_attention_weights,
)
from thalia.coordination.growth import GrowthManager
from thalia.diagnostics import (
    HealthConfig,
    HealthMonitor,
    PerformanceProfiler,
)
from thalia.io import CheckpointManager
from thalia.learning.critical_periods import CriticalPeriodGating
from thalia.memory.consolidation import (
    MemoryPressureDetector,
    SleepStageController,
)
from thalia.training.curriculum.curriculum import (
    InterleavedCurriculumSampler,
    SpacedRepetitionScheduler,
    StageTransitionProtocol,
    TestingPhaseProtocol,
)
from thalia.training.curriculum.noise_scheduler import (
    NoiseScheduler,
    NoiseSchedulerConfig,
)
from thalia.training.curriculum.safety_system import CurriculumSafetySystem
from thalia.training.curriculum.stage_monitoring import InterventionType
from thalia.training.visualization.live_diagnostics import LiveDiagnostics


# ============================================================================
# Cognitive Load Monitoring
# ============================================================================


class MechanismPriority(IntEnum):
    """Priority levels for cognitive mechanisms."""

    CRITICAL = 1  # Cannot be disabled (e.g., basic perception)
    HIGH = 2  # Core mechanisms for current stage
    MEDIUM = 3  # Supporting mechanisms
    LOW = 4  # Optional enhancements


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
            return {"min": current, "max": current, "mean": current, "current": current}

        # Filter by time window
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            history = [(t, load) for t, load in self._load_history if t >= cutoff]
        else:
            history = self._load_history

        if not history:
            current = self.calculate_load()
            return {"min": current, "max": current, "mean": current, "current": current}

        loads = [load for _, load in history]
        return {
            "min": min(loads),
            "max": max(loads),
            "mean": sum(loads) / len(loads),
            "current": self.calculate_load(),
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
        brain: Any,  # DynamicBrain
        growth_config: Optional[CurriculumGrowthConfig] = None,
        checkpoint_dir: Optional[str] = None,
        device: str = "cpu",
        verbose: bool = True,
        enable_live_diagnostics: bool = False,
        diagnostics_interval: int = 100,
        enable_safety_system: bool = True,
        callbacks: Optional[List[Callable[[int, Dict[str, Any]], None]]] = None,
    ):
        """Initialize curriculum trainer.

        Args:
            brain: DynamicBrain instance to train
            growth_config: Growth configuration (uses default if None)
            checkpoint_dir: Directory for checkpoints (creates if needed)
            device: Device for training
            verbose: Whether to print progress
            enable_live_diagnostics: Whether to enable real-time visualization
            diagnostics_interval: Steps between diagnostic updates (if enabled)
            enable_safety_system: Whether to enable safety monitoring and gates
            callbacks: Optional list of callback functions called with (step, metrics)
        """
        self.brain = brain
        self.device = device
        self.verbose = verbose
        self.callbacks = callbacks or []

        # Checkpoint manager (Tier 3.2 - unified checkpoint management)
        self.checkpoint_manager = CheckpointManager(
            brain=brain,
            default_compression="zstd",
            default_precision="fp32",
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
        self.noise_scheduler = NoiseScheduler(
            NoiseSchedulerConfig(verbose=verbose)
        )  # NEW: Noise scheduling

        # Safety system (NEW: Comprehensive safety monitoring)
        self.enable_safety_system = enable_safety_system
        self.safety_system: Optional[CurriculumSafetySystem] = None
        if enable_safety_system:
            self.safety_system = CurriculumSafetySystem(
                brain=brain,
                current_stage=0,  # Will be updated when stage set
                enable_auto_intervention=True,
                checkpoint_callback=self._safety_checkpoint_callback,
            )

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
        self.current_attention_stage: Optional[Any] = None  # AttentionStage
        self.global_step = 0
        self.stage_start_step = 0

        # Critical period tracking
        self._last_phase: Dict[str, str] = {}  # domain -> last_phase
        self._last_milestone_results: Dict[str, bool] = {}  # Cached milestone results

        # History
        self.training_history: List[TrainingResult] = []
        self.performance_history: Dict[str, List[float]] = {}

        # Stage task loader cache (for backward compatibility testing)
        self.stage_task_loaders: Dict[CurriculumStage, Any] = {}
        self.stage_configs: Dict[CurriculumStage, StageConfig] = {}

        # Goal hierarchy cache (for Stages 3+)
        self._goal_hierarchies: Dict[str, Goal] = {}

    def _get_brain_regions(self) -> Dict[str, Any]:
        """Get brain regions from DynamicBrain.

        Returns:
            Dictionary mapping region names to region objects
        """
        # DynamicBrain - regions in components ModuleDict
        components = self.brain.components
        return {
            "cortex": components["cortex"] if "cortex" in components else None,
            "hippocampus": components["hippocampus"] if "hippocampus" in components else None,
            "pfc": components["pfc"] if "pfc" in components else None,
            "prefrontal": components["pfc"] if "pfc" in components else None,  # Alias
            "striatum": components["striatum"] if "striatum" in components else None,
            "cerebellum": components["cerebellum"] if "cerebellum" in components else None,
            "thalamus": components["thalamus"] if "thalamus" in components else None,
        }

    def _safety_checkpoint_callback(self, stage: int, reason: str):
        """Callback for safety system to trigger checkpoints.

        Args:
            stage: Current stage number
            reason: Reason for checkpoint (e.g., 'emergency_stop', 'stage_transition')
        """
        if self.current_stage is not None:
            checkpoint_path = self._save_checkpoint(
                self.current_stage, self.global_step - self.stage_start_step, reason=reason
            )
            if self.verbose:
                print(f"Safety checkpoint saved: {checkpoint_path} (reason: {reason})")

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

        # Update attention control for new stage
        self._update_attention_stage(stage)

        # Update noise scheduler for new stage
        self.noise_scheduler.set_stage(stage)
        self._apply_noise_profile_to_brain()

        if self.verbose:
            print(f"🔊 Noise profile: {self.noise_scheduler.get_current_profile()}")

        # Update safety system for new stage
        if self.safety_system:
            self.safety_system.current_stage = stage.value
            self.safety_system.stage_start_step = self.global_step
            if self.verbose:
                print(f"🛡️ Safety system active for Stage {stage.value}")

        # TODO Phase 2: Replace goal hierarchies with example-driven training
        # Setup goal hierarchies for stages that need them (3+)
        # if stage in [CurriculumStage.READING, CurriculumStage.ABSTRACT]:
        #     self._setup_stage_goal_hierarchies(stage)

        # Initialize result tracking
        result = TrainingResult(stage=stage, success=False)

        # Setup task sampling weights
        task_weights = {
            name: cfg.weight for name, cfg in config.task_configs.items() if cfg.enabled
        }

        try:
            # Main training loop
            for step in range(config.duration_steps):
                # 0. Update neurogenesis tracking for all regions
                # This enables proper timestamping of neurons created during growth
                for _component_name, component in self.brain.components.items():
                    if hasattr(component, "set_training_step"):
                        component.set_training_step(self.global_step)

                # 1. Sample next task (interleaved practice)
                if config.interleaved_practice:
                    task_name = self.task_sampler.sample_next_task(task_weights)  # type: ignore[arg-type]
                else:
                    # Sequential (less effective, but sometimes needed)
                    task_name = list(task_weights.keys())[step % len(task_weights)]  # type: ignore[assignment]

                # 2. Get task from loader
                task_data = task_loader.get_task(task_name)  # type: ignore[arg-type]

                # 2.5. Apply critical period modulation (NEW)
                if config.enable_critical_periods:
                    self._apply_critical_period_modulation(
                        task_name=task_name,  # type: ignore[arg-type]
                        domains=config.domain_mappings.get(task_name, []),  # type: ignore[call-overload]
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
                self.brain.forward(
                    task_data["input"],
                    n_timesteps=task_data.get("n_timesteps", 10),
                )
                self.profiler.end_forward()

                # 4. External reward (ONLY for RL tasks)
                # Modulates dopamine → affects striatum three-factor learning
                # For supervised tasks (MNIST, temporal), learning is unsupervised
                # via cortical plasticity. Evaluation happens separately.
                if "reward" in task_data and task_data.get("task_type") in [
                    "motor_control",
                    "reaching",
                    "manipulation",
                    "prediction",
                    "reinforcement_learning",  # Explicit RL tasks
                ]:
                    # Deliver external reward for RL tasks
                    # This modulates dopamine for striatum/PFC
                    self.brain.deliver_reward(external_reward=task_data["reward"])

                # For classification/prediction tasks (MNIST, temporal, phonology):
                # - Forward pass creates cortical representations
                # - Learning via unsupervised BCM/STDP
                # - No external reward during training
                # - Evaluation happens periodically via milestone checks

                # Track task-specific performance (if available)
                if "accuracy" in task_data or "reward" in task_data:
                    task_type = task_data.get("task_type", "unknown")
                    perf_value = task_data.get("accuracy", task_data.get("reward", 0.0))
                    self.task_performance[task_type].append(float(perf_value))

                # Record step completion for profiler
                self.profiler.record_step()

                # Sample memory periodically (every 100 steps)
                if step % 100 == 0:
                    self.profiler.record_memory(self.brain, self.device)

                # 4.5. Safety monitoring (NEW: Check for interventions)
                if self.safety_system:
                    # Build task result for safety monitoring
                    task_result = {
                        "accuracy": task_data.get("accuracy", 0.0),
                        "reward": task_data.get("reward", 0.0),
                        "task_type": task_data.get("task_type", "unknown"),
                    }

                    # Add n-back accuracy if available (critical for Stage 1)
                    if "n_back_accuracy" in task_data:
                        task_result["n_back_accuracy"] = task_data["n_back_accuracy"]

                    # Add module-specific performances if available
                    if hasattr(self.brain, "get_module_performances"):
                        task_result["module_performances"] = self.brain.get_module_performances()

                    # Update safety monitoring
                    intervention = self.safety_system.update(
                        self.brain, self.global_step, task_result
                    )

                    # Handle intervention if triggered
                    if intervention:
                        self._handle_safety_intervention(intervention, stage, config)

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

                    # Update noise scheduler with current performance and criticality
                    # Calculate average performance across all tasks (using last 100 samples each)
                    perf_values = []
                    for task_name in self.task_performance:  # type: ignore[assignment]
                        if self.task_performance[task_name]:  # type: ignore[index]
                            recent_perf = list(self.task_performance[task_name])[-100:]  # type: ignore[index]
                            perf_values.extend(recent_perf)
                    avg_performance = (
                        sum(perf_values) / max(1, len(perf_values)) if perf_values else 0.0
                    )
                    criticality = metrics.get("criticality", 1.0) if metrics else 1.0
                    self.noise_scheduler.update(
                        stage, performance=avg_performance, criticality=criticality
                    )

                    # Apply updated noise profile
                    if step % 5000 == 0:  # Re-apply every 5000 steps for adaptation
                        self._apply_noise_profile_to_brain()

                # Call callbacks every step for progress tracking (BEFORE incrementing global_step)
                # Pass the current loop step number, not global_step
                if self.callbacks:
                    if metrics is None:
                        metrics = {}  # Empty dict for progress tracking
                    for callback in self.callbacks:
                        callback(step, metrics)

                # Increment global step counter
                self.global_step += 1

                # 9. Live diagnostics (if enabled)
                if self.enable_live_diagnostics and step % self.diagnostics_interval == 0:
                    metrics = self._collect_metrics()
                    self.live_diagnostics.update(step, self.brain, metrics)  # type: ignore[union-attr]
                    # Save plot to file instead of displaying interactively
                    plot_path = f"{self.checkpoint_dir}/../plots/diagnostics_step_{self.global_step:06d}.png"
                    self.live_diagnostics.show(save_path=plot_path)  # type: ignore[union-attr]

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

        # Safety system gate check (NEW: Hard gate for critical stages)
        if self.safety_system:
            can_advance, gate_result = self.safety_system.can_advance_stage()

            if not can_advance:
                if self.verbose:
                    print(f"❌ SAFETY GATE FAILED for Stage {stage.name}")
                    print(f"Failures: {gate_result.failures}")  # type: ignore[union-attr]
                    print(f"Recommendations: {gate_result.recommendations}")  # type: ignore[union-attr]
                    print("\nMust address these issues before proceeding.")
                return False
            elif self.verbose:
                print(f"✅ Safety gate PASSED for Stage {stage.name}")
                print(f"Gate metrics: {gate_result.metrics}")  # type: ignore[union-attr]

        # Get last training result for this stage
        stage_results = [r for r in self.training_history if r.stage == stage]
        if not stage_results:
            print(f"⚠️  No training history for Stage {stage.name}")
            return False

        last_result = stage_results[-1]

        # Check milestone results
        all_passed = all(last_result.milestone_results.values())

        if self.verbose:
            print("\nMilestone Results:")
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

        # Check safety gate before transition (CRITICAL)
        if self.safety_system:
            can_advance, gate_result = self.safety_system.can_advance_stage()
            if not can_advance:
                raise RuntimeError(
                    f"Cannot transition from {old_stage.name} to {new_stage.name}: "
                    f"Safety gate failures: {gate_result.failures}"  # type: ignore[union-attr]
                )
            if self.verbose:
                print("✅ Safety gate passed - proceeding with transition")

        # Analyze pre-transition state
        pre_transition_metrics = self.analyze_transition(old_stage, new_stage, phase="before")

        # Extended consolidation before transition
        if self.verbose:
            print("Performing extended consolidation...")

        self._extended_consolidation(cycles=10)  # 2x normal

        # Update safety system to new stage
        if self.safety_system:
            self.safety_system.advance_to_next_stage()
            if self.verbose:
                print(f"🛡️ Safety system updated to Stage {new_stage.value}")

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
                print(f"  Cognitive load: {config.cognitive_load.value}")  # type: ignore[attr-defined]

        # Analyze post-transition state
        post_transition_metrics = self.analyze_transition(old_stage, new_stage, phase="after")

        # Log transition analysis
        if self.verbose:
            self._log_transition_analysis(pre_transition_metrics, post_transition_metrics)

        if self.verbose:
            print(f"\n✅ Transition complete: Now in Stage {new_stage.name}")

    def analyze_transition(
        self,
        from_stage: CurriculumStage,
        to_stage: CurriculumStage,
        phase: str = "after",
    ) -> Dict[str, Any]:
        """Analyze brain metrics before/after stage transition.

        Captures capacity, performance, and stability metrics to understand
        the effectiveness of curriculum transitions and detect potential issues
        (catastrophic forgetting, capacity saturation, instability).

        Args:
            from_stage: Previous curriculum stage
            to_stage: New curriculum stage
            phase: 'before' or 'after' transition

        Returns:
            Dictionary with transition analysis metrics:
                - capacity_metrics: Region sizes, pathway strengths
                - performance_metrics: Task accuracies, firing rates
                - stability_metrics: Weight variance, activity patterns
                - health_status: Any detected issues

        Example:
            >>> pre = trainer.analyze_transition(old, new, phase='before')
            >>> # ... perform transition ...
            >>> post = trainer.analyze_transition(old, new, phase='after')
            >>> capacity_growth = post['capacity']['total_neurons'] - pre['capacity']['total_neurons']
        """
        analysis = {
            "from_stage": from_stage.name,
            "to_stage": to_stage.name,
            "phase": phase,
            "global_step": self.global_step,
            "timestamp": time.time(),
        }

        # 1. CAPACITY METRICS (region growth)
        capacity_metrics = {}
        if hasattr(self.brain, "regions"):
            for region_name, region in self.brain.regions.items():
                if hasattr(region, "config"):
                    size = getattr(region.config, "n_output", None) or getattr(
                        region.config, "n_neurons", None
                    )
                    if size:
                        capacity_metrics[f"{region_name}_size"] = size

            capacity_metrics["total_neurons"] = sum(capacity_metrics.values())

        analysis["capacity"] = capacity_metrics

        # 2. PERFORMANCE METRICS (task accuracy, firing rates)
        performance_metrics = {}

        # Get recent task performance from history
        if self.task_performance:
            for task_name, perf_buffer in self.task_performance.items():
                if perf_buffer:
                    recent_performance = list(perf_buffer)[-100:]  # Last 100 trials
                    performance_metrics[f"{task_name}_accuracy"] = sum(recent_performance) / len(
                        recent_performance
                    )

        # Get brain diagnostics (firing rates, etc.)
        if hasattr(self.brain, "get_diagnostics"):
            brain_diag = self.brain.get_diagnostics()
            for region_name, region_diag in brain_diag.items():
                if isinstance(region_diag, dict):
                    firing_rate = region_diag.get("firing_rate", region_diag.get("avg_firing_rate"))
                    if firing_rate is not None:
                        performance_metrics[f"{region_name}_firing_rate"] = firing_rate

        analysis["performance"] = performance_metrics

        # 3. STABILITY METRICS (weight variance, activity patterns)
        stability_metrics = {}

        # Weight statistics from regions
        if hasattr(self.brain, "regions"):
            for region_name, region in self.brain.regions.items():
                if hasattr(region, "weights"):
                    weights = region.weights.detach()
                    stability_metrics[f"{region_name}_weight_mean"] = float(weights.mean())
                    stability_metrics[f"{region_name}_weight_std"] = float(weights.std())

        # Weight statistics from pathways
        if hasattr(self.brain, "connections"):
            for pathway_key, pathway in self.brain.connections.items():
                if hasattr(pathway, "get_diagnostics"):
                    pathway_stats = pathway.get_diagnostics()
                    if isinstance(pathway_stats, dict) and "weight_mean" in pathway_stats:
                        pathway_name = f"{pathway_key[0]}_to_{pathway_key[1]}"
                        stability_metrics[f"{pathway_name}_weight_mean"] = pathway_stats[
                            "weight_mean"
                        ]
                        stability_metrics[f"{pathway_name}_weight_std"] = pathway_stats.get(
                            "weight_std", 0.0
                        )

        analysis["stability"] = stability_metrics

        # 4. HEALTH STATUS (detect issues)
        health_status = {}
        if hasattr(self.brain, "check_health"):
            try:
                health_report = self.brain.check_health()
                health_status["has_issues"] = (
                    bool(health_report.issues) if hasattr(health_report, "issues") else False  # type: ignore[assignment]
                )
                health_status["issue_count"] = (
                    len(health_report.issues) if hasattr(health_report, "issues") else 0  # type: ignore[assignment]
                )
            except Exception as e:
                health_status["error"] = str(e)  # type: ignore[assignment]

        analysis["health"] = health_status

        return analysis

    def _log_transition_analysis(
        self,
        pre_metrics: Dict[str, Any],
        post_metrics: Dict[str, Any],
    ) -> None:
        """Log transition analysis comparing before/after metrics.

        Args:
            pre_metrics: Metrics captured before transition
            post_metrics: Metrics captured after transition
        """
        print(f"\n{'='*80}")
        print("TRANSITION ANALYSIS")
        print(f"{'='*80}")

        # Capacity growth
        print("\n📈 Capacity Growth:")
        pre_capacity = pre_metrics.get("capacity", {})
        post_capacity = post_metrics.get("capacity", {})

        if "total_neurons" in pre_capacity and "total_neurons" in post_capacity:
            growth = post_capacity["total_neurons"] - pre_capacity["total_neurons"]
            growth_pct = (
                (growth / pre_capacity["total_neurons"] * 100)
                if pre_capacity["total_neurons"] > 0
                else 0
            )
            print(
                f"  Total neurons: {pre_capacity['total_neurons']:,} → {post_capacity['total_neurons']:,} (+{growth:,}, +{growth_pct:.1f}%)"
            )

        # Performance delta
        print("\n📊 Performance Delta:")
        pre_perf = pre_metrics.get("performance", {})
        post_perf = post_metrics.get("performance", {})

        for key in set(pre_perf.keys()) & set(post_perf.keys()):
            if "accuracy" in key:
                delta = post_perf[key] - pre_perf[key]
                print(f"  {key}: {pre_perf[key]:.2%} → {post_perf[key]:.2%} ({delta:+.2%})")

        # Stability
        print("\n🔄 Stability:")
        pre_stability = pre_metrics.get("stability", {})
        post_stability = post_metrics.get("stability", {})

        # Check for weight drift
        weight_drifts = []
        for key in set(pre_stability.keys()) & set(post_stability.keys()):
            if "weight_mean" in key:
                delta = abs(post_stability[key] - pre_stability[key])
                if delta > 0.1:  # Significant drift
                    weight_drifts.append((key, delta))

        if weight_drifts:
            print(f"  ⚠️  Weight drift detected in {len(weight_drifts)} components")
        else:
            print("  ✅ Weights stable across transition")

        # Health status
        print("\n🏥 Health Status:")
        pre_health = pre_metrics.get("health", {})
        post_health = post_metrics.get("health", {})

        pre_issues = pre_health.get("issue_count", 0)
        post_issues = post_health.get("issue_count", 0)

        if post_issues > pre_issues:
            print(f"  ⚠️  New health issues: {pre_issues} → {post_issues}")
        elif post_issues < pre_issues:
            print(f"  ✅ Health improved: {pre_issues} → {post_issues} issues")
        else:
            print(f"  ℹ️  Health unchanged: {post_issues} issues")

        print(f"\n{'='*80}\n")

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
        milestones: dict[str, Any] = {}
        if self.current_stage and hasattr(self, "_last_milestone_results"):
            milestones = getattr(self, "_last_milestone_results", {})

        full_metadata = {
            "global_step": self.global_step,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stage_start_step": self.stage_start_step,
            "milestones": milestones,
            "training_history": [
                {
                    "stage": r.stage.value,
                    "success": r.success,
                    "total_steps": r.total_steps,
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
            print(
                f"   Size: {save_info.get('size_mb', 0):.2f} MB, Time: {save_info.get('time_s', 0):.2f}s"
            )

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
            metadata = load_info.get("metadata", {})
            self.global_step = metadata.get("global_step", 0)
            self.stage_start_step = metadata.get("stage_start_step", 0)

            stage_value = metadata.get("current_stage")
            if stage_value is not None:
                self.current_stage = CurriculumStage(stage_value)

        if self.verbose:
            print(f"✅ Checkpoint loaded: {path}")
            if restore_trainer_state:
                print(f"   Global step: {self.global_step}")
                print(
                    f"   Current stage: {self.current_stage.name if self.current_stage else 'None'}"
                )

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

        # Get brain regions in unified way
        region_mapping = self._get_brain_regions()

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
                if event.get("component_name") == region_name:
                    last_growth_step = event.get("step", 0)

            if last_growth_step is not None:
                steps_since_growth = self.global_step - last_growth_step
                if steps_since_growth < trigger.min_steps_between:
                    if self.verbose:
                        print(
                            f"  ⏳ {region_name}: Growth on cooldown "
                            f"({steps_since_growth}/{trigger.min_steps_between} steps)"
                        )
                    continue

            # Calculate growth amount
            n_new_neurons = int(region.n_output * trigger.expansion_rate)
            n_new_neurons = max(
                growth_config.min_neurons_per_growth,
                min(n_new_neurons, growth_config.max_neurons_per_growth),
            )

            # Check total growth limit
            original_size = region.n_output - sum(
                e.get("n_neurons_added", 0)
                for e in result.growth_events
                if e.get("component_name") == region_name
            )
            current_ratio = region.n_output / max(original_size, 1)
            if current_ratio >= growth_config.max_total_growth:
                if self.verbose:
                    print(
                        f"  🛑 {region_name}: Max growth limit reached "
                        f"({current_ratio:.1f}x original)"
                    )
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
            growth_manager.grow_component(
                component=region,
                n_new=n_new_neurons,
                initialization="sparse_random",
                sparsity=0.1,
                reason=metrics.growth_reason,
                component_type="region",
            )

            # Grow connected pathways (automatic via DynamicPathwayManager)
            if hasattr(self.brain, "pathway_manager"):
                # Grow connected pathways (no longer needs adapters)
                self.brain.pathway_manager.grow_connected_pathways(
                    component_name=region_name,
                    growth_amount=n_new_neurons,
                )
                if self.verbose:
                    print(f"  🔗 Grew connected pathways for {region_name}")

            # Consolidate after growth (if enabled)
            if trigger.consolidate_after:
                if self.verbose:
                    print("  Consolidating after growth...")
                self.brain.consolidate(n_cycles=5, batch_size=32, verbose=False)

            # Record growth event
            result.growth_events.append(
                {
                    "step": self.global_step,
                    "stage": stage.name,
                    "component_name": region_name,
                    "component_type": "region",
                    "n_neurons_added": n_new_neurons,
                    "old_size": region.n_output - n_new_neurons,
                    "new_size": region.n_output,
                    "reason": metrics.growth_reason,
                    "utilization_before": metrics.utilization,
                }
            )

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
            n_cycles=config.consolidation_cycles, batch_size=32, verbose=self.verbose
        )

        # Record consolidation event
        result.consolidation_events.append(
            {
                "step": self.global_step,
                "stage": stage.name,
                "cycles": stats["cycles_completed"],
                "total_replayed": stats["total_replayed"],
                "her_enabled": stats["her_enabled"],
            }
        )

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
        stats = self.brain.consolidate(n_cycles=cycles, batch_size=64, verbose=self.verbose)

        if self.verbose:
            print(f"  Replayed {stats['total_replayed']} total experiences")

    def _handle_safety_intervention(
        self,
        intervention: InterventionType,
        stage: CurriculumStage,
        config: StageConfig,
    ):
        """Handle safety system intervention.

        Args:
            intervention: Type of intervention needed
            stage: Current stage
            config: Stage configuration
        """
        if self.verbose:
            print(f"\n⚠️ SAFETY INTERVENTION: {intervention.value}")

        actions = self.safety_system.handle_intervention(intervention)  # type: ignore[union-attr]

        if intervention == InterventionType.EMERGENCY_STOP:
            # Critical failure - halt training
            print("❌ EMERGENCY STOP triggered")
            print(f"Actions: {actions['actions']}")
            print("System frozen. Manual investigation required.")
            raise RuntimeError(f"Emergency stop triggered: {actions}")

        elif intervention == InterventionType.CONSOLIDATE:
            # Emergency consolidation needed
            print("⏸️ Triggering emergency consolidation")
            self._check_and_trigger_consolidation(stage, config, None)  # type: ignore[arg-type]

        elif intervention == InterventionType.REDUCE_LOAD:
            # Reduce cognitive load
            print("📉 Reducing cognitive load")
            # Lower learning rates temporarily
            if hasattr(self.brain, "learning_rate"):
                original_lr = self.brain.learning_rate
                self.brain.learning_rate *= 0.5
                if self.verbose:
                    print(f"   Learning rate: {original_lr:.4f} → {self.brain.learning_rate:.4f}")
            # TODO(future): Reduce task complexity if applicable
            # Requires task loaders to support difficulty adjustment
            # See: docs/design/curriculum_strategy.md for complexity scaling approach

        elif intervention == InterventionType.TEMPORAL_SEPARATION:
            # Enable temporal separation of modalities
            print("🔀 Enabling temporal separation of modalities")
            # Set flag for subsequent training steps
            if not hasattr(self, "_temporal_separation_active"):
                self._temporal_separation_active = True
                if self.verbose:
                    print("   Will alternate modality training")

        elif intervention == InterventionType.ROLLBACK:
            # Rollback to previous checkpoint
            print("⏪ Rollback triggered")
            raise RuntimeError(f"Rollback requested: {actions}")

    def _save_checkpoint(
        self,
        stage: CurriculumStage,
        step: int,
        final: bool = False,
        reason: Optional[str] = None,
    ) -> str:
        """Save checkpoint internally during training.

        Args:
            stage: Current curriculum stage
            step: Current step within the stage
            final: Whether this is a final checkpoint
            reason: Optional reason for checkpoint (e.g., 'emergency_stop')
        """
        if reason:
            suffix = reason
        elif final:
            suffix = "final"
        else:
            suffix = f"step_{step}"

        name = f"stage_{stage.value}_{suffix}"

        # Include current milestone progress in metadata
        milestone_metadata = {}
        if hasattr(self, "_last_milestone_results"):
            milestone_metadata = {"milestones": self._last_milestone_results}
        if reason:
            milestone_metadata["checkpoint_reason"] = reason  # type: ignore[assignment]

        return self.save_checkpoint(name=name, metadata=milestone_metadata)

    def _update_attention_stage(self, curriculum_stage: CurriculumStage) -> None:
        """Update attention control parameters based on curriculum stage.

        Adjusts thalamic gating and PFC feedback to shift attention from
        reactive (bottom-up) to proactive (top-down) across development.

        Args:
            curriculum_stage: Current curriculum training stage
        """
        # Get attention stage and weights for this curriculum stage
        attention_stage = get_attention_stage_for_curriculum(curriculum_stage)
        bottom_up_weight, top_down_weight = get_attention_weights(attention_stage)

        # Store current attention stage
        self.current_attention_stage = attention_stage

        if self.verbose:
            print(
                f"🎯 Attention control: {attention_stage.name} "
                f"(bottom-up: {bottom_up_weight:.0%}, top-down: {top_down_weight:.0%})"
            )

        # Get thalamus region
        regions = self._get_brain_regions()
        thalamus = regions.get("thalamus")

        if thalamus is None:
            if self.verbose:
                print("  ⚠️  Warning: No thalamus found, skipping attention control update")
            return

        # Adjust thalamic alpha suppression (bottom-up gating)
        # Higher bottom-up weight → lower suppression (more reactive to salience)
        # Lower bottom-up weight → higher suppression (ignore distractors)
        if hasattr(thalamus, "thalamus_config"):
            # Scale suppression inversely with bottom-up weight
            # Infant (100% bottom-up): 0.5x suppression (very reactive)
            # School-age (30% bottom-up): 1.5x suppression (ignore distractors)
            suppression_scale = 0.5 + (1.0 - bottom_up_weight)
            thalamus.thalamus_config.alpha_suppression_strength = (
                THALAMUS_ALPHA_SUPPRESSION * suppression_scale
            )

            # Adjust L6 feedback strength (top-down modulation)
            # Higher top-down weight → stronger PFC→thalamus feedback
            # L6a (type I): Inhibitory modulation via TRN
            # L6b (type II): Excitatory modulation of relay neurons
            base_l6a = 0.8  # From ThalamicRelayConfig defaults
            base_l6b = 0.6

            thalamus.thalamus_config.l6a_to_trn_strength = base_l6a * (0.5 + top_down_weight)
            thalamus.thalamus_config.l6b_to_relay_strength = base_l6b * (0.5 + top_down_weight)

            if self.verbose:
                print(
                    f"  • Alpha suppression: {thalamus.thalamus_config.alpha_suppression_strength:.3f} "
                    f"(scale: {suppression_scale:.2f}x)"
                )
                print(f"  • L6a→TRN feedback: {thalamus.thalamus_config.l6a_to_trn_strength:.3f}")
                print(
                    f"  • L6b→relay feedback: {thalamus.thalamus_config.l6b_to_relay_strength:.3f}"
                )

    def _apply_noise_profile_to_brain(self) -> None:
        """Apply current noise profile to all brain regions and oscillators.

        This updates membrane noise, weight noise settings, oscillator phase noise,
        and other noise-related parameters across all neural components.
        """
        profile = self.noise_scheduler.get_current_profile()

        # Apply membrane noise to all regions with neurons (DynamicBrain)
        region_mapping = self._get_brain_regions()

        for region_name, region in region_mapping.items():
            if region is None:
                continue

            # Get region-specific membrane noise
            membrane_noise = self.noise_scheduler.get_membrane_noise_for_region(region_name)

            # Apply to neurons if they exist
            if hasattr(region, "neurons") and region.neurons is not None:
                if hasattr(region.neurons, "config"):
                    region.neurons.config.noise_std = membrane_noise

            # Apply weight noise flag to region (for learning)
            if hasattr(region, "_weight_noise_enabled"):
                region._weight_noise_enabled = profile.enable_weight_noise
            if hasattr(region, "_weight_noise_std"):
                region._weight_noise_std = profile.weight_noise_std

        # Apply working memory noise to PFC specifically
        regions = self._get_brain_regions()
        pfc = regions.get("pfc")
        if pfc is not None and hasattr(pfc, "pfc_config"):
            pfc.pfc_config.wm_noise_std = profile.wm_noise_std

        # Apply oscillator phase noise to all oscillators
        if hasattr(self.brain, "oscillators"):
            phase_noise_std = self.noise_scheduler.get_oscillator_phase_noise_std()
            # OscillatorManager stores oscillators in .oscillators dict
            oscillators = (
                self.brain.oscillators.oscillators
                if hasattr(self.brain.oscillators, "oscillators")
                else {}
            )
            for _, oscillator in oscillators.items():
                if hasattr(oscillator, "config"):
                    oscillator.config.phase_noise_std = phase_noise_std

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
                multiplier = status["multiplier"]
                multipliers[domain] = multiplier
                total_multiplier += multiplier

                # Log if at critical transition points
                last_phase = self._last_phase.get(domain)
                if status["phase"] != last_phase:
                    if self.verbose:
                        print(
                            f"  🧠 Critical period: {domain} entering {status['phase']} phase "
                            f"(multiplier: {multiplier:.2f})"
                        )
                    self._last_phase[domain] = status["phase"]

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Warning: Unknown domain '{domain}' for task '{task_name}': {e}")
                continue

        # Average multiplier for this task
        if multipliers:
            avg_multiplier = total_multiplier / len(multipliers)

            # Apply to brain plasticity
            # Set as global plasticity modulator
            if hasattr(self.brain, "set_plasticity_modulator"):
                self.brain.set_plasticity_modulator(avg_multiplier)
            elif hasattr(self.brain, "state"):
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
            "global_step": float(self.global_step),
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

            # Ensure brain_diag is not None
            if brain_diag is None:
                brain_diag = {}

            # 3.1 Firing rates per region
            spike_counts = brain_diag.get("spike_counts", {})

            # Create structured region_firing_rates dict for progress callback
            region_firing_rates = {}
            for region_name, spike_count in spike_counts.items():
                # Normalize by rough neuron count estimate
                # This is approximate; actual firing rate calculation would need neuron counts
                fr_value = spike_count / 100.0
                metrics[f"firing_rate/{region_name}"] = fr_value
                region_firing_rates[region_name] = fr_value

            # Add structured dict for easy access in callbacks
            if region_firing_rates:
                metrics["region_firing_rates"] = region_firing_rates  # type: ignore[assignment]

            # Overall average firing rate
            if spike_counts:
                avg_fr = sum(spike_counts.values()) / (len(spike_counts) * 100.0)
                metrics["firing_rate/average"] = avg_fr
                metrics["avg_firing_rate"] = avg_fr  # Alias for progress callback
            else:
                metrics["firing_rate/average"] = 0.0
                metrics["avg_firing_rate"] = 0.0

            # 3.2 Weight statistics per pathway
            pathway_diag = brain_diag.get("pathways", {})
            for pathway_name, pathway_data in pathway_diag.items():
                # Extract weight statistics if available
                if isinstance(pathway_data, dict):
                    if "weights_mean" in pathway_data:
                        metrics[f"weights/{pathway_name}_mean"] = float(
                            pathway_data["weights_mean"]
                        )
                    if "weights_std" in pathway_data:
                        metrics[f"weights/{pathway_name}_std"] = float(pathway_data["weights_std"])
                    if "weights_min" in pathway_data:
                        metrics[f"weights/{pathway_name}_min"] = float(pathway_data["weights_min"])
                    if "weights_max" in pathway_data:
                        metrics[f"weights/{pathway_name}_max"] = float(pathway_data["weights_max"])

            # 3.3 Neuromodulator levels
            dopamine = brain_diag.get("dopamine", {})
            if dopamine:
                da_global = float(dopamine.get("global", 0.0))
                metrics["neuromodulator/dopamine_global"] = da_global
                metrics["dopamine"] = da_global  # Alias for progress callback
                metrics["neuromodulator/dopamine_tonic"] = float(dopamine.get("tonic", 0.0))
                metrics["neuromodulator/dopamine_phasic"] = float(dopamine.get("phasic", 0.0))
            else:
                metrics["dopamine"] = 0.0

            lc_state = brain_diag.get("locus_coeruleus", {})
            if lc_state:
                metrics["neuromodulator/norepinephrine"] = float(
                    lc_state.get("norepinephrine", 0.0)
                )
                metrics["neuromodulator/arousal"] = float(lc_state.get("arousal", 0.0))

            nb_state = brain_diag.get("nucleus_basalis", {})
            if nb_state:
                metrics["neuromodulator/acetylcholine"] = float(nb_state.get("acetylcholine", 0.0))

            # 3.4 Oscillator state
            metrics["oscillator/theta_phase"] = float(brain_diag.get("theta_phase", 0.0))
            metrics["oscillator/theta_frequency"] = float(brain_diag.get("theta_frequency", 8.0))

        except Exception as e:
            # Graceful degradation if brain diagnostics fail
            if self.verbose:
                print(f"  ⚠️ Warning: Failed to collect brain diagnostics: {e}")
                traceback.print_exc()  # Always print traceback
            metrics["firing_rate/average"] = 0.0

        # =====================================================================
        # 4. HEALTH STATUS
        # =====================================================================
        try:
            # Reuse brain_diag from previous block if available, otherwise fetch it
            if "brain_diag" not in locals() or brain_diag is None:
                brain_diag = self.brain.get_diagnostics()
                if brain_diag is None:
                    brain_diag = {}

            health_report = self.health_monitor.check_health(brain_diag)

            metrics["health/is_healthy"] = 1.0 if health_report.is_healthy else 0.0
            metrics["health/issue_count"] = float(len(health_report.issues))

            # Maximum severity across all issues
            if health_report.issues:
                max_severity = max(issue.severity for issue in health_report.issues)
                metrics["health/max_severity"] = float(max_severity)
            else:
                metrics["health/max_severity"] = 0.0

            # Count by issue type
            issue_counts: dict[str, int] = {}
            for issue in health_report.issues:
                issue_type = issue.issue_type.value
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

            for issue_type, count in issue_counts.items():
                metrics[f"health/issue_{issue_type}"] = float(count)

        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Warning: Failed to check health: {e}")
                traceback.print_exc()  # Always print traceback
            metrics["health/is_healthy"] = 1.0  # Assume healthy if check fails
            metrics["health/issue_count"] = 0.0
            metrics["health/max_severity"] = 0.0

        # =====================================================================
        # 5. TASK-SPECIFIC PERFORMANCE
        # =====================================================================
        # Create structured task_performance dict for progress callback
        task_performance_dict = {}
        for task_name, perf_buffer in self.task_performance.items():
            if perf_buffer:
                # Recent average (last 100 samples)
                recent_avg = float(np.mean(list(perf_buffer)[-100:]))
                metrics[f"task/{task_name}_accuracy"] = recent_avg
                task_performance_dict[task_name] = recent_avg
                # Overall average
                metrics[f"task/{task_name}_accuracy_all"] = float(np.mean(perf_buffer))

        # Add structured dict for easy access in callbacks
        if task_performance_dict:
            metrics["task_performance"] = task_performance_dict  # type: ignore[assignment]
            # Set highest performing task as current accuracy
            metrics["accuracy"] = (
                max(task_performance_dict.values()) if task_performance_dict else 0.0
            )
            metrics["task_accuracy"] = metrics["accuracy"]  # Alias

        # =====================================================================
        # 6. CRITICAL PERIOD STATUS
        # =====================================================================
        for domain in self.critical_period_gating.get_all_domains():
            try:
                status = self.critical_period_gating.get_window_status(domain, self.global_step)
                metrics[f"critical_period/{domain}_multiplier"] = float(status["multiplier"])
                metrics[f"critical_period/{domain}_progress"] = float(status["progress"])
                metrics[f"critical_period/{domain}_phase"] = float(
                    {"early": 0.0, "peak": 1.0, "late": 2.0}.get(status["phase"], -1.0)
                )
            except Exception:  # nosec B110
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
            steps_per_sec = metrics.get("performance/steps_per_sec", 0.0)
            forward_ms = metrics.get("performance/avg_forward_ms", 0.0)
            gpu_mb = metrics.get("memory/gpu_mb", 0.0)
            cpu_mb = metrics.get("memory/cpu_mb", 0.0)
            avg_fr = metrics.get("firing_rate/average", 0.0)
            is_healthy = metrics.get("health/is_healthy", 1.0)

            health_icon = "✅" if is_healthy > 0.5 else "⚠️"

            print(
                f"[Stage {stage.name}] Step {step:,} | "
                f"{health_icon} {steps_per_sec:.1f} steps/s | "
                f"Forward: {forward_ms:.1f}ms | "
                f"GPU: {gpu_mb:.0f}MB | CPU: {cpu_mb:.0f}MB | "
                f"FR: {avg_fr:.3f}"
            )

            # Safety system status (NEW)
            if self.safety_system:
                status = self.safety_system.get_status()
                health_emoji = (
                    "🟢"
                    if status.health_score >= 0.7
                    else "🟡" if status.health_score >= 0.5 else "🔴"
                )
                print(
                    f"         🛡️ Safety: {health_emoji} Health={status.health_score:.2f} | "
                    f"Alerts={len(status.active_alerts)} | "
                    f"Interventions={status.interventions_triggered}"
                )

                if status.active_alerts:
                    print(f"              Active alerts: {', '.join(status.active_alerts)}")

                if status.degraded_modules:
                    print(f"              Degraded modules: {', '.join(status.degraded_modules)}")

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

            if hasattr(task_loader, "evaluate"):
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
        results["firing_rate_stability"] = firing_rate_ok

        if self.verbose:
            status = "✅" if firing_rate_ok else "❌"
            print(f"  {status} firing_rate_stability: {firing_rate_ok}")
            if not firing_rate_ok:
                for region, fr in firing_rates.items():
                    if not 0.05 <= fr <= 0.15:
                        print(f"      ⚠️  {region}: {fr:.3f}")

        # 2.2 No runaway excitation
        no_runaway = all(fr < 0.8 for fr in firing_rates.values())
        results["no_runaway_excitation"] = no_runaway

        # 2.3 No silent regions
        no_silence = all(fr > FIRING_RATE_MINIMUM for fr in firing_rates.values())
        results["no_silent_regions"] = no_silence

        # 2.4 Weight health (not saturated)
        weight_health = self._check_weight_saturation()
        results["weight_health"] = weight_health

        if self.verbose:
            status = "✅" if weight_health else "❌"
            print(f"  {status} weight_health: {weight_health}")

        # 2.5 Oscillator accuracy (if theta oscillations used)
        if hasattr(self.brain, "oscillators") and hasattr(self.brain.oscillators, "theta"):
            theta_freq = self.brain.oscillators.theta.frequency_hz
            theta_ok = 7.5 <= theta_freq <= 8.5
            results["theta_oscillations"] = theta_ok

            if self.verbose:
                status = "✅" if theta_ok else "❌"
                print(f"  {status} theta_oscillations: {theta_freq:.2f} Hz")

        # =====================================================================
        # 3. BACKWARD COMPATIBILITY
        # =====================================================================
        # Ensure previous stage performance is maintained (>90% of original)

        if stage.value > -1:  # Not the first stage
            prev_stage_ok = self._check_backward_compatibility(stage)
            results["backward_compatibility"] = prev_stage_ok

            if self.verbose:
                status = "✅" if prev_stage_ok else "❌"
                print(f"  {status} backward_compatibility: {prev_stage_ok}")

        # =====================================================================
        # 4. GROWTH PROGRESS (if enabled)
        # =====================================================================
        if config.enable_growth:
            growth_ok = self._check_growth_progress(stage)
            results["growth_progress"] = growth_ok

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
        parts = criterion.split("_")
        if len(parts) >= 2:
            task_name = "_".join(parts[:-1])  # e.g., "mnist", "reaching"
            metric = parts[-1]  # e.g., "accuracy", "success"
        else:
            task_name = criterion
            metric = "accuracy"

        # Run evaluation trials (e.g., 100 test samples)
        n_trials = 100
        correct = 0

        for _ in range(n_trials):
            try:
                # Get test sample
                if hasattr(task_loader, "get_test_sample"):
                    test_data = task_loader.get_test_sample(task_name)
                else:
                    # Fallback
                    test_data = task_loader.get_task(task_name)

                # Forward pass
                output = self.brain.forward(
                    test_data["input"],
                    n_timesteps=test_data.get("n_timesteps", 10),
                )

                # Evaluate based on metric type
                if metric in ["accuracy", "correct", "success"]:
                    # Classification/binary success
                    if "label" in test_data:
                        # Compare prediction to label
                        prediction = self._extract_prediction(output)
                        correct += int(prediction == test_data["label"])
                    elif "reward" in test_data:
                        # RL task - check if reward achieved
                        correct += int(test_data["reward"] > 0)
                elif metric in ["error", "loss"]:
                    # Error metric (lower is better)
                    if "target" in test_data:
                        error = self._compute_error(output, test_data["target"])
                        correct += 1.0 - min(error, 1.0)  # type: ignore[assignment]
            except Exception:  # nosec B112
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
        if hasattr(self.brain, "striatum"):
            action, _confidence = self.brain.select_action(explore=False)
            return int(action)

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
            "cortex": self.brain.components.get("cortex"),
            "hippocampus": self.brain.components.get("hippocampus"),
            "pfc": self.brain.components.get("pfc"),
            "striatum": self.brain.components.get("striatum"),
            "cerebellum": self.brain.components.get("cerebellum"),
        }

        for region_name, region in region_mapping.items():
            if region is None:
                continue

            # Get firing rate from current state
            if hasattr(region, "state") and hasattr(region.state, "spikes"):
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
            "cortex": self.brain.components.get("cortex"),
            "hippocampus": self.brain.components.get("hippocampus"),
            "pfc": self.brain.components.get("pfc"),
            "striatum": self.brain.components.get("striatum"),
            "cerebellum": self.brain.components.get("cerebellum"),
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
        previous_stages = [stage for stage in CurriculumStage if stage.value < current_stage.value]

        if not previous_stages:
            # First stage - no backward compatibility to check
            return True

        # Check each previous stage
        for prev_stage in previous_stages:
            # Find original performance from training history
            stage_results = [
                r for r in self.training_history if r.stage == prev_stage and r.success
            ]

            if not stage_results:
                # Previous stage was never successfully completed
                if self.verbose:
                    print(
                        f"  ⚠️  backward_compatibility: No successful training for {prev_stage.name}"
                    )
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

                # Re-evaluate this criterion by re-running actual tasks
                try:
                    current_performance = self._run_regression_test(
                        prev_stage, criterion, n_trials=50
                    )

                    # Get original performance threshold from stage config
                    prev_config = self.stage_configs.get(prev_stage)
                    if prev_config and criterion in prev_config.success_criteria:
                        original_threshold = prev_config.success_criteria[criterion]
                        # Consider retained if current >= 90% of original threshold
                        retention_threshold = original_threshold * 0.90

                        if current_performance >= retention_threshold:
                            retained_count += 1
                            if self.verbose:
                                print(
                                    f"    ✅ Retained: {criterion} = {current_performance:.3f} "
                                    f"(threshold: {retention_threshold:.3f})"
                                )
                        else:
                            if self.verbose:
                                print(
                                    f"    ❌ Lost: {criterion} = {current_performance:.3f} "
                                    f"< {retention_threshold:.3f} (from {prev_stage.name})"
                                )
                    else:
                        # No threshold available - assume retained if > 0.5
                        if current_performance >= 0.50:
                            retained_count += 1
                        elif self.verbose:
                            print(
                                f"    ⚠️  {criterion}: {current_performance:.3f} "
                                f"(no original threshold available)"
                            )

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
                        print(
                            f"  ❌ {prev_stage.name}: Only {retention_rate:.1%} retained "
                            f"({retained_count}/{total_count})"
                        )
                    return False
                elif self.verbose:
                    print(
                        f"  ✅ {prev_stage.name}: {retention_rate:.1%} retained "
                        f"({retained_count}/{total_count})"
                    )

        # All previous stages maintained
        return True

    def _run_regression_test(
        self,
        stage: CurriculumStage,
        criterion: str,
        n_trials: int = 50,
    ) -> float:
        """Run regression test by re-executing tasks from a previous stage.

        This method implements proper catastrophic forgetting detection by:
        1. Loading the cached task loader for the target stage
        2. Re-running actual tasks from that stage
        3. Measuring current performance on those tasks

        Args:
            stage: Previous stage to test
            criterion: Specific criterion to evaluate (e.g., 'mnist_accuracy')
            n_trials: Number of test trials to run

        Returns:
            Current performance on the criterion (0.0 to 1.0)

        Raises:
            ValueError: If no task loader cached for the stage
        """
        # Get cached task loader for this stage
        if stage not in self.stage_task_loaders:
            raise ValueError(
                f"No task loader cached for {stage.name}. "
                f"Stage must complete successfully before regression testing."
            )

        task_loader = self.stage_task_loaders[stage]

        # Parse criterion to extract task name and metric
        # Format: "task_metric" (e.g., "mnist_accuracy", "reaching_success")
        parts = criterion.split("_")
        if len(parts) >= 2:
            task_name = "_".join(parts[:-1])  # e.g., "mnist", "reaching"
            metric = parts[-1]  # e.g., "accuracy", "success"
        else:
            task_name = criterion
            metric = "accuracy"

        # Run test trials
        correct = 0
        total = 0
        accumulated_value = 0.0

        for _ in range(n_trials):
            try:
                # Get task sample
                if hasattr(task_loader, "get_test_sample"):
                    task_data = task_loader.get_test_sample(task_name)
                else:
                    task_data = task_loader.get_task(task_name)

                # Forward pass (disable learning to avoid corrupting weights)
                original_plasticity_states = self._disable_plasticity()

                try:
                    output = self.brain.forward(
                        task_data["input"],
                        n_timesteps=task_data.get("n_timesteps", 10),
                    )
                finally:
                    # Restore plasticity state
                    self._restore_plasticity(original_plasticity_states)

                # Evaluate based on metric type
                if metric in ["accuracy", "correct", "success"]:
                    # Classification/binary success
                    if "label" in task_data:
                        prediction = self._extract_prediction(output)
                        correct += int(prediction == task_data["label"])
                        total += 1
                    elif "reward" in task_data:
                        # RL task: reward > 0 counts as success
                        correct += int(task_data["reward"] > 0.0)
                        total += 1
                    elif "target" in task_data:
                        # Supervised task: check if close to target
                        error = self._compute_error(output, task_data["target"])
                        correct += int(error < 0.1)  # Threshold for success
                        total += 1
                elif metric in ["error", "loss"]:
                    # Lower is better
                    if "target" in task_data:
                        error = self._compute_error(output, task_data["target"])
                        accumulated_value += error
                        total += 1
                elif metric == "reward":
                    # Higher is better
                    if "reward" in task_data:
                        accumulated_value += task_data["reward"]
                        total += 1

            except Exception as e:
                # Skip failed trials
                if self.verbose:
                    print(f"      ⚠️  Trial failed: {e}")
                continue

        # Compute final performance
        if total == 0:
            return 0.0

        if metric in ["accuracy", "correct", "success"]:
            return correct / total
        elif metric in ["error", "loss"]:
            # Convert error to performance (1 - normalized_error)
            avg_error = accumulated_value / total
            return max(0.0, 1.0 - avg_error)
        elif metric == "reward":
            # Normalize reward to [0, 1] range (assuming rewards in [-1, 1])
            avg_reward = accumulated_value / total
            return (avg_reward + 1.0) / 2.0
        else:
            # Unknown metric
            return 0.0

    def _disable_plasticity(self) -> Dict[str, bool]:
        """Disable learning in all regions to prevent weight corruption during testing.

        Returns:
            Dictionary mapping region names to their original plasticity states
        """
        original_states = {}

        for name, component in self.brain.components.items():
            if hasattr(component, "plasticity_enabled"):
                original_states[name] = component.plasticity_enabled
                component.plasticity_enabled = False

        return original_states

    def _restore_plasticity(self, original_states: Dict[str, bool]) -> None:
        """Restore plasticity states after testing.

        Args:
            original_states: Dictionary of original plasticity states from _disable_plasticity()
        """
        for name, was_enabled in original_states.items():
            if name in self.brain.components:
                component = self.brain.components[name]
                if hasattr(component, "plasticity_enabled"):
                    component.plasticity_enabled = was_enabled

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
            CurriculumStage.PHONOLOGY: 50000,  # +15k
            CurriculumStage.TODDLER: 75000,  # +25k
            CurriculumStage.GRAMMAR: 100000,  # +25k
            CurriculumStage.READING: 120000,  # +20k
            CurriculumStage.ABSTRACT: 135000,  # +15k
        }

        if stage not in expected_sizes:
            return True

        # Count current neurons across all regions
        total_neurons = 0

        region_mapping = {
            "cortex": self.brain.components.get("cortex"),
            "hippocampus": self.brain.components.get("hippocampus"),
            "pfc": self.brain.components.get("pfc"),
            "striatum": self.brain.components.get("striatum"),
            "cerebellum": self.brain.components.get("cerebellum"),
        }

        for region in region_mapping.values():
            if region is not None and hasattr(region, "n_output"):
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
        if (
            not hasattr(self.brain, "prefrontal")
            or not hasattr(self.brain.prefrontal, "goal_manager")
            or self.brain.prefrontal.goal_manager is None
        ):
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
        self._goal_hierarchies["tower_hanoi"] = tower_hanoi

        # 2. Essay writing goal hierarchy
        essay = self._create_essay_goal()
        self._goal_hierarchies["essay_writing"] = essay

        # 3. Maze solving goal hierarchy
        maze = self._create_maze_goal()
        self._goal_hierarchies["maze_solving"] = maze

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
        self._goal_hierarchies["ravens_matrices"] = ravens

        # 2. Hypothesis testing goal hierarchy
        hypothesis = self._create_hypothesis_testing_goal()
        self._goal_hierarchies["hypothesis_testing"] = hypothesis

        # 3. Multi-premise reasoning goal hierarchy
        reasoning = self._create_reasoning_goal()
        self._goal_hierarchies["multi_premise_reasoning"] = reasoning

        # Set default goal hierarchy (hypothesis testing is most general)
        self.brain.prefrontal.set_goal_hierarchy(hypothesis)

        if self.verbose:
            print("    - Raven's matrices: 3-level hierarchy (analyze → induce → predict)")
            print("    - Hypothesis testing: 3-level hierarchy (generate → test → revise)")
            print(
                "    - Multi-premise reasoning: 3-level hierarchy (gather → integrate → conclude)"
            )
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
            disk_goal = Goal(goal_id=4 + i, name=f"move_disk_{i}", level=1)
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
                    goal_id=section_id * 10 + i, name=f"{section.name}_sentence_{i}", level=1
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
            waypoint = Goal(goal_id=21 + i, name=f"reach_waypoint_{i}", level=1)
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

        if hasattr(self.brain, "prefrontal") and hasattr(
            self.brain.prefrontal, "set_goal_hierarchy"
        ):
            self.brain.prefrontal.set_goal_hierarchy(goal)
            return True

        return False
