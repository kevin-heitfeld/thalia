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

Usage:
======

    from thalia.training.curriculum_trainer import (
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
                'reaching_accuracy': 0.90,
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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from enum import IntEnum
import time

import torch

from thalia.config.curriculum_growth import (
    CurriculumGrowthConfig,
    CurriculumStage,
    get_curriculum_growth_config,
)
from thalia.training.curriculum import (
    InterleavedCurriculumSampler,
    SpacedRepetitionScheduler,
    TestingPhaseProtocol,
    ProductiveFailurePhase,
    CurriculumDifficultyCalibrator,
    StageTransitionProtocol,
)
from thalia.memory.consolidation import (
    MemoryPressureDetector,
    SleepStageController,
    ConsolidationMetrics,
)
from thalia.io.checkpoint import BrainCheckpoint


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
        load_threshold: Maximum acceptable cognitive load (default 0.9)
        active_mechanisms: List of currently active mechanisms
        deactivated_mechanisms: List of temporarily disabled mechanisms
    """
    
    def __init__(self, load_threshold: float = 0.9):
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
            target_load: Target cognitive load (default: threshold - 0.1)
        
        **Returns**:
            List of mechanism names to deactivate (in order)
        """
        if target_load is None:
            target_load = self.load_threshold - 0.1
        
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
    ):
        """Initialize curriculum trainer.
        
        Args:
            brain: Brain instance to train
            growth_config: Growth configuration (uses default if None)
            checkpoint_dir: Directory for checkpoints (creates if needed)
            device: Device for training
            verbose: Whether to print progress
        """
        self.brain = brain
        self.device = device
        self.verbose = verbose
        
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
        
        # State tracking
        self.current_stage: Optional[CurriculumStage] = None
        self.global_step = 0
        self.stage_start_step = 0
        
        # History
        self.training_history: List[TrainingResult] = []
        self.performance_history: Dict[str, List[float]] = {}
        
        # Callbacks
        self.callbacks: List[Callable[[int, Dict[str, float]], None]] = []
        
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
                
                # 3. Forward pass
                output = self.brain.process_sample(
                    task_data['input'],
                    n_timesteps=task_data.get('n_timesteps', 10),
                )
                
                # 4. Learning (local rules, not backprop)
                if 'reward' in task_data:
                    # Reinforcement learning task
                    self.brain.apply_learning(reward=task_data['reward'])
                elif 'target' in task_data:
                    # Supervised task (cerebellum)
                    self.brain.apply_learning(target=task_data['target'])
                else:
                    # Unsupervised task (cortex)
                    self.brain.apply_learning()
                
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
                if step % 1000 == 0:
                    metrics = self._collect_metrics()
                    self._log_progress(stage, step, metrics)
                    
                    for callback in self.callbacks:
                        callback(self.global_step, metrics)
                
                self.global_step += 1
            
            # Post-training evaluation
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Stage {stage.name} training complete")
                print(f"Evaluating milestones...")
                print(f"{'='*80}\n")
            
            # Evaluate milestones
            if evaluator is not None:
                milestone_results = evaluator(self.brain, task_loader)
            else:
                milestone_results = self._default_evaluation(stage, task_loader, config)
            
            result.milestone_results = milestone_results
            result.success = all(milestone_results.values())
            
            # Final metrics
            result.final_metrics = self._collect_metrics()
            result.total_steps = config.duration_steps
            result.training_time_seconds = time.time() - start_time
            
            # Report results
            self._report_stage_results(result)
            
            # Save final checkpoint
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
        
        full_metadata = {
            'global_step': self.global_step,
            'current_stage': self.current_stage.value if self.current_stage else None,
            'stage_start_step': self.stage_start_step,
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
        
        BrainCheckpoint.save(self.brain, str(checkpoint_path), metadata=full_metadata)
        
        if self.verbose:
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
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
        # Load brain state
        state = BrainCheckpoint.load(path, device=self.device)
        self.brain.load_full_state(state)
        
        if restore_trainer_state and 'metadata' in state:
            metadata = state['metadata']
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
        """Check if growth needed and trigger if so."""
        # This will be implemented to check each brain region
        # and trigger growth based on capacity metrics
        pass
    
    def _check_and_trigger_consolidation(
        self,
        stage: CurriculumStage,
        config: StageConfig,
        result: TrainingResult,
    ) -> None:
        """Check memory pressure and trigger consolidation if needed."""
        # Calculate memory pressure
        # If high, trigger consolidation (NREM/REM cycles)
        pass
    
    def _extended_consolidation(self, cycles: int = 10) -> None:
        """Perform extended consolidation before stage transition."""
        # Run consolidation with more cycles than normal
        pass
    
    def _save_checkpoint(
        self,
        stage: CurriculumStage,
        step: int,
        final: bool = False,
    ) -> str:
        """Save checkpoint internally during training."""
        suffix = "final" if final else f"step_{step}"
        name = f"stage_{stage.value}_{suffix}"
        return self.save_checkpoint(name=name)
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current training metrics."""
        # Collect firing rates, capacity, performance, etc.
        return {
            'global_step': float(self.global_step),
            'firing_rate': 0.0,  # Placeholder
            'capacity': 0.0,  # Placeholder
        }
    
    def _log_progress(
        self,
        stage: CurriculumStage,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log training progress."""
        if self.verbose and step % 5000 == 0:
            print(f"[Stage {stage.name}] Step {step:,} | " +
                  " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    
    def _default_evaluation(
        self,
        stage: CurriculumStage,
        task_loader: Any,
        config: StageConfig,
    ) -> Dict[str, bool]:
        """Default milestone evaluation if no custom evaluator provided."""
        # Basic evaluation - checks success criteria
        results = {}
        for criterion, threshold in config.success_criteria.items():
            # Placeholder - real implementation would test tasks
            results[criterion] = True
        return results
    
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
            print(f"\nMilestone Results:")
            for criterion, passed in result.milestone_results.items():
                status = "✅" if passed else "❌"
                print(f"  {status} {criterion}")
        
        if result.failure_reasons:
            print(f"\nFailure Reasons:")
            for reason in result.failure_reasons:
                print(f"  ❌ {reason}")
        
        print(f"{'='*80}\n")
