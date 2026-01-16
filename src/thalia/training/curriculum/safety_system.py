"""
Integrated safety system for curriculum training.

Combines stage gates, continuous monitoring, and graceful degradation
into a unified API for curriculum training.

Usage:
    safety = CurriculumSafetySystem(brain, current_stage=1)

    # During training loop
    for step in range(training_steps):
        result = train_step(brain, batch)

        # Update safety monitoring
        intervention = safety.update(brain, step, result)

        # Handle intervention if needed
        if intervention:
            handle_intervention(intervention)

    # Check if can proceed to next stage
    if safety.can_advance_stage():
        safety.advance_to_next_stage()
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Optional, Tuple, List

from .stage_gates import (
    Stage1SurvivalGate,
    GracefulDegradationManager,
    GateResult,
    GateDecision
)
from .stage_monitoring import (
    ContinuousMonitor,
    Stage1Monitor,
    InterventionType,
    MonitoringMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class SafetyStatus:
    """Current safety system status."""
    stage: int
    can_advance: bool
    degraded_modules: List[str]
    active_alerts: List[str]
    interventions_triggered: int
    health_score: float  # 0.0-1.0


class CurriculumSafetySystem:
    """
    Integrated safety system for curriculum training.

    Combines:
        - Stage gates (hard transition criteria)
        - Continuous monitoring (real-time health checks)
        - Graceful degradation (module failure handling)

    This system implements the consensus design from expert review
    and ChatGPT engineering analysis:
        "You cannot have a stage where a single failure cascades
        and destroys the entire system."
    """

    def __init__(
        self,
        brain,
        current_stage: int = 0,
        enable_auto_intervention: bool = True,
        checkpoint_callback=None
    ):
        """
        Args:
            brain: Brain instance to monitor
            current_stage: Current curriculum stage
            enable_auto_intervention: Auto-trigger interventions
            checkpoint_callback: Function to call for checkpointing
        """
        self.brain = brain
        self.current_stage = current_stage
        self.enable_auto_intervention = enable_auto_intervention
        self.checkpoint_callback = checkpoint_callback

        # Initialize components
        self.stage_gates = self._create_stage_gates()
        self.monitor = self._create_monitor(current_stage)
        self.degradation_manager = GracefulDegradationManager()

        # State tracking
        self.stage_start_step = 0
        self.last_gate_check_step = 0
        self.gate_check_interval = 5000  # Check gate every 5k steps

        # Performance baselines
        self.baselines = {}

        logger.info(
            f"Safety system initialized for stage {current_stage}"
        )

    def _create_stage_gates(self) -> Dict[int, any]:
        """Create stage-specific gates."""
        gates = {}

        # Stage 1 has explicit gate (highest risk)
        gates[1] = Stage1SurvivalGate()

        # Other stages would have their own gates (future work)
        # gates[2] = Stage2Gate()
        # gates[3] = Stage3Gate()

        return gates

    def _create_monitor(self, stage: int) -> ContinuousMonitor:
        """Create stage-appropriate monitor."""
        if stage == 1:
            # Stage 1 gets specialized monitor (more stringent)
            return Stage1Monitor(enable_auto_intervention=self.enable_auto_intervention)
        else:
            # General monitor for other stages
            return ContinuousMonitor(enable_auto_intervention=self.enable_auto_intervention)

    def update(
        self,
        brain,
        step: int,
        task_result: Optional[Dict] = None
    ) -> Optional[InterventionType]:
        """
        Update safety monitoring with current step.

        Args:
            brain: Brain instance
            step: Current training step
            task_result: Results from latest task

        Returns:
            InterventionType if action needed, None otherwise
        """
        # Update continuous monitoring
        metrics, intervention = self.monitor.update(brain, step, task_result)

        # Update stage gate if applicable
        if self.current_stage in self.stage_gates:
            gate = self.stage_gates[self.current_stage]
            gate.update(brain, self._metrics_to_dict(metrics))

        # Check for module failures
        if task_result and 'module_performances' in task_result:
            self._check_module_health(task_result['module_performances'])

        # Periodic gate check
        if step - self.last_gate_check_step >= self.gate_check_interval:
            self._periodic_gate_check(brain)
            self.last_gate_check_step = step

        # Handle intervention if needed
        if intervention and self.enable_auto_intervention:
            self.monitor.record_intervention(intervention)
            logger.warning(
                f"Auto-intervention triggered at step {step}: {intervention.value}"
            )
            return intervention

        return None

    def _metrics_to_dict(self, metrics: MonitoringMetrics) -> Dict:
        """Convert MonitoringMetrics to dict for gate update."""
        return {
            'theta_frequency': metrics.theta_frequency,
            'gamma_theta_phase_locking': metrics.gamma_theta_phase_locking,
            'n_back_accuracy': metrics.n_back_accuracy,
            'mean_firing_rate': metrics.mean_firing_rate,
            'replay_improvement': metrics.replay_improvement,
        }

    def _check_module_health(self, module_performances: Dict[str, float]):
        """Check each module against baseline and handle failures."""
        for module_name, current_perf in module_performances.items():
            if module_name not in self.baselines:
                # First time seeing this module, set baseline
                self.baselines[module_name] = current_perf
                continue

            baseline = self.baselines[module_name]

            # Check for significant drop
            if baseline > 0 and current_perf < baseline * 0.8:  # 20% drop
                response = self.degradation_manager.handle_module_failure(
                    module_name, baseline, current_perf
                )

                if response['severity'] == 'CRITICAL':
                    logger.error(
                        f"Critical failure in {module_name}: "
                        f"{baseline:.3f} → {current_perf:.3f}"
                    )
                    logger.error(f"Response: {response}")
                elif response['severity'] == 'MEDIUM':
                    logger.warning(
                        f"Module degradation in {module_name}: "
                        f"{baseline:.3f} → {current_perf:.3f}"
                    )

    def _periodic_gate_check(self, brain):
        """Periodically check stage gate status."""
        if self.current_stage not in self.stage_gates:
            return

        gate = self.stage_gates[self.current_stage]
        result = gate.evaluate(brain)

        if not result.passed:
            logger.warning(
                f"Stage {self.current_stage} gate check FAILED at step "
                f"{self.monitor.steps}"
            )
            logger.warning(f"Failures: {result.failures}")
            logger.warning(f"Recommendations: {result.recommendations}")

    def can_advance_stage(self) -> Tuple[bool, Optional[GateResult]]:
        """
        Check if can advance to next stage.

        Returns:
            (can_advance, gate_result)
        """
        if self.current_stage not in self.stage_gates:
            logger.warning(
                f"No gate defined for stage {self.current_stage}, "
                f"allowing advancement"
            )
            return True, None

        gate = self.stage_gates[self.current_stage]
        result = gate.evaluate(self.brain)

        if result.passed:
            logger.info(
                f"✅ Stage {self.current_stage} gate PASSED - "
                f"can advance to stage {self.current_stage + 1}"
            )
            logger.info(f"Metrics: {result.metrics}")
        else:
            logger.warning(
                f"❌ Stage {self.current_stage} gate FAILED - "
                f"cannot advance yet"
            )
            logger.warning(f"Failures: {result.failures}")
            logger.warning(f"Recommendations: {result.recommendations}")

            # Handle gate decision
            if result.decision == GateDecision.EMERGENCY_STOP:
                logger.error("EMERGENCY STOP recommended")
            elif result.decision == GateDecision.ROLLBACK:
                logger.error("ROLLBACK recommended")
            elif result.decision == GateDecision.EXTEND:
                logger.warning("EXTEND stage recommended")

        return result.passed, result

    def advance_to_next_stage(self):
        """
        Advance to next curriculum stage.

        Should only be called after can_advance_stage() returns True.
        """
        can_advance, gate_result = self.can_advance_stage()

        if not can_advance:
            raise ValueError(
                f"Cannot advance from stage {self.current_stage}: "
                f"gate criteria not met"
            )

        # Save checkpoint before advancing
        if self.checkpoint_callback:
            logger.info(f"Saving checkpoint before advancing to stage {self.current_stage + 1}")
            self.checkpoint_callback(
                stage=self.current_stage,
                reason='stage_transition'
            )

        # Advance
        old_stage = self.current_stage
        self.current_stage += 1
        self.stage_start_step = self.monitor.steps

        # Reset monitor for new stage
        self.monitor = self._create_monitor(self.current_stage)

        logger.info(
            f"✅ Advanced from stage {old_stage} to stage {self.current_stage}"
        )

    def get_status(self) -> SafetyStatus:
        """Get current safety system status."""
        # Calculate health score
        health_score = self._calculate_health_score()

        # Get degraded modules
        degradation_status = self.degradation_manager.get_system_status()

        # Get active alerts
        active_alerts = list(self.monitor.active_alerts)

        # Can advance?
        can_advance = False
        if self.current_stage in self.stage_gates:
            can_advance, _ = self.can_advance_stage()

        return SafetyStatus(
            stage=self.current_stage,
            can_advance=can_advance,
            degraded_modules=degradation_status['degraded_modules'],
            active_alerts=active_alerts,
            interventions_triggered=len(self.monitor.intervention_history),
            health_score=health_score
        )

    def _calculate_health_score(self) -> float:
        """
        Calculate overall health score (0.0-1.0).

        Lower scores indicate more issues.
        """
        score = 1.0

        # Penalize active alerts
        score -= len(self.monitor.active_alerts) * 0.1

        # Penalize degraded modules
        degradation_status = self.degradation_manager.get_system_status()
        score -= len(degradation_status['degraded_modules']) * 0.15

        # Penalize critical system degradation heavily
        critical_degraded = degradation_status['degraded_modules']
        if any(m in critical_degraded for m in ['working_memory', 'oscillators', 'replay']):
            score -= 0.5

        # Penalize recent interventions
        if len(self.monitor.intervention_history) > 0:
            recent_interventions = [
                i for i, step in self.monitor.intervention_history
                if self.monitor.steps - step < 5000
            ]
            score -= len(recent_interventions) * 0.05

        return max(0.0, min(1.0, score))

    def get_summary(self) -> Dict:
        """Get comprehensive summary for logging/reporting."""
        status = self.get_status()
        metrics_summary = self.monitor.get_metrics_summary()
        intervention_summary = self.monitor.get_intervention_summary()

        return {
            'stage': status.stage,
            'health_score': status.health_score,
            'can_advance': status.can_advance,
            'degraded_modules': status.degraded_modules,
            'active_alerts': status.active_alerts,
            'total_interventions': status.interventions_triggered,
            'interventions_by_type': intervention_summary.get('by_type', {}),
            'recent_metrics': metrics_summary,
            'steps_in_stage': self.monitor.steps - self.stage_start_step,
        }

    def force_emergency_stop(self, reason: str):
        """
        Force emergency stop (for external triggers).

        Args:
            reason: Reason for emergency stop
        """
        logger.error(f"EMERGENCY STOP triggered: {reason}")
        self.monitor.record_intervention(InterventionType.EMERGENCY_STOP)

        if self.checkpoint_callback:
            self.checkpoint_callback(
                stage=self.current_stage,
                reason=f'emergency_stop_{reason}'
            )

    def handle_intervention(
        self,
        intervention: InterventionType,
        brain
    ) -> Dict:
        """
        Execute intervention response.

        Args:
            intervention: Type of intervention needed
            brain: Brain instance

        Returns:
            Dict with actions taken
        """
        actions = {'intervention': intervention.value, 'actions': []}

        if intervention == InterventionType.EMERGENCY_STOP:
            actions['actions'].append('freeze_learning')
            actions['actions'].append('rollback_to_checkpoint')
            if self.checkpoint_callback:
                self.checkpoint_callback(
                    stage=self.current_stage,
                    reason='emergency_stop'
                )

        elif intervention == InterventionType.CONSOLIDATE:
            actions['actions'].append('trigger_consolidation')
            logger.info("Triggering emergency consolidation")

        elif intervention == InterventionType.REDUCE_LOAD:
            actions['actions'].append('reduce_task_complexity')
            actions['actions'].append('lower_learning_rates')
            logger.info("Reducing cognitive load")

        elif intervention == InterventionType.TEMPORAL_SEPARATION:
            actions['actions'].append('enable_temporal_separation')
            logger.info("Enabling temporal separation of modalities")

        elif intervention == InterventionType.ROLLBACK:
            actions['actions'].append('rollback_to_checkpoint')
            if self.checkpoint_callback:
                self.checkpoint_callback(
                    stage=self.current_stage,
                    reason='rollback'
                )

        return actions
