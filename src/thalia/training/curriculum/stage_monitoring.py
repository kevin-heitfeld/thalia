"""
Continuous monitoring system for curriculum training.

Implements real-time health checks, intervention triggers, and
metric collection for stage gate evaluation.

Integration with stage_gates.py for comprehensive safety system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of interventions that can be triggered."""
    NONE = "none"
    REDUCE_LOAD = "reduce_load"
    CONSOLIDATE = "consolidate"
    TEMPORAL_SEPARATION = "temporal_separation"
    EMERGENCY_STOP = "emergency_stop"
    ROLLBACK = "rollback"


@dataclass
class MonitoringMetrics:
    """Collected metrics for stage gate evaluation."""
    
    # Oscillator metrics
    theta_frequency: float = 0.0
    theta_variance: float = 0.0
    gamma_theta_phase_locking: float = 0.0
    
    # Working memory metrics
    n_back_accuracy: float = 0.0
    wm_capacity: int = 0
    attractor_stability: float = 0.0
    
    # Performance metrics
    task_accuracy: float = 0.0
    modality_performances: Dict[str, float] = field(default_factory=dict)
    
    # Health metrics
    mean_firing_rate: float = 0.0
    region_firing_rates: Dict[str, float] = field(default_factory=dict)
    dopamine_level: float = 0.5
    
    # Consolidation metrics
    replay_improvement: float = 0.0
    weight_saturation: float = 0.0
    
    # Stability indicators
    performance_stability: float = 0.0
    no_region_silence: bool = True


class ContinuousMonitor:
    """
    Real-time monitoring during curriculum training.
    
    Collects metrics, detects anomalies, triggers interventions.
    Works with Stage1SurvivalGate and GracefulDegradationManager.
    """
    
    def __init__(
        self,
        check_interval: int = 1000,
        window_size: int = 10000,
        enable_auto_intervention: bool = True
    ):
        """
        Args:
            check_interval: Steps between health checks
            window_size: Rolling window for metrics
            enable_auto_intervention: Automatically trigger interventions
        """
        self.check_interval = check_interval
        self.window_size = window_size
        self.enable_auto_intervention = enable_auto_intervention
        
        # Metric history
        self.metrics_history = deque(maxlen=window_size)
        self.intervention_history = []
        
        # Current state
        self.steps = 0
        self.last_check_step = 0
        self.current_stage = 0
        
        # Baseline performance (set during initial training)
        self.baselines = {}
        
        # Alerts
        self.active_alerts = set()
    
    def update(
        self,
        brain,
        step: int,
        task_result: Optional[Dict] = None
    ) -> Tuple[MonitoringMetrics, Optional[InterventionType]]:
        """
        Update monitoring with current step.
        
        Args:
            brain: Brain instance
            step: Current training step
            task_result: Results from latest task
        
        Returns:
            (current_metrics, intervention_needed)
        """
        self.steps = step
        
        # Collect current metrics
        metrics = self._collect_metrics(brain, task_result)
        self.metrics_history.append(metrics)
        
        # Check if intervention needed
        intervention = None
        if step - self.last_check_step >= self.check_interval:
            intervention = self._check_health(brain, metrics)
            self.last_check_step = step
        
        return metrics, intervention
    
    def _collect_metrics(
        self,
        brain,
        task_result: Optional[Dict]
    ) -> MonitoringMetrics:
        """Collect all monitoring metrics from brain."""
        metrics = MonitoringMetrics()
        
        # Oscillator metrics
        if hasattr(brain, 'theta_oscillator'):
            metrics.theta_frequency = brain.theta_oscillator.get_frequency()
            
            # Calculate variance from recent history
            if len(self.metrics_history) > 100:
                recent_freqs = [
                    m.theta_frequency for m in list(self.metrics_history)[-100:]
                ]
                metrics.theta_variance = np.std(recent_freqs) / np.mean(recent_freqs)
        
        if hasattr(brain, 'measure_phase_locking'):
            metrics.gamma_theta_phase_locking = brain.measure_phase_locking()
        
        # Working memory metrics
        if task_result and 'n_back_accuracy' in task_result:
            metrics.n_back_accuracy = task_result['n_back_accuracy']
        
        if hasattr(brain, 'prefrontal') and hasattr(brain.prefrontal, 'capacity'):
            metrics.wm_capacity = brain.prefrontal.capacity
        
        # Task performance
        if task_result and 'accuracy' in task_result:
            metrics.task_accuracy = task_result['accuracy']
        
        # Modality-specific performance
        if hasattr(brain, 'get_modality_performance'):
            for modality in ['phonology', 'vision', 'language']:
                try:
                    perf = brain.get_modality_performance(modality)
                    metrics.modality_performances[modality] = perf
                except:
                    pass
        
        # Firing rates
        if hasattr(brain, 'get_mean_firing_rate'):
            metrics.mean_firing_rate = brain.get_mean_firing_rate()
        
        if hasattr(brain, 'get_region_firing_rates'):
            metrics.region_firing_rates = brain.get_region_firing_rates()
            
            # Check for silent regions
            silent_regions = [
                name for name, rate in metrics.region_firing_rates.items()
                if rate < 0.01
            ]
            metrics.no_region_silence = len(silent_regions) == 0
            
            if silent_regions:
                logger.warning(f"Silent regions detected: {silent_regions}")
        
        # Neuromodulation
        if hasattr(brain, 'neuromodulation'):
            metrics.dopamine_level = brain.neuromodulation.get_dopamine()
        
        # Consolidation effectiveness
        if task_result and 'replay_improvement' in task_result:
            metrics.replay_improvement = task_result['replay_improvement']
        
        # Stability
        if len(self.metrics_history) > 100:
            recent_acc = [
                m.task_accuracy for m in list(self.metrics_history)[-100:]
            ]
            metrics.performance_stability = 1.0 - np.std(recent_acc)
        
        return metrics
    
    def _check_health(
        self,
        brain,
        current_metrics: MonitoringMetrics
    ) -> Optional[InterventionType]:
        """
        Check system health and determine if intervention needed.
        
        Returns:
            InterventionType if action required, None otherwise
        """
        if not self.enable_auto_intervention:
            return None
        
        # Critical: Oscillator instability
        if self._check_oscillator_instability(current_metrics):
            logger.error("CRITICAL: Oscillator instability detected")
            self.active_alerts.add("OSCILLATOR_INSTABILITY")
            return InterventionType.EMERGENCY_STOP
        
        # Critical: Working memory collapse
        if self._check_wm_collapse(current_metrics):
            logger.error("CRITICAL: Working memory collapse detected")
            self.active_alerts.add("WM_COLLAPSE")
            return InterventionType.EMERGENCY_STOP
        
        # Critical: Region silence
        if not current_metrics.no_region_silence:
            logger.error("CRITICAL: Silent region detected")
            self.active_alerts.add("REGION_SILENCE")
            return InterventionType.CONSOLIDATE
        
        # Warning: Cross-modal interference
        if self._check_interference(current_metrics):
            logger.warning("WARNING: Cross-modal interference detected")
            self.active_alerts.add("INTERFERENCE")
            return InterventionType.TEMPORAL_SEPARATION
        
        # Warning: Performance degradation
        if self._check_performance_degradation(current_metrics):
            logger.warning("WARNING: Performance degradation detected")
            self.active_alerts.add("PERFORMANCE_DROP")
            return InterventionType.CONSOLIDATE
        
        # Warning: High cognitive load
        if self._check_cognitive_overload(brain, current_metrics):
            logger.warning("WARNING: Cognitive overload detected")
            self.active_alerts.add("OVERLOAD")
            return InterventionType.REDUCE_LOAD
        
        # Clear alerts if all healthy
        self.active_alerts.clear()
        return None
    
    def _check_oscillator_instability(
        self, metrics: MonitoringMetrics
    ) -> bool:
        """Check for oscillator instability."""
        # Theta frequency out of range
        if not (6.5 <= metrics.theta_frequency <= 8.5):
            return True
        
        # High variance
        if metrics.theta_variance > 0.20:  # 20% variance is critical
            return True
        
        # Poor phase locking
        if metrics.gamma_theta_phase_locking < 0.3:
            return True
        
        return False
    
    def _check_wm_collapse(self, metrics: MonitoringMetrics) -> bool:
        """Check for working memory collapse."""
        # Performance below critical threshold
        if metrics.n_back_accuracy < 0.60:  # 60% is critical
            return True
        
        # Check for rapid decline
        if len(self.metrics_history) > 100:
            recent = [m.n_back_accuracy for m in list(self.metrics_history)[-100:]]
            if len(recent) > 10:
                early = np.mean(recent[:50])
                late = np.mean(recent[-50:])
                if early - late > 0.15:  # 15% rapid drop
                    return True
        
        return False
    
    def _check_interference(self, metrics: MonitoringMetrics) -> bool:
        """Check for cross-modal interference."""
        # Simultaneous performance drops in multiple modalities
        if len(metrics.modality_performances) >= 2:
            poor_modalities = [
                mod for mod, perf in metrics.modality_performances.items()
                if perf < 0.70
            ]
            if len(poor_modalities) >= 2:
                return True
        
        return False
    
    def _check_performance_degradation(
        self, metrics: MonitoringMetrics
    ) -> bool:
        """Check for general performance degradation."""
        if len(self.metrics_history) < 200:
            return False
        
        # Compare recent to baseline
        recent = [m.task_accuracy for m in list(self.metrics_history)[-100:]]
        baseline = [m.task_accuracy for m in list(self.metrics_history)[-200:-100]]
        
        if len(baseline) > 0 and len(recent) > 0:
            baseline_mean = np.mean(baseline)
            recent_mean = np.mean(recent)
            
            # 10% drop indicates need for consolidation
            if baseline_mean - recent_mean > 0.10:
                return True
        
        return False
    
    def _check_cognitive_overload(
        self, brain, metrics: MonitoringMetrics
    ) -> bool:
        """Check for cognitive overload."""
        # High firing rates indicate overload
        if metrics.mean_firing_rate > 0.20:
            return True
        
        # Dopamine saturation
        if metrics.dopamine_level > 0.95 or metrics.dopamine_level < 0.05:
            return True
        
        return False
    
    def set_baseline(self, modality: str, performance: float):
        """Set baseline performance for a modality."""
        self.baselines[modality] = performance
        logger.info(f"Set baseline for {modality}: {performance:.3f}")
    
    def get_intervention_summary(self) -> Dict:
        """Get summary of interventions triggered."""
        if not self.intervention_history:
            return {'total': 0, 'by_type': {}}
        
        by_type = {}
        for intervention, step in self.intervention_history:
            if intervention not in by_type:
                by_type[intervention] = 0
            by_type[intervention] += 1
        
        return {
            'total': len(self.intervention_history),
            'by_type': by_type,
            'most_recent': self.intervention_history[-1] if self.intervention_history else None
        }
    
    def record_intervention(self, intervention: InterventionType):
        """Record that an intervention was triggered."""
        self.intervention_history.append((intervention, self.steps))
        logger.info(
            f"Intervention triggered at step {self.steps}: {intervention.value}"
        )
    
    def get_metrics_summary(self, window: int = 1000) -> Dict:
        """Get summary statistics over recent window."""
        if len(self.metrics_history) == 0:
            return {}
        
        recent = list(self.metrics_history)[-window:]
        
        return {
            'mean_theta_frequency': np.mean([m.theta_frequency for m in recent]),
            'mean_n_back_accuracy': np.mean([m.n_back_accuracy for m in recent]),
            'mean_task_accuracy': np.mean([m.task_accuracy for m in recent]),
            'mean_firing_rate': np.mean([m.mean_firing_rate for m in recent]),
            'active_alerts': list(self.active_alerts),
            'num_interventions': len(self.intervention_history),
        }


class Stage1Monitor(ContinuousMonitor):
    """
    Specialized monitoring for Stage 1 (highest risk).
    
    More stringent checks and more frequent monitoring than other stages.
    """
    
    def __init__(self, **kwargs):
        # More frequent checks for Stage 1
        kwargs.setdefault('check_interval', 500)  # Check every 500 steps
        super().__init__(**kwargs)
        
        # Stage 1 specific thresholds (more conservative)
        self.wm_critical_threshold = 0.65  # Higher than general 0.60
        self.theta_variance_threshold = 0.18  # Stricter than general 0.20
        self.performance_drop_threshold = 0.08  # Stricter than general 0.10
    
    def _check_wm_collapse(self, metrics: MonitoringMetrics) -> bool:
        """Stage 1 version with stricter threshold."""
        if metrics.n_back_accuracy < self.wm_critical_threshold:
            return True
        
        # Check for rapid decline (same as parent)
        if len(self.metrics_history) > 100:
            recent = [m.n_back_accuracy for m in list(self.metrics_history)[-100:]]
            if len(recent) > 10:
                early = np.mean(recent[:50])
                late = np.mean(recent[-50:])
                if early - late > 0.12:  # 12% rapid drop (stricter)
                    return True
        
        return False
    
    def _check_oscillator_instability(
        self, metrics: MonitoringMetrics
    ) -> bool:
        """Stage 1 version with stricter threshold."""
        # Tighter frequency range
        if not (7.0 <= metrics.theta_frequency <= 8.0):
            return True
        
        # Lower variance tolerance
        if metrics.theta_variance > self.theta_variance_threshold:
            return True
        
        # Higher phase locking requirement
        if metrics.gamma_theta_phase_locking < 0.35:  # Stricter than 0.3
            return True
        
        return False
