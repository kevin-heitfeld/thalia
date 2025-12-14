"""
Stage Transition Gates - Hard criteria that must be met before advancing.

This module implements the survival checklists that prevent premature stage
transitions. Consensus design from expert review + ChatGPT engineering analysis.

Critical Design Principle:
    "You cannot have a stage where a single failure cascades and destroys
    the entire system." - All modules must degrade gracefully except
    critical infrastructure (WM, oscillators, replay).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from enum import Enum


class GateDecision(Enum):
    """Gate decision outcomes."""
    PROCEED = "proceed"
    EXTEND = "extend_stage"
    ROLLBACK = "rollback_checkpoint"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class GateResult:
    """Result of stage gate evaluation."""
    decision: GateDecision
    passed: bool
    failures: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]


class Stage1SurvivalGate:
    """
    Hard gate for Stage 1 → Stage 2 transition.
    
    Stage 1 is the highest-risk transition point in the curriculum.
    ALL criteria must be met. If ANY fails, extend Stage 1.
    
    Architecture:
        - Oscillator stability (theta, gamma coupling)
        - Working memory capacity (n-back, attractors)
        - Cross-modal interference (phonology/vision)
        - Global health (firing rates, dopamine, replay)
    """
    
    # Oscillator stability criteria
    THETA_FREQ_RANGE = (6.5, 8.5)  # Hz
    THETA_MAX_VARIANCE = 0.15  # 15% max variance
    GAMMA_THETA_MIN_LOCKING = 0.4  # Phase-locking value
    MAX_FREQ_DRIFT = 0.05  # 5% drift over observation window
    
    # Working memory criteria
    N_BACK_MIN_ACCURACY = 0.80  # 80% on 2-back
    MIN_STABLE_ATTRACTORS = 3  # Reliably retrievable states
    MIN_CONSOLIDATION_CYCLES = 3  # Must be stable across cycles
    
    # Interference criteria
    MAX_CROSS_ENTROPY_DRIFT = 0.20  # Phonology/vision interference
    
    # Global health
    FIRING_RATE_RANGE = (0.05, 0.15)  # Hz per neuron
    MIN_REPLAY_IMPROVEMENT = 0.02  # 2% improvement from consolidation
    
    def __init__(self, observation_window: int = 10000):
        """
        Args:
            observation_window: Steps to observe for stability metrics
        """
        self.observation_window = observation_window
        self.reset()
    
    def reset(self):
        """Reset monitoring state."""
        self.history = {
            'theta_freq': [],
            'gamma_theta_locking': [],
            'n_back_accuracy': [],
            'firing_rates': [],
            'replay_improvements': [],
        }
    
    def update(self, brain, metrics: Dict[str, float]):
        """
        Update gate monitoring with latest metrics.
        
        Args:
            brain: Brain instance
            metrics: Latest training metrics
        """
        # Track oscillator metrics
        if 'theta_frequency' in metrics:
            self.history['theta_freq'].append(metrics['theta_frequency'])
        
        if 'gamma_theta_phase_locking' in metrics:
            self.history['gamma_theta_locking'].append(
                metrics['gamma_theta_phase_locking']
            )
        
        # Track WM metrics
        if 'n_back_accuracy' in metrics:
            self.history['n_back_accuracy'].append(metrics['n_back_accuracy'])
        
        # Track global health
        if 'mean_firing_rate' in metrics:
            self.history['firing_rates'].append(metrics['mean_firing_rate'])
        
        if 'replay_improvement' in metrics:
            self.history['replay_improvements'].append(
                metrics['replay_improvement']
            )
        
        # Trim to observation window
        for key in self.history:
            if len(self.history[key]) > self.observation_window:
                self.history[key] = self.history[key][-self.observation_window:]
    
    def evaluate(self, brain) -> GateResult:
        """
        Evaluate all Stage 1 survival criteria.
        
        Returns:
            GateResult with decision and detailed metrics
        """
        failures = []
        recommendations = []
        metrics = {}
        
        # 1. Check oscillator stability
        osc_pass, osc_failures, osc_metrics = self._check_oscillators()
        failures.extend(osc_failures)
        metrics.update(osc_metrics)
        
        if not osc_pass:
            recommendations.append("Reduce WM load to stabilize oscillations")
            recommendations.append("Increase consolidation frequency")
        
        # 2. Check working memory
        wm_pass, wm_failures, wm_metrics = self._check_working_memory()
        failures.extend(wm_failures)
        metrics.update(wm_metrics)
        
        if not wm_pass:
            recommendations.append("Extend 2-back training period")
            recommendations.append("Reduce task complexity temporarily")
        
        # 3. Check interference
        int_pass, int_failures, int_metrics = self._check_interference(brain)
        failures.extend(int_failures)
        metrics.update(int_metrics)
        
        if not int_pass:
            recommendations.append("Enable temporal separation of modalities")
            recommendations.append("Reduce simultaneous learning demands")
        
        # 4. Check global health
        health_pass, health_failures, health_metrics = self._check_global_health(brain)
        failures.extend(health_failures)
        metrics.update(health_metrics)
        
        if not health_pass:
            recommendations.append("Emergency consolidation required")
            recommendations.append("Check E/I balance in failing regions")
        
        # Determine decision
        if len(failures) == 0:
            decision = GateDecision.PROCEED
            passed = True
        elif len(failures) > 5:
            decision = GateDecision.EMERGENCY_STOP
            passed = False
            recommendations.insert(0, "CRITICAL: Multiple system failures detected")
        else:
            decision = GateDecision.EXTEND
            passed = False
        
        return GateResult(
            decision=decision,
            passed=passed,
            failures=failures,
            recommendations=recommendations,
            metrics=metrics
        )
    
    def _check_oscillators(self) -> Tuple[bool, List[str], Dict[str, float]]:
        """Check oscillator stability criteria."""
        failures = []
        metrics = {}
        
        if len(self.history['theta_freq']) < 100:
            failures.append("Insufficient oscillator data")
            return False, failures, metrics
        
        # Theta frequency in range
        theta_freqs = np.array(self.history['theta_freq'][-1000:])
        mean_theta = theta_freqs.mean()
        theta_var = theta_freqs.std() / mean_theta
        
        metrics['mean_theta_frequency'] = float(mean_theta)
        metrics['theta_variance'] = float(theta_var)
        
        if not (self.THETA_FREQ_RANGE[0] <= mean_theta <= self.THETA_FREQ_RANGE[1]):
            failures.append(
                f"Theta frequency {mean_theta:.2f}Hz out of range "
                f"{self.THETA_FREQ_RANGE}"
            )
        
        if theta_var > self.THETA_MAX_VARIANCE:
            failures.append(
                f"Theta variance {theta_var:.3f} exceeds max "
                f"{self.THETA_MAX_VARIANCE}"
            )
        
        # Frequency drift check
        if len(theta_freqs) >= 1000:
            early_mean = theta_freqs[:500].mean()
            late_mean = theta_freqs[-500:].mean()
            drift = abs(late_mean - early_mean) / early_mean
            metrics['theta_drift'] = float(drift)
            
            if drift > self.MAX_FREQ_DRIFT:
                failures.append(
                    f"Theta drift {drift:.3f} exceeds max {self.MAX_FREQ_DRIFT}"
                )
        
        # Gamma-theta phase locking
        if len(self.history['gamma_theta_locking']) >= 100:
            locking = np.array(self.history['gamma_theta_locking'][-1000:])
            mean_locking = locking.mean()
            metrics['gamma_theta_locking'] = float(mean_locking)
            
            if mean_locking < self.GAMMA_THETA_MIN_LOCKING:
                failures.append(
                    f"Gamma-theta locking {mean_locking:.3f} below min "
                    f"{self.GAMMA_THETA_MIN_LOCKING}"
                )
        else:
            failures.append("Insufficient gamma-theta locking data")
        
        return len(failures) == 0, failures, metrics
    
    def _check_working_memory(self) -> Tuple[bool, List[str], Dict[str, float]]:
        """Check working memory capacity criteria."""
        failures = []
        metrics = {}
        
        if len(self.history['n_back_accuracy']) < 100:
            failures.append("Insufficient WM performance data")
            return False, failures, metrics
        
        # N-back accuracy (rolling window)
        accuracies = np.array(self.history['n_back_accuracy'][-1000:])
        mean_accuracy = accuracies.mean()
        metrics['n_back_2_accuracy'] = float(mean_accuracy)
        
        if mean_accuracy < self.N_BACK_MIN_ACCURACY:
            failures.append(
                f"2-back accuracy {mean_accuracy:.3f} below min "
                f"{self.N_BACK_MIN_ACCURACY}"
            )
        
        # Check for performance decay
        if len(accuracies) >= 1000:
            # Split into segments
            segment_size = len(accuracies) // 5
            segments = [
                accuracies[i*segment_size:(i+1)*segment_size].mean()
                for i in range(5)
            ]
            
            # Check if declining
            if segments[-1] < segments[0] - 0.05:  # 5% decay
                failures.append(
                    f"WM performance decaying: {segments[0]:.3f} → {segments[-1]:.3f}"
                )
            
            metrics['wm_stability'] = float(segments[-1] - segments[0])
        
        return len(failures) == 0, failures, metrics
    
    def _check_interference(
        self, brain
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """Check cross-modal interference."""
        failures = []
        metrics = {}
        
        # Check if both phonology and vision regions exist
        has_phonology = hasattr(brain, 'phonology_region')
        has_vision = hasattr(brain, 'visual_cortex')
        
        if not (has_phonology and has_vision):
            # Can't check interference if regions don't exist
            return True, failures, metrics
        
        # Check for simultaneous performance collapse
        # (both modalities failing at same time indicates interference)
        if hasattr(brain, 'get_modality_performance'):
            phon_perf = brain.get_modality_performance('phonology')
            vis_perf = brain.get_modality_performance('vision')
            
            metrics['phonology_performance'] = phon_perf
            metrics['vision_performance'] = vis_perf
            
            if phon_perf < 0.7 and vis_perf < 0.7:
                failures.append(
                    f"Simultaneous collapse detected: "
                    f"phonology={phon_perf:.2f}, vision={vis_perf:.2f}"
                )
        
        return len(failures) == 0, failures, metrics
    
    def _check_global_health(
        self, brain
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """Check global health metrics."""
        failures = []
        metrics = {}
        
        # Check firing rates across regions
        if len(self.history['firing_rates']) >= 100:
            firing_rates = np.array(self.history['firing_rates'][-1000:])
            mean_fr = firing_rates.mean()
            metrics['mean_firing_rate'] = float(mean_fr)
            
            if not (self.FIRING_RATE_RANGE[0] <= mean_fr <= self.FIRING_RATE_RANGE[1]):
                failures.append(
                    f"Mean firing rate {mean_fr:.4f} out of range "
                    f"{self.FIRING_RATE_RANGE}"
                )
        
        # Check replay effectiveness
        if len(self.history['replay_improvements']) >= 3:
            improvements = self.history['replay_improvements'][-3:]
            mean_improvement = np.mean(improvements)
            metrics['replay_improvement'] = float(mean_improvement)
            
            if mean_improvement < self.MIN_REPLAY_IMPROVEMENT:
                failures.append(
                    f"Replay improvement {mean_improvement:.3f} below min "
                    f"{self.MIN_REPLAY_IMPROVEMENT}"
                )
        
        # Check dopamine system health
        if hasattr(brain, 'neuromodulation'):
            dopamine_level = brain.neuromodulation.get_dopamine()
            metrics['dopamine_level'] = float(dopamine_level)
            
            # Check for saturation (chronic high/low)
            if dopamine_level > 0.95 or dopamine_level < 0.05:
                failures.append(
                    f"Dopamine saturation detected: {dopamine_level:.3f}"
                )
        
        return len(failures) == 0, failures, metrics


class GracefulDegradationManager:
    """
    Handles module failures with appropriate responses.
    
    Critical Design Principle:
        Non-critical systems (language, vision) can degrade gracefully.
        Critical systems (WM, oscillators, replay) trigger emergency stops.
    
    Kill-Switch Map:
        ✅ DEGRADABLE: language, grammar, reading
        ⚠️ LIMITED: vision, phonology
        ❌ CRITICAL: working_memory, oscillators, replay
    """
    
    CRITICAL_SYSTEMS = {'working_memory', 'oscillators', 'replay'}
    DEGRADABLE_SYSTEMS = {'language', 'grammar', 'reading'}
    LIMITED_DEGRADATION = {'vision', 'phonology'}
    
    # Failure thresholds (performance drop percentages)
    CRITICAL_THRESHOLD = 0.30  # 30% drop triggers emergency
    LIMITED_THRESHOLD = 0.50   # 50% drop triggers partial shutdown
    DEGRADABLE_THRESHOLD = 0.70  # 70% drop before intervention
    
    def __init__(self):
        """Initialize degradation manager."""
        self.degraded_modules = set()
        self.failure_history = {}
    
    def handle_module_failure(
        self,
        module_name: str,
        baseline_performance: float,
        current_performance: float
    ) -> Dict[str, any]:
        """
        Route module failures to appropriate responses.
        
        Args:
            module_name: Name of failing module
            baseline_performance: Expected performance level
            current_performance: Actual performance level
        
        Returns:
            Dict with action, severity, and recommendations
        """
        # Calculate performance drop
        if baseline_performance > 0:
            drop = (baseline_performance - current_performance) / baseline_performance
        else:
            drop = 0.0
        
        # Record failure
        if module_name not in self.failure_history:
            self.failure_history[module_name] = []
        self.failure_history[module_name].append(drop)
        
        # Route to appropriate handler
        if module_name in self.CRITICAL_SYSTEMS:
            return self._handle_critical_failure(module_name, drop)
        
        elif module_name in self.DEGRADABLE_SYSTEMS:
            return self._handle_degradable_failure(module_name, drop)
        
        elif module_name in self.LIMITED_DEGRADATION:
            return self._handle_limited_failure(module_name, drop)
        
        else:
            # Unknown module - treat as degradable with warning
            return {
                'action': 'GRACEFUL_DEGRADATION',
                'severity': 'MEDIUM',
                'module': module_name,
                'alert': f'UNKNOWN_MODULE_{module_name}_DEGRADED',
                'recommendations': ['Verify module classification']
            }
    
    def _handle_critical_failure(
        self, module_name: str, drop: float
    ) -> Dict[str, any]:
        """Handle failure of critical system."""
        if drop > self.CRITICAL_THRESHOLD:
            return {
                'action': 'EMERGENCY_STOP',
                'severity': 'CRITICAL',
                'module': module_name,
                'freeze_learning': True,
                'rollback_to_checkpoint': True,
                'alert': f'CRITICAL_FAILURE_{module_name}',
                'recommendations': [
                    'Rollback to last stable checkpoint',
                    'Reduce cognitive load',
                    'Emergency consolidation',
                    'Check oscillator stability' if module_name == 'oscillators' else 'Check WM capacity'
                ]
            }
        else:
            return {
                'action': 'HIGH_PRIORITY_INTERVENTION',
                'severity': 'HIGH',
                'module': module_name,
                'reduce_load': True,
                'increase_monitoring': True,
                'alert': f'WARNING_{module_name}_DEGRADING',
                'recommendations': [
                    'Reduce task complexity',
                    'Increase consolidation frequency',
                    'Monitor closely for further degradation'
                ]
            }
    
    def _handle_degradable_failure(
        self, module_name: str, drop: float
    ) -> Dict[str, any]:
        """Handle failure of degradable system."""
        if drop > self.DEGRADABLE_THRESHOLD:
            self.degraded_modules.add(module_name)
            return {
                'action': 'GRACEFUL_DEGRADATION',
                'severity': 'MEDIUM',
                'module': module_name,
                'disable_module': True,
                'continue_learning': True,
                'alert': f'{module_name}_DEGRADED',
                'recommendations': [
                    f'Continue without {module_name}',
                    'System can still think and plan',
                    f'Re-enable {module_name} after consolidation'
                ]
            }
        else:
            return {
                'action': 'MONITOR',
                'severity': 'LOW',
                'module': module_name,
                'continue_normally': True,
                'alert': f'{module_name}_MINOR_DEGRADATION'
            }
    
    def _handle_limited_failure(
        self, module_name: str, drop: float
    ) -> Dict[str, any]:
        """Handle failure of limited degradation system."""
        if drop > self.LIMITED_THRESHOLD:
            self.degraded_modules.add(module_name)
            return {
                'action': 'PARTIAL_SHUTDOWN',
                'severity': 'MEDIUM',
                'module': module_name,
                'reduce_module_load': True,
                'enable_fallback': True,
                'continue_learning': True,
                'alert': f'{module_name}_LIMITED_MODE',
                'recommendations': [
                    f'Reduce {module_name} task complexity',
                    'Enable cross-modal compensation',
                    'Continue with reduced capability'
                ]
            }
        else:
            return {
                'action': 'MONITOR',
                'severity': 'LOW',
                'module': module_name,
                'continue_normally': True,
                'alert': f'{module_name}_MINOR_DEGRADATION'
            }
    
    def get_system_status(self) -> Dict[str, any]:
        """Get overall system degradation status."""
        return {
            'degraded_modules': list(self.degraded_modules),
            'critical_systems_healthy': all(
                module not in self.degraded_modules
                for module in self.CRITICAL_SYSTEMS
            ),
            'num_failures': len(self.failure_history),
            'operational': len(self.degraded_modules & self.CRITICAL_SYSTEMS) == 0
        }
