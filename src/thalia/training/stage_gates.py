"""
Stage Transition Gates - Hard criteria that must be met before advancing.

This module implements the survival checklists that prevent premature stage
transitions. Consensus design from expert review + ChatGPT engineering analysis.

Critical Design Principle:
    "You cannot have a stage where a single failure cascades and destroys
    the entire system." - All modules must degrade gracefully except
    critical infrastructure (WM, oscillators, replay).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from .constants import (
    CRITICAL_SYSTEMS,
    DEGRADABLE_SYSTEMS,
    LIMITED_DEGRADATION,
)


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

    def __init__(self):
        """Initialize degradation manager."""
        self.degraded_modules = set()
        self.failure_history = {}

    def handle_module_failure(
        self, module_name: str, baseline_performance: float, current_performance: float
    ) -> Dict[str, Any]:
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
        if module_name in CRITICAL_SYSTEMS:
            return self._handle_critical_failure(module_name, drop)

        elif module_name in DEGRADABLE_SYSTEMS:
            return self._handle_degradable_failure(module_name, drop)

        elif module_name in LIMITED_DEGRADATION:
            return self._handle_limited_failure(module_name, drop)

        else:
            # Unknown module - treat as degradable with warning
            return {
                "action": "GRACEFUL_DEGRADATION",
                "severity": "MEDIUM",
                "module": module_name,
                "alert": f"UNKNOWN_MODULE_{module_name}_DEGRADED",
                "recommendations": ["Verify module classification"],
            }

    def _handle_critical_failure(self, module_name: str, drop: float, threshold: float = 0.30) -> Dict[str, Any]:
        """Handle failure of critical system."""
        if drop > threshold:
            return {
                "action": "EMERGENCY_STOP",
                "severity": "CRITICAL",
                "module": module_name,
                "freeze_learning": True,
                "rollback_to_checkpoint": True,
                "alert": f"CRITICAL_FAILURE_{module_name}",
                "recommendations": [
                    "Rollback to last stable checkpoint",
                    "Reduce cognitive load",
                    "Emergency consolidation",
                    (
                        "Check oscillator stability"
                        if module_name == "oscillators"
                        else "Check WM capacity"
                    ),
                ],
            }
        else:
            return {
                "action": "HIGH_PRIORITY_INTERVENTION",
                "severity": "HIGH",
                "module": module_name,
                "reduce_load": True,
                "increase_monitoring": True,
                "alert": f"WARNING_{module_name}_DEGRADING",
                "recommendations": [
                    "Reduce task complexity",
                    "Increase consolidation frequency",
                    "Monitor closely for further degradation",
                ],
            }

    def _handle_degradable_failure(self, module_name: str, drop: float, threshold: float = 0.70) -> Dict[str, Any]:
        """Handle failure of degradable system."""
        if drop > threshold:
            self.degraded_modules.add(module_name)
            return {
                "action": "GRACEFUL_DEGRADATION",
                "severity": "MEDIUM",
                "module": module_name,
                "disable_module": True,
                "continue_learning": True,
                "alert": f"{module_name}_DEGRADED",
                "recommendations": [
                    f"Continue without {module_name}",
                    "System can still think and plan",
                    f"Re-enable {module_name} after consolidation",
                ],
            }
        else:
            return {
                "action": "MONITOR",
                "severity": "LOW",
                "module": module_name,
                "continue_normally": True,
                "alert": f"{module_name}_MINOR_DEGRADATION",
            }

    def _handle_limited_failure(self, module_name: str, drop: float, threshold: float = 0.50) -> Dict[str, Any]:
        """Handle failure of limited degradation system."""
        if drop > threshold:
            self.degraded_modules.add(module_name)
            return {
                "action": "PARTIAL_SHUTDOWN",
                "severity": "MEDIUM",
                "module": module_name,
                "reduce_module_load": True,
                "enable_fallback": True,
                "continue_learning": True,
                "alert": f"{module_name}_LIMITED_MODE",
                "recommendations": [
                    f"Reduce {module_name} task complexity",
                    "Enable cross-modal compensation",
                    "Continue with reduced capability",
                ],
            }
        else:
            return {
                "action": "MONITOR",
                "severity": "LOW",
                "module": module_name,
                "continue_normally": True,
                "alert": f"{module_name}_MINOR_DEGRADATION",
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system degradation status."""
        return {
            "degraded_modules": list(self.degraded_modules),
            "critical_systems_healthy": all(
                module not in self.degraded_modules for module in CRITICAL_SYSTEMS
            ),
            "num_failures": len(self.failure_history),
            "operational": len(self.degraded_modules & CRITICAL_SYSTEMS) == 0,
        }
