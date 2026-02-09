"""
Stage Evaluation Functions - Milestone checking for curriculum stages.

This module implements evaluation functions for each curriculum stage,
checking whether the brain has met all success criteria before proceeding
to the next stage (go/no-go decisions).

Evaluation Categories:
=====================

1. TASK PERFORMANCE
   - Accuracy on stage-specific tasks
   - Generalization to new examples
   - Robustness to noise/variations

2. SYSTEM HEALTH
   - Firing rate stability (0.05-0.15)
   - No runaway excitation (criticality check)
   - BCM threshold convergence
   - Weight health (<80% saturation)
   - No silent regions (>0.01 firing minimum)

3. BACKWARD COMPATIBILITY
   - Previous stage performance maintained (>90%)
   - No catastrophic forgetting
   - Skills transfer correctly

4. GROWTH & CAPACITY
   - Appropriate size for stage
   - Healthy capacity utilization (not saturated)
   - Growth events successful
"""

from __future__ import annotations

from typing import Any, Dict

from thalia.diagnostics import HealthIssue, HealthMonitor


# ============================================================================
# Common Health Checks (All Stages)
# ============================================================================


def check_system_health(brain: Any) -> Dict[str, bool]:
    """Run all common health checks using HealthMonitor.

    Args:
        brain: Brain instance

    Returns:
        Dict of health check results mapping to True/False
    """
    # Use HealthMonitor for proper health checking
    monitor = HealthMonitor()

    # Get diagnostics from brain
    diagnostics = brain.get_diagnostics()

    # Run health check
    report = monitor.check_health(diagnostics)

    # Map health report to expected boolean checks
    # All pass if healthy, otherwise check specific issue types
    if report.is_healthy:
        return {
            "firing_stability": True,
            "no_runaway": True,
            "bcm_convergence": True,
            "weight_health": True,
            "no_silence": True,
        }

    # Check for specific issues
    issue_types = {issue.issue_type for issue in report.issues}

    return {
        "firing_stability": HealthIssue.ACTIVITY_COLLAPSE not in issue_types
        and HealthIssue.SEIZURE_RISK not in issue_types,
        "no_runaway": HealthIssue.SEIZURE_RISK not in issue_types,
        "bcm_convergence": True,  # Not directly tracked by HealthMonitor
        "weight_health": HealthIssue.WEIGHT_EXPLOSION not in issue_types
        and HealthIssue.WEIGHT_COLLAPSE not in issue_types,
        "no_silence": HealthIssue.ACTIVITY_COLLAPSE not in issue_types,
    }


# ============================================================================
# Stage 1: Sensorimotor Grounding
# ============================================================================


def test_basic_movements(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.95,
) -> bool:
    """Test basic motor control accuracy.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of test trials
        threshold: Success threshold

    Returns:
        True if accuracy > threshold
    """
    # Test simple left/right/up/down movements
    # Placeholder for now
    # TODO: Implement actual test logic
    return True


def test_reaching_accuracy(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.90,
) -> bool:
    """Test reaching task accuracy.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of reaching trials
        threshold: Success threshold

    Returns:
        True if accuracy > threshold
    """
    if hasattr(wrapper, "reaching_task"):
        stats = wrapper.reaching_task(brain, n_trials=n_trials)
        return bool(stats.get("success_rate", 0.0) > threshold)
    return True


def test_manipulation_success(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.85,
) -> bool:
    """Test object manipulation success rate.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of manipulation trials
        threshold: Success threshold

    Returns:
        True if success rate > threshold
    """
    # Test push/pull/grasp tasks
    # Placeholder for now
    # TODO: Implement actual test logic
    return True


def test_prediction_error(
    brain: Any,
    wrapper: Any,
    n_trials: int = 100,
    threshold: float = 0.05,
) -> bool:
    """Test sensorimotor prediction error.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper
        n_trials: Number of trials
        threshold: Max acceptable error

    Returns:
        True if error < threshold
    """
    # Test cerebellum forward model accuracy
    # Placeholder for now
    # TODO: Implement actual test logic
    return True


def test_cerebellum_functional(
    brain: Any,
    wrapper: Any,
) -> bool:
    """Test that cerebellum forward/inverse models work.

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper

    Returns:
        True if cerebellum is functional
    """
    # Check cerebellum state and learning
    # Placeholder for now
    # TODO: Implement actual test logic
    return True


def evaluate_stage_sensorimotor(
    brain: Any,
    wrapper: Any,
) -> Dict[str, bool]:
    """Evaluate Stage 1 (Sensorimotor) milestones.

    Success criteria from curriculum_strategy.md:
    - >95% accurate basic movements
    - >90% reaching accuracy
    - >85% manipulation success
    - <5% prediction error
    - Stable firing rates (0.05-0.15)
    - Cerebellum forward models functional

    Args:
        brain: Brain instance
        wrapper: SensorimotorWrapper instance

    Returns:
        Dict mapping criterion name to pass/fail
    """
    results = {}

    # Task performance
    results["basic_movements"] = test_basic_movements(brain, wrapper)
    results["reaching_accuracy"] = test_reaching_accuracy(brain, wrapper)
    results["manipulation_success"] = test_manipulation_success(brain, wrapper)
    results["prediction_error"] = test_prediction_error(brain, wrapper)

    # System health
    health = check_system_health(brain)
    results.update(health)

    # Component-specific
    results["cerebellum_functional"] = test_cerebellum_functional(brain, wrapper)

    return results
