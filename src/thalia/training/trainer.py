"""Training loop with integrated health monitoring and logging."""

from __future__ import annotations

import time
from dataclasses import dataclass

from thalia.brain.brain import Brain
from thalia.training.checkpointing.checkpoint import save_checkpoint
from thalia.training.encoding.spike_decoder import ReadoutGroup
from thalia.training.monitoring.health_monitor import HealthMonitor
from thalia.training.monitoring.training_logger import LoggerConfig, TrainingLogger
from thalia.training.tasks.base import Task, TrialResult
from thalia.training.tasks.pattern_association import ITI_STEPS, REWARD_FORWARD_STEPS
from thalia.training.trial import run_trial


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Configuration for a training run."""

    n_trials: int = 1000
    log_interval: int = 10
    diagnostics_interval: int = 100
    checkpoint_interval: int = 200
    max_criticals_for_stop: int = 5
    device: str = "cpu"
    log_dir: str = "data/training"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_loop(
    brain: Brain,
    task: Task,
    readout_groups: list[ReadoutGroup],
    config: TrainConfig,
) -> list[TrialResult]:
    """Run the training loop with health monitoring and CSV logging.

    Args:
        brain: Brain instance.
        task: Task to train on.
        readout_groups: Readout groups for spike decoding.
        config: Training configuration.

    Returns:
        List of all trial results.
    """
    # Ensure learning is active for training
    brain.set_learning_disabled(False)

    monitor = HealthMonitor(brain)
    logger = TrainingLogger(LoggerConfig(log_dir=config.log_dir))

    results: list[TrialResult] = []
    t0 = time.perf_counter()
    stopped_early = False

    try:
        for trial_idx in range(config.n_trials):
            # 1. Generate and run trial (with Tier 1 monitoring)
            trial = task.generate_trial(trial_idx)
            spike_counts = run_trial(brain, task, trial, readout_groups, monitor)

            # 2. Evaluate response
            result = task.evaluate(trial, spike_counts)
            results.append(result)

            # 3. Tier 1 health summary
            health_summary = monitor.end_trial()

            # 4. Deliver reward and let VTA process it
            brain.deliver_reward(result.reward)
            for _ in range(REWARD_FORWARD_STEPS):
                brain.forward()

            # 5. Inter-trial interval (spontaneous dynamics / consolidation)
            for _ in range(ITI_STEPS):
                brain.forward()

            # 6. Log trial to CSV
            logger.log_trial(trial_idx, result, health_summary, results)

            # 7. Console progress
            if (trial_idx + 1) % config.log_interval == 0:
                _log_progress(trial_idx + 1, results, config.log_interval, t0)

            # 8. Tier 2 periodic full diagnostics
            if (trial_idx + 1) % config.diagnostics_interval == 0:
                print(f"  Running Tier 2 diagnostics at trial {trial_idx + 1}...")
                report = monitor.run_full_diagnostics()
                n_critical = len(report.health.critical_issues)
                n_warning = len(report.health.warnings)
                brain_state = report.health.global_brain_state
                logger.log_diagnostics(trial_idx, n_critical, n_warning, brain_state)
                print(
                    f"  Diagnostics: {n_critical} criticals, "
                    f"{n_warning} warnings, state={brain_state}"
                )

                # Early stopping on critical health issues
                if n_critical >= config.max_criticals_for_stop:
                    msg = (
                        f"EARLY STOP: {n_critical} critical issues detected "
                        f"(threshold={config.max_criticals_for_stop})"
                    )
                    print(f"\n=== {msg} ===")
                    logger.log_alert(trial_idx, msg)
                    stopped_early = True
                    break

            # 9. Checkpoint
            if (trial_idx + 1) % config.checkpoint_interval == 0:
                ckpt_path = logger.run_dir / f"checkpoint_trial_{trial_idx + 1}.pt"
                save_checkpoint(
                    brain, ckpt_path,
                    metadata={"trial_idx": trial_idx, "n_results": len(results)},
                )
                print(f"  Checkpoint saved: {ckpt_path.name}")

            # 10. Convergence check
            if task.is_learned(results):
                print(f"\n=== TASK LEARNED at trial {trial_idx + 1} ===")
                logger.log_event(trial_idx, "TASK_LEARNED")
                break
    finally:
        logger.close()

    if not stopped_early:
        print(f"  Logs saved to: {logger.run_dir}")

    return results


def _log_progress(
    trial_num: int,
    results: list[TrialResult],
    window: int,
    t0: float,
) -> None:
    """Print a one-line progress summary for the last *window* trials."""
    recent = results[-window:]
    accuracy = sum(1 for r in recent if r.correct) / len(recent)
    avg_reward = sum(r.reward for r in recent) / len(recent)
    elapsed = time.perf_counter() - t0
    print(
        f"Trial {trial_num:5d} | "
        f"acc {accuracy:.0%} | "
        f"reward {avg_reward:+.2f} | "
        f"elapsed {elapsed:.1f}s"
    )
