"""CSV-based training metrics logger.

Writes one row per trial to a CSV file, plus a separate events log for
diagnostics summaries, alerts, and milestones.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from thalia.training.monitoring.health_monitor import TrialHealthSummary
from thalia.training.tasks.base import TrialResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoggerConfig:
    """Configuration for training logger output.

    Attributes:
        log_dir: Directory to write CSV files into.
        trial_csv: Filename for per-trial metrics.
        events_log: Filename for event / diagnostics log.
    """

    log_dir: str = "data/training"
    trial_csv: str = "trials.csv"
    events_log: str = "events.log"


# ---------------------------------------------------------------------------
# Trial CSV columns
# ---------------------------------------------------------------------------

_TRIAL_COLUMNS = [
    "trial",
    "pattern",
    "correct",
    "reward",
    "count_a",
    "count_b",
    "margin",
    "accuracy_10",
    "accuracy_100",
    "n_silent_pops",
    "n_hyperactive_pops",
    "wall_clock_s",
]


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TrainingLogger:
    """Writes per-trial metrics to CSV and events to a text log.

    Usage::

        logger = TrainingLogger(LoggerConfig(log_dir="data/training/run_01"))
        # ... in loop ...
        logger.log_trial(idx, result, health_summary, results_so_far)
        # ... periodically ...
        logger.log_diagnostics(idx, n_critical, n_warning, brain_state)
        logger.close()
    """

    def __init__(self, config: LoggerConfig | None = None) -> None:
        if config is None:
            config = LoggerConfig()
        self.config = config

        # Build timestamped subdirectory
        ts = time.strftime("%Y-%m-%dT%H%M%S")
        self.run_dir = Path(config.log_dir) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Open CSV
        csv_path = self.run_dir / config.trial_csv
        self._csv_file: TextIO = open(csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_TRIAL_COLUMNS)
        self._csv_writer.writeheader()

        # Open events log
        events_path = self.run_dir / config.events_log
        self._events_file: TextIO = open(events_path, "w", encoding="utf-8")  # noqa: SIM115
        self._events_file.write(f"# Training events log — {ts}\n")

        self._t0 = time.perf_counter()

    # -- Per-trial -----------------------------------------------------------

    def log_trial(
        self,
        trial_idx: int,
        result: TrialResult,
        health: TrialHealthSummary,
        all_results: list[TrialResult],
    ) -> None:
        """Write one row to the trials CSV.

        Args:
            trial_idx: Zero-based trial index.
            result: Evaluation result for this trial.
            health: Tier 1 health summary for this trial.
            all_results: Full list of results so far (for running accuracy).
        """
        n = len(all_results)
        acc_10 = _accuracy(all_results, 10)
        acc_100 = _accuracy(all_results, 100)

        row: dict[str, str | int] = {
            "trial": trial_idx,
            "pattern": str(result.metrics.get("pattern", "")),
            "correct": int(result.correct),
            "reward": f"{result.reward:.2f}",
            "count_a": int(result.metrics.get("count_a", 0)),
            "count_b": int(result.metrics.get("count_b", 0)),
            "margin": int(result.metrics.get("margin", 0)),
            "accuracy_10": f"{acc_10:.3f}",
            "accuracy_100": f"{acc_100:.3f}" if n >= 100 else "",
            "n_silent_pops": len(health.silent_populations),
            "n_hyperactive_pops": len(health.hyperactive_populations),
            "wall_clock_s": f"{time.perf_counter() - self._t0:.1f}",
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # -- Events / diagnostics ------------------------------------------------

    def log_diagnostics(
        self,
        trial_idx: int,
        n_critical: int,
        n_warning: int,
        brain_state: str,
    ) -> None:
        """Log a Tier 2 diagnostics summary."""
        self._write_event(
            trial_idx,
            f"DIAGNOSTICS  criticals={n_critical}  warnings={n_warning}  "
            f"brain_state={brain_state}",
        )

    def log_alert(self, trial_idx: int, message: str) -> None:
        """Log a health alert (e.g. new criticals appeared)."""
        self._write_event(trial_idx, f"ALERT  {message}")

    def log_event(self, trial_idx: int, message: str) -> None:
        """Log a generic milestone event."""
        self._write_event(trial_idx, message)

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open files."""
        self._csv_file.close()
        self._events_file.close()

    # -- Internal ------------------------------------------------------------

    def _write_event(self, trial_idx: int, message: str) -> None:
        elapsed = time.perf_counter() - self._t0
        line = f"[trial {trial_idx:5d} | {elapsed:8.1f}s] {message}\n"
        self._events_file.write(line)
        self._events_file.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accuracy(results: list[TrialResult], window: int) -> float:
    """Compute accuracy over the last *window* results."""
    if not results:
        return 0.0
    recent = results[-window:]
    return sum(1 for r in recent if r.correct) / len(recent)
