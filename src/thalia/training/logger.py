"""Enhanced logging for curriculum training.

This module provides rich logging capabilities for curriculum training,
including stage progress, growth events, consolidation, milestone evaluation,
and comprehensive reporting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Logging levels for curriculum training."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class StageLog:
    """Log data for a single training stage."""

    stage: int
    start_time: float
    end_time: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    step_metrics: List[Dict[str, Any]] = field(default_factory=list)
    growth_events: List[Dict[str, Any]] = field(default_factory=list)
    consolidation_events: List[Dict[str, Any]] = field(default_factory=list)
    milestone_checks: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)

    def duration_seconds(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def duration_hours(self) -> Optional[float]:
        """Calculate stage duration in hours."""
        duration = self.duration_seconds()
        if duration is None:
            return None
        return duration / 3600.0


class CurriculumLogger:
    """Enhanced logging for curriculum training.

    Provides rich logging for:
    - Stage initialization and progress
    - Per-step training metrics
    - Growth events (when/where/why)
    - Consolidation events
    - Milestone evaluation results
    - Stage transitions
    - Comprehensive stage reports

    **Attributes**:
        log_dir: Directory for log files
        log_level: Minimum logging level
        console_output: Whether to print to console
        file_output: Whether to write to files
        current_stage: Currently active stage number
        stage_logs: Dictionary mapping stage number to StageLog
    """

    def __init__(
        self,
        log_dir: str = "logs/curriculum",
        log_level: LogLevel = LogLevel.INFO,
        console_output: bool = True,
        file_output: bool = True,
    ):
        """Initialize curriculum logger.

        **Args**:
            log_dir: Directory for log files
            log_level: Minimum logging level
            console_output: Whether to print to console
            file_output: Whether to write to files
        """
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.console_output = console_output
        self.file_output = file_output

        # Create log directory
        if file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Python logger
        self.logger = logging.getLogger("CurriculumTrainer")
        self.logger.setLevel(getattr(logging, log_level.value))

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.value))
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file_output:
            file_handler = logging.FileHandler(self.log_dir / "curriculum_training.log")
            file_handler.setLevel(getattr(logging, log_level.value))
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # State
        self.current_stage: Optional[int] = None
        self.stage_logs: Dict[int, StageLog] = {}
        self.session_start_time = time.time()

    def log_stage_start(
        self,
        stage: int,
        config: Dict[str, Any],
    ) -> None:
        """Log stage initialization.

        **Args**:
            stage: Stage number
            config: Stage configuration dictionary
        """
        self.current_stage = stage
        self.stage_logs[stage] = StageLog(
            stage=stage,
            start_time=time.time(),
            config=config,
        )

        # Format stage name
        stage_names = {
            -1: "Sensorimotor",
            0: "Sensory Foundations",
            1: "Toddler",
            2: "Grammar & Executive",
            3: "Reading & Theory of Mind",
            4: "Abstract Reasoning",
            5: "Expert Domains",
            6: "LLM-Level",
        }
        stage_name = stage_names.get(stage, f"Stage {stage}")

        # Log start message
        msg = f"\n{'='*80}\n"
        msg += f"[Stage {stage} Start] {stage_name}\n"
        msg += f"  Duration: {config.get('duration_weeks', '?')} weeks\n"

        if "tasks" in config:
            tasks = config["tasks"]
            weights = config.get("task_weights", {})
            msg += "  Tasks:\n"
            for task in tasks:
                weight_pct = weights.get(task, 1.0 / len(tasks)) * 100
                msg += f"    - {task}: {weight_pct:.0f}%\n"

        if "success_criteria" in config:
            msg += "  Success Criteria:\n"
            for metric, threshold in config["success_criteria"].items():
                msg += f"    - {metric}: >{threshold:.2f}\n"

        msg += f"{'='*80}\n"

        self.logger.info(msg)

        # Write JSON snapshot
        if self.file_output:
            self._write_stage_json(stage)

    def log_stage_end(self, stage: int) -> None:
        """Log stage completion.

        **Args**:
            stage: Stage number
        """
        if stage not in self.stage_logs:
            self.logger.warning(f"Stage {stage} not found in logs")
            return

        stage_log = self.stage_logs[stage]
        stage_log.end_time = time.time()

        duration_hours = stage_log.duration_hours()
        msg = f"\n[Stage {stage} End] Duration: {duration_hours:.1f} hours\n"
        self.logger.info(msg)

        # Write final JSON
        if self.file_output:
            self._write_stage_json(stage)

    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log per-step metrics.

        **Args**:
            step: Training step number
            metrics: Dictionary of metric name -> value
        """
        if self.current_stage is None:
            self.logger.warning("No active stage for step logging")
            return

        stage_log = self.stage_logs[self.current_stage]

        # Store metrics
        metric_entry = {
            "step": step,
            "timestamp": time.time(),
            **metrics,
        }
        stage_log.step_metrics.append(metric_entry)

        # Log to console (only every 1000 steps to avoid spam)
        if step % 1000 == 0:
            msg = f"[Step {step}] "
            metric_strs = [f"{k}={v:.3f}" for k, v in metrics.items()]
            msg += ", ".join(metric_strs)
            self.logger.info(msg)

            # Check for health issues
            if "firing_rate" in metrics:
                firing = metrics["firing_rate"]
                if firing > 0.30:
                    self.logger.warning(f"  ⚠️  High firing rate: {firing:.3f}")
                elif firing < 0.02:
                    self.logger.warning(f"  ⚠️  Low firing rate: {firing:.3f}")

            if "capacity" in metrics:
                capacity = metrics["capacity"]
                if capacity > 0.85:
                    self.logger.warning(f"  ⚠️  High capacity: {capacity:.3f}")

    def log_growth_event(
        self,
        region: str,
        n_added: int,
        reason: str,
        step: Optional[int] = None,
    ) -> None:
        """Log when growth happens and why.

        **Args**:
            region: Brain region name
            n_added: Number of neurons added
            reason: Reason for growth
            step: Training step (optional)
        """
        if self.current_stage is None:
            self.logger.warning("No active stage for growth event")
            return

        stage_log = self.stage_logs[self.current_stage]

        # Store event
        event = {
            "step": step,
            "timestamp": time.time(),
            "region": region,
            "n_added": n_added,
            "reason": reason,
        }
        stage_log.growth_events.append(event)

        # Log to console
        msg = f"\n[Growth Event]"
        if step is not None:
            msg += f" Step {step}"
        msg += f" - {region} +{n_added} neurons\n"
        msg += f"  Reason: {reason}\n"
        self.logger.info(msg)

    def log_consolidation(
        self,
        stage_name: str,
        n_patterns: int,
        duration_seconds: float,
        step: Optional[int] = None,
    ) -> None:
        """Log consolidation events.

        **Args**:
            stage_name: Sleep stage name (NREM1, NREM2, NREM3, REM)
            n_patterns: Number of patterns replayed
            duration_seconds: Duration of consolidation
            step: Training step (optional)
        """
        if self.current_stage is None:
            self.logger.warning("No active stage for consolidation")
            return

        stage_log = self.stage_logs[self.current_stage]

        # Store event
        event = {
            "step": step,
            "timestamp": time.time(),
            "stage": stage_name,
            "n_patterns": n_patterns,
            "duration_seconds": duration_seconds,
        }
        stage_log.consolidation_events.append(event)

        # Log to console
        msg = f"\n[Consolidation]"
        if step is not None:
            msg += f" Step {step}"
        msg += f" - {stage_name}\n"
        msg += f"  Patterns replayed: {n_patterns}\n"
        msg += f"  Duration: {duration_seconds:.1f}s\n"
        self.logger.info(msg)

    def log_milestone_evaluation(
        self,
        stage: int,
        results: Dict[str, bool],
        week: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log milestone check results.

        **Args**:
            stage: Stage number being evaluated
            results: Dictionary of milestone name -> passed (bool)
            week: Week number (optional)
            step: Training step (optional)
        """
        if stage not in self.stage_logs:
            self.logger.warning(f"Stage {stage} not found in logs")
            return

        stage_log = self.stage_logs[stage]

        # Store evaluation
        evaluation = {
            "step": step,
            "week": week,
            "timestamp": time.time(),
            "results": results,
        }
        stage_log.milestone_checks.append(evaluation)

        # Count passes/fails
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        all_passed = passed == total

        # Log to console
        msg = f"\n[Milestone Check] Stage {stage}"
        if week is not None:
            msg += f" Week {week}"
        msg += f" - {passed}/{total} passed\n"

        for milestone, result in results.items():
            symbol = "✅" if result else "⚠️ "
            msg += f"  {symbol} {milestone}\n"

        if not all_passed:
            msg += "  Action: EXTENDING STAGE\n"

        self.logger.info(msg)

    def log_transition(
        self,
        old_stage: int,
        new_stage: int,
        reason: str = "Milestones passed",
    ) -> None:
        """Log stage transitions.

        **Args**:
            old_stage: Previous stage number
            new_stage: New stage number
            reason: Reason for transition
        """
        if old_stage in self.stage_logs:
            stage_log = self.stage_logs[old_stage]
            transition = {
                "timestamp": time.time(),
                "old_stage": old_stage,
                "new_stage": new_stage,
                "reason": reason,
            }
            stage_log.transitions.append(transition)

        # Log to console
        msg = f"\n{'='*80}\n"
        msg += f"[Stage Transition] {old_stage} → {new_stage}\n"
        msg += f"  Reason: {reason}\n"
        msg += f"{'='*80}\n"
        self.logger.info(msg)

    def log_stage_extension(
        self,
        stage: int,
        additional_weeks: int,
        reason: str,
    ) -> None:
        """Log stage extension due to milestone failure.

        **Args**:
            stage: Stage being extended
            additional_weeks: Number of weeks added
            reason: Reason for extension
        """
        msg = f"\n[Stage Extension] Stage {stage}\n"
        msg += f"  Adding {additional_weeks} weeks\n"
        msg += f"  Reason: {reason}\n"
        self.logger.warning(msg)

    def generate_stage_report(self, stage: int) -> str:
        """Generate comprehensive stage summary.

        **Args**:
            stage: Stage number

        **Returns**:
            Formatted report string
        """
        if stage not in self.stage_logs:
            return f"No log data for stage {stage}"

        stage_log = self.stage_logs[stage]
        report = []

        # Header
        report.append("=" * 80)
        report.append(f"Stage {stage} Report")
        report.append("=" * 80)
        report.append("")

        # Duration
        duration_hours = stage_log.duration_hours()
        if duration_hours is not None:
            report.append(f"Duration: {duration_hours:.1f} hours")
        else:
            report.append("Duration: In progress")
        report.append("")

        # Configuration
        if stage_log.config:
            report.append("Configuration:")
            for key, value in stage_log.config.items():
                if isinstance(value, dict):
                    report.append(f"  {key}:")
                    for k, v in value.items():
                        report.append(f"    {k}: {v}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        # Training metrics summary
        if stage_log.step_metrics:
            report.append("Training Metrics:")
            last_metrics = stage_log.step_metrics[-1]
            report.append(f"  Total steps: {last_metrics['step']}")

            # Average metrics over last 10% of training
            n_recent = max(1, len(stage_log.step_metrics) // 10)
            recent_metrics = stage_log.step_metrics[-n_recent:]

            metric_names = [k for k in recent_metrics[0].keys() if k not in ["step", "timestamp"]]

            for metric in metric_names:
                values = [m[metric] for m in recent_metrics if metric in m]
                if values:
                    avg = sum(values) / len(values)
                    report.append(f"  {metric}: {avg:.3f} (avg last 10%)")
            report.append("")

        # Growth events
        if stage_log.growth_events:
            report.append(f"Growth Events: {len(stage_log.growth_events)}")
            for event in stage_log.growth_events:
                step_info = f"Step {event['step']}" if event["step"] else "N/A"
                report.append(f"  {step_info} - {event['region']}: " f"+{event['n_added']} neurons")
                report.append(f"    Reason: {event['reason']}")
            report.append("")

        # Consolidation events
        if stage_log.consolidation_events:
            report.append(f"Consolidation Events: {len(stage_log.consolidation_events)}")
            total_patterns = sum(e["n_patterns"] for e in stage_log.consolidation_events)
            total_duration = sum(e["duration_seconds"] for e in stage_log.consolidation_events)
            report.append(f"  Total patterns replayed: {total_patterns}")
            report.append(f"  Total duration: {total_duration:.1f}s")
            report.append("")

        # Milestone checks
        if stage_log.milestone_checks:
            report.append(f"Milestone Checks: {len(stage_log.milestone_checks)}")
            for i, check in enumerate(stage_log.milestone_checks, 1):
                week_info = f"Week {check['week']}" if check["week"] else "N/A"
                passed = sum(1 for v in check["results"].values() if v)
                total = len(check["results"])
                report.append(f"  Check {i} ({week_info}): {passed}/{total} passed")

                for milestone, result in check["results"].items():
                    symbol = "✅" if result else "❌"
                    report.append(f"    {symbol} {milestone}")
            report.append("")

        # Transitions
        if stage_log.transitions:
            report.append("Transitions:")
            for transition in stage_log.transitions:
                report.append(f"  {transition['old_stage']} → {transition['new_stage']}")
                report.append(f"    Reason: {transition['reason']}")
            report.append("")

        # Footer
        report.append("=" * 80)

        return "\n".join(report)

    def generate_session_report(self) -> str:
        """Generate report for entire training session.

        **Returns**:
            Formatted report string
        """
        report = []

        # Header
        report.append("=" * 80)
        report.append("Curriculum Training Session Report")
        report.append("=" * 80)
        report.append("")

        # Session duration
        session_duration = time.time() - self.session_start_time
        report.append(f"Session duration: {session_duration / 3600:.1f} hours")
        report.append(f"Stages completed: {len(self.stage_logs)}")
        report.append("")

        # Per-stage summaries
        for stage in sorted(self.stage_logs.keys()):
            stage_log = self.stage_logs[stage]
            duration_hours = stage_log.duration_hours()

            report.append(f"Stage {stage}:")
            if duration_hours is not None:
                report.append(f"  Duration: {duration_hours:.1f} hours")
            else:
                report.append("  Duration: In progress")

            if stage_log.step_metrics:
                total_steps = stage_log.step_metrics[-1]["step"]
                report.append(f"  Total steps: {total_steps}")

            report.append(f"  Growth events: {len(stage_log.growth_events)}")
            report.append(f"  Consolidation events: {len(stage_log.consolidation_events)}")
            report.append(f"  Milestone checks: {len(stage_log.milestone_checks)}")
            report.append("")

        # Footer
        report.append("=" * 80)

        return "\n".join(report)

    def _write_stage_json(self, stage: int) -> None:
        """Write stage log to JSON file.

        **Args**:
            stage: Stage number
        """
        if stage not in self.stage_logs:
            return

        stage_log = self.stage_logs[stage]
        filename = self.log_dir / f"stage_{stage}_log.json"

        # Convert to dict
        log_dict = {
            "stage": stage_log.stage,
            "start_time": stage_log.start_time,
            "end_time": stage_log.end_time,
            "duration_hours": stage_log.duration_hours(),
            "config": stage_log.config,
            "step_metrics": stage_log.step_metrics,
            "growth_events": stage_log.growth_events,
            "consolidation_events": stage_log.consolidation_events,
            "milestone_checks": stage_log.milestone_checks,
            "transitions": stage_log.transitions,
        }

        with open(filename, "w") as f:
            json.dump(log_dict, f, indent=2)

    def save_session(self, filename: Optional[str] = None) -> None:
        """Save entire session to JSON.

        **Args**:
            filename: Output filename (default: session_<timestamp>.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"

        filepath = self.log_dir / filename

        session_data: dict[str, Any] = {
            "session_start_time": self.session_start_time,
            "session_duration_hours": (time.time() - self.session_start_time) / 3600,
            "stages": {},
        }

        for stage, stage_log in self.stage_logs.items():
            session_data["stages"][str(stage)] = {
                "start_time": stage_log.start_time,
                "end_time": stage_log.end_time,
                "duration_hours": stage_log.duration_hours(),
                "config": stage_log.config,
                "n_steps": len(stage_log.step_metrics),
                "n_growth_events": len(stage_log.growth_events),
                "n_consolidation_events": len(stage_log.consolidation_events),
                "n_milestone_checks": len(stage_log.milestone_checks),
                "n_transitions": len(stage_log.transitions),
            }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

        self.logger.info(f"Session saved to {filepath}")
