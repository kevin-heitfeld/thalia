"""
Diagnostics and Logging System for Thalia Brain Simulation

This module provides a centralized, configurable diagnostics system for
monitoring and debugging brain region activity, learning, and decision-making.

Features:
=========
1. DIAGNOSTIC LEVELS: Control verbosity (OFF, SUMMARY, DETAILED, TRACE)
2. STRUCTURED DATA: All diagnostics return typed dictionaries
3. PER-COMPONENT CONFIG: Enable/disable diagnostics for specific regions
4. AGGREGATION: BrainSystem can collect diagnostics from all regions
5. HISTORY: Optional rolling history for time-series analysis

Usage:
======
    from thalia.core.diagnostics import DiagnosticsManager, DiagnosticLevel

    # Create manager with desired level
    diag = DiagnosticsManager(level=DiagnosticLevel.DETAILED)

    # Configure per-component
    diag.configure_component("striatum", enabled=True, level=DiagnosticLevel.TRACE)
    diag.configure_component("hippocampus", enabled=True)
    diag.configure_component("cortex", enabled=False)

    # Record diagnostics
    diag.record("striatum", {"d1_mean": 0.5, "d2_mean": 0.4, "net": 0.1})

    # Get summary
    summary = diag.get_summary()

    # Print formatted output
    diag.print_trial_summary(trial_num=1, is_match=True, action=0, correct=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from collections import deque
import time


class DiagnosticLevel(Enum):
    """Verbosity levels for diagnostics."""
    OFF = auto()       # No diagnostics
    SUMMARY = auto()   # Epoch-level summaries only
    DETAILED = auto()  # Per-trial key metrics
    TRACE = auto()     # Full per-timestep traces (expensive!)


@dataclass
class ComponentConfig:
    """Configuration for a single component's diagnostics."""
    enabled: bool = True
    level: Optional[DiagnosticLevel] = None  # None = use global level
    history_size: int = 100  # Rolling history buffer size


@dataclass
class DiagnosticsConfig:
    """Global diagnostics configuration."""
    level: DiagnosticLevel = DiagnosticLevel.SUMMARY
    print_to_console: bool = True
    collect_history: bool = False
    history_size: int = 1000
    timestamp_entries: bool = False
    # Component-specific configs
    components: Dict[str, ComponentConfig] = field(default_factory=dict)


class DiagnosticsManager:
    """
    Centralized diagnostics manager for the brain simulation.

    Collects, stores, and formats diagnostic information from all brain
    components (regions, pathways, learning systems).

    Example:
        diag = DiagnosticsManager(level=DiagnosticLevel.DETAILED)

        # During trial
        diag.record("striatum", {
            "d1_mean": 0.5,
            "d2_mean": 0.4,
            "action": 0,
            "exploring": False,
        })

        # End of trial
        diag.print_trial_summary(trial=1, gt="M", action="M", correct=True)

        # End of epoch
        diag.print_epoch_summary(epoch=1, accuracy=0.75)
    """

    def __init__(
        self,
        level: DiagnosticLevel = DiagnosticLevel.SUMMARY,
        config: Optional[DiagnosticsConfig] = None,
    ):
        self.config = config or DiagnosticsConfig(level=level)
        self._current: Dict[str, Dict[str, Any]] = {}  # Current trial data
        self._history: Dict[str, deque] = {}  # Rolling history per component
        self._epoch_data: List[Dict[str, Any]] = []  # Epoch summaries
        self._trial_count = 0
        self._epoch_count = 0
        self._start_time = time.time()

    def configure_component(
        self,
        name: str,
        enabled: bool = True,
        level: Optional[DiagnosticLevel] = None,
        history_size: int = 100,
    ) -> None:
        """Configure diagnostics for a specific component."""
        self.config.components[name] = ComponentConfig(
            enabled=enabled,
            level=level,
            history_size=history_size,
        )
        if name not in self._history:
            self._history[name] = deque(maxlen=history_size)

    def get_level(self, component: Optional[str] = None) -> DiagnosticLevel:
        """Get effective diagnostic level for a component."""
        if component and component in self.config.components:
            comp_config = self.config.components[component]
            if not comp_config.enabled:
                return DiagnosticLevel.OFF
            if comp_config.level is not None:
                return comp_config.level
        return self.config.level

    def is_enabled(self, component: str, min_level: DiagnosticLevel = DiagnosticLevel.SUMMARY) -> bool:
        """Check if diagnostics are enabled for a component at given level."""
        level = self.get_level(component)
        return level.value >= min_level.value

    def record(self, component: str, data: Dict[str, Any]) -> None:
        """Record diagnostic data for a component."""
        if not self.is_enabled(component):
            return

        if self.config.timestamp_entries:
            data["_timestamp"] = time.time() - self._start_time

        self._current[component] = data

        if self.config.collect_history and component in self._history:
            self._history[component].append(data.copy())

    def get_current(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get current diagnostic data."""
        if component:
            return self._current.get(component, {})
        return self._current.copy()

    def get_history(self, component: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get diagnostic history for a component."""
        if component not in self._history:
            return []
        hist = list(self._history[component])
        if n is not None:
            return hist[-n:]
        return hist

    def clear_current(self) -> None:
        """Clear current trial data (call at trial end)."""
        self._current.clear()

    def new_trial(self) -> None:
        """Start a new trial."""
        self._trial_count += 1
        self.clear_current()

    def new_epoch(self) -> None:
        """Start a new epoch."""
        self._epoch_count += 1
        self._trial_count = 0

    @property
    def trial_count(self) -> int:
        """Get current trial count."""
        return self._trial_count

    @property
    def epoch_count(self) -> int:
        """Get current epoch count."""
        return self._epoch_count

    # =========================================================================
    # FORMATTED OUTPUT
    # =========================================================================

    def format_trial_line(
        self,
        trial: int,
        gt: str,
        action: str,
        correct: bool,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a single trial result line."""
        rwd = "+" if correct else "-"
        line = f"  Trial {trial:3d}: GT={gt}, Act={action}, {rwd}"

        if extras and self.get_level().value >= DiagnosticLevel.DETAILED.value:
            extra_strs = []
            for k, v in extras.items():
                if isinstance(v, float):
                    extra_strs.append(f"{k}={v:.3f}")
                else:
                    extra_strs.append(f"{k}={v}")
            if extra_strs:
                line += f" ({', '.join(extra_strs)})"

        return line

    def print_trial(
        self,
        trial: int,
        gt: str,
        action: str,
        correct: bool,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print a trial result if enabled."""
        if not self.config.print_to_console:
            return
        if self.get_level().value < DiagnosticLevel.DETAILED.value:
            return
        print(self.format_trial_line(trial, gt, action, correct, extras))

    def format_epoch_summary(
        self,
        epoch: int,
        accuracy: float,
        match_acc: float,
        nomatch_acc: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format epoch summary line."""
        line = f"  Epoch {epoch:3d}: Acc={accuracy*100:.1f}% (M:{match_acc*100:.1f}%, NM:{nomatch_acc*100:.1f}%)"

        if extras:
            extra_strs = []
            for k, v in extras.items():
                if isinstance(v, float):
                    extra_strs.append(f"{k}={v:.2f}")
                elif isinstance(v, (list, tuple)):
                    extra_strs.append(f"{k}=[{', '.join(f'{x:.2f}' for x in v)}]")
                else:
                    extra_strs.append(f"{k}={v}")
            if extra_strs:
                line += f" [{', '.join(extra_strs)}]"

        return line

    def print_epoch(
        self,
        epoch: int,
        accuracy: float,
        match_acc: float,
        nomatch_acc: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print epoch summary if enabled."""
        if not self.config.print_to_console:
            return
        if self.get_level().value < DiagnosticLevel.SUMMARY.value:
            return
        print(self.format_epoch_summary(epoch, accuracy, match_acc, nomatch_acc, extras))

    def format_weights_summary(
        self,
        d1_per_action: List[float],
        d2_per_action: List[float],
        net_per_action: Optional[List[float]] = None,
        val_per_action: Optional[List[float]] = None,
    ) -> str:
        """Format D1/D2/NET weight summary."""
        if net_per_action is None:
            net_per_action = [d1 - d2 for d1, d2 in zip(d1_per_action, d2_per_action)]

        d1_str = ", ".join(f"{x:.4f}" for x in d1_per_action)
        d2_str = ", ".join(f"{x:.4f}" for x in d2_per_action)
        net_str = ", ".join(f"{x:.4f}" for x in net_per_action)

        line = f"D1: [{d1_str}] | D2: [{d2_str}] | NET: [{net_str}]"

        if val_per_action:
            val_str = ", ".join(f"{x:.3f}" for x in val_per_action)
            line += f" | VAL: [{val_str}]"

        return line

    def print_weights(
        self,
        d1_per_action: List[float],
        d2_per_action: List[float],
        net_per_action: Optional[List[float]] = None,
        val_per_action: Optional[List[float]] = None,
        prefix: str = "",
    ) -> None:
        """Print weight summary if enabled."""
        if not self.config.print_to_console:
            return
        if self.get_level().value < DiagnosticLevel.SUMMARY.value:
            return
        line = self.format_weights_summary(d1_per_action, d2_per_action, net_per_action, val_per_action)
        print(f"{prefix}{line}")


# =========================================================================
# COMPONENT DIAGNOSTIC INTERFACES
# =========================================================================

@dataclass
class StriatumDiagnostics:
    """Structured diagnostics for Striatum."""
    # Per-action weights
    d1_per_action: List[float] = field(default_factory=list)
    d2_per_action: List[float] = field(default_factory=list)
    net_per_action: List[float] = field(default_factory=list)

    # Value estimates (if RPE enabled)
    value_per_action: List[float] = field(default_factory=list)

    # Eligibility traces
    d1_elig_per_action: List[float] = field(default_factory=list)
    d2_elig_per_action: List[float] = field(default_factory=list)

    # Action selection
    last_action: Optional[int] = None
    exploring: bool = False
    exploration_prob: float = 0.0

    # UCB
    ucb_bonus: List[float] = field(default_factory=list)
    action_counts: List[int] = field(default_factory=list)
    total_trials: int = 0

    # Activity
    d1_spikes_per_action: List[float] = field(default_factory=list)
    d2_spikes_per_action: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "d1_per_action": self.d1_per_action,
            "d2_per_action": self.d2_per_action,
            "net_per_action": self.net_per_action,
            "value_per_action": self.value_per_action,
            "d1_elig_per_action": self.d1_elig_per_action,
            "d2_elig_per_action": self.d2_elig_per_action,
            "last_action": self.last_action,
            "exploring": self.exploring,
            "exploration_prob": self.exploration_prob,
            "ucb_bonus": self.ucb_bonus,
            "action_counts": self.action_counts,
            "total_trials": self.total_trials,
            "d1_spikes_per_action": self.d1_spikes_per_action,
            "d2_spikes_per_action": self.d2_spikes_per_action,
        }


@dataclass
class HippocampusDiagnostics:
    """Structured diagnostics for Hippocampus."""
    # CA1 activity (implicit comparison via NMDA coincidence detection)
    ca1_total_spikes: float = 0.0
    ca1_normalized: float = 0.0  # Normalized activity [0, 1]

    # NMDA gating
    nmda_gate_mean: float = 0.0
    nmda_gate_max: float = 0.0

    # Layer activity
    dg_spikes: float = 0.0
    ca3_spikes: float = 0.0
    ca1_spikes: float = 0.0

    # Memory metrics
    n_stored_episodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ca1_total_spikes": self.ca1_total_spikes,
            "ca1_normalized": self.ca1_normalized,
            "nmda_gate_mean": self.nmda_gate_mean,
            "nmda_gate_max": self.nmda_gate_max,
            "dg_spikes": self.dg_spikes,
            "ca3_spikes": self.ca3_spikes,
            "ca1_spikes": self.ca1_spikes,
            "n_stored_episodes": self.n_stored_episodes,
        }


@dataclass
class BrainSystemDiagnostics:
    """Aggregated diagnostics for entire brain system."""
    # Trial info
    trial_num: int = 0
    is_match: bool = False
    selected_action: int = 0
    correct: bool = False

    # Component diagnostics
    striatum: Optional[StriatumDiagnostics] = None
    hippocampus: Optional[HippocampusDiagnostics] = None

    # Timing
    trial_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_num": self.trial_num,
            "is_match": self.is_match,
            "selected_action": self.selected_action,
            "correct": self.correct,
            "striatum": self.striatum.to_dict() if self.striatum else None,
            "hippocampus": self.hippocampus.to_dict() if self.hippocampus else None,
            "trial_time_ms": self.trial_time_ms,
        }


# =========================================================================
# GLOBAL DIAGNOSTICS INSTANCE (optional convenience)
# =========================================================================

_global_diagnostics: Optional[DiagnosticsManager] = None


def get_diagnostics() -> DiagnosticsManager:
    """Get the global diagnostics manager (creates one if needed)."""
    global _global_diagnostics
    if _global_diagnostics is None:
        _global_diagnostics = DiagnosticsManager()
    return _global_diagnostics


def set_diagnostics(manager: DiagnosticsManager) -> None:
    """Set the global diagnostics manager."""
    global _global_diagnostics
    _global_diagnostics = manager


def configure_diagnostics(
    level: DiagnosticLevel = DiagnosticLevel.SUMMARY,
    print_to_console: bool = True,
    collect_history: bool = False,
) -> DiagnosticsManager:
    """Configure and return the global diagnostics manager."""
    config = DiagnosticsConfig(
        level=level,
        print_to_console=print_to_console,
        collect_history=collect_history,
    )
    manager = DiagnosticsManager(config=config)
    set_diagnostics(manager)
    return manager
