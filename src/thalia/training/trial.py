"""Run a single training trial against a Brain instance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from thalia.brain.brain import Brain
from thalia.typing import BrainOutput

from thalia.training.encoding.spike_decoder import ReadoutGroup
from thalia.training.tasks.base import Task, Trial

if TYPE_CHECKING:
    from thalia.training.monitoring.health_monitor import HealthMonitor


def run_trial(
    brain: Brain,
    task: Task,
    trial: Trial,
    readout_groups: list[ReadoutGroup],
    monitor: HealthMonitor | None = None,
) -> dict[str, int]:
    """Step the brain through *trial* and return spike counts per readout group.

    Args:
        brain: Initialised brain instance.
        task: The task providing per-timestep inputs.
        trial: Trial specification (duration, response window, etc.).
        readout_groups: Readout channels to accumulate spikes for.
        monitor: Optional health monitor to record per-timestep spike counts.

    Returns:
        Mapping from readout-group name to total spike count within the
        trial's response window.
    """
    win_start, win_end = trial.response_window
    spike_counts: dict[str, int] = {g.name: 0 for g in readout_groups}

    for t in range(trial.duration_steps):
        synaptic_input = task.make_input(trial, t)
        outputs: BrainOutput = brain.forward(synaptic_input)

        if monitor is not None:
            monitor.record_step(outputs)

        # Accumulate spikes only during the response window.
        if win_start <= t < win_end:
            for group in readout_groups:
                spike_counts[group.name] += group.count(outputs)

    return spike_counts
