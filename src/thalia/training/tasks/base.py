"""Base types and protocols for training tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from thalia.typing import SynapticInput


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """A single training trial presented to the brain.

    Attributes:
        pattern_id: Identifier for the stimulus pattern (task-defined).
        duration_steps: Total number of timesteps for this trial.
        response_window: ``(start_step, end_step)`` — the window during which
            readout spikes are counted for evaluation.
        metadata: Arbitrary task-specific data carried through to evaluation.
    """

    pattern_id: Any
    duration_steps: int
    response_window: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """Outcome of evaluating the brain's response to a trial.

    Attributes:
        reward: Scalar passed to ``brain.deliver_reward()``.
        correct: Whether the brain's response matched the expected output.
        metrics: Task-specific metrics (e.g. reaction time, margin).
    """

    reward: float
    correct: bool
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task protocol
# ---------------------------------------------------------------------------


class Task(Protocol):
    """Interface that every training task must implement."""

    name: str

    def generate_trial(self, trial_idx: int) -> Trial:
        """Create a new trial for the given index."""
        ...

    def make_input(self, trial: Trial, timestep: int) -> SynapticInput | None:
        """Return the external spike input for *timestep* within *trial*.

        Returns ``None`` for timesteps with no external drive (e.g. baseline
        or settling periods).
        """
        ...

    def evaluate(
        self,
        trial: Trial,
        spike_counts: dict[str, int],
    ) -> TrialResult:
        """Score the brain's response.

        Args:
            trial: The trial that was just run.
            spike_counts: Mapping from readout-group name to total spike count
                accumulated during the trial's response window.
        """
        ...

    def is_learned(self, recent_results: list[TrialResult]) -> bool:
        """Return True if the task convergence criterion is met."""
        ...


# ---------------------------------------------------------------------------
# Stimulus generator helper
# ---------------------------------------------------------------------------


def stimulus_iterator(
    task: Task,
    trial: Trial,
) -> Iterator[SynapticInput | None]:
    """Yield one ``SynapticInput | None`` per timestep for *trial*."""
    for t in range(trial.duration_steps):
        yield task.make_input(trial, t)
