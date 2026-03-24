"""Reusable brain simulation loop with diagnostics recording.

This module provides the core simulation loop that drives a :class:`~thalia.brain.Brain`
forward for a given number of timesteps while recording into a
:class:`~.diagnostics_recorder.DiagnosticsRecorder`.  It is used by the sweep/single-run
orchestrators in :mod:`.sweep` but can also be called directly for training, evaluation,
or any other context that needs to step a brain with sensory input and recording.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Optional

from .diagnostics_report import PerformanceMetrics
from .sensory_patterns import make_sensory_input

if TYPE_CHECKING:
    from thalia.brain import Brain
    from .diagnostics_recorder import DiagnosticsRecorder


def simulate(
    brain: Brain,
    recorder: DiagnosticsRecorder,
    n_steps: int,
    pattern: str,
    *,
    report_interval: int,
    label: str,
    spike_accumulator: Optional[Dict[str, int]] = None,
) -> PerformanceMetrics:
    """Run the core simulation loop for *n_steps* timesteps.

    Parameters
    ----------
    brain:
        The live ``Brain`` instance.
    recorder:
        ``DiagnosticsRecorder`` to record into.
    n_steps:
        Number of simulation steps to run.
    pattern:
        Sensory input pattern key (forwarded to :func:`~.sensory_patterns.make_sensory_input`).
    report_interval:
        Print a progress line every *report_interval* steps.
    label:
        Prefix used in the progress line (e.g. ``"step"`` or ``"warmup"``).
    spike_accumulator:
        Optional dict to accumulate total spike counts per region across the
        run.  If provided, ``spike_accumulator[region_name]`` is incremented by
        the number of spikes (across all populations in that region) each step.
        Useful for post-warmup activity status reporting.

    Returns
    -------
    PerformanceMetrics
        Timing breakdown for the simulation (wall-clock, forward, record).
    """
    t_start = time.perf_counter()

    t_forward_total = 0.0
    t_record_total = 0.0
    for t in range(n_steps):
        synaptic_inputs = make_sensory_input(brain, t, pattern, n_timesteps=n_steps)

        _tf0 = time.perf_counter()
        outputs = brain.forward(synaptic_inputs)
        _tf1 = time.perf_counter()
        t_forward_total += _tf1 - _tf0

        _tr0 = time.perf_counter()
        recorder.record(t, synaptic_inputs, outputs)
        t_record_total += time.perf_counter() - _tr0

        if spike_accumulator is not None:
            for rn, region_output in outputs.items():
                count = sum(int(spikes.sum().item()) for spikes in region_output.values())
                spike_accumulator[rn] = spike_accumulator.get(rn, 0) + count

        if (t + 1) % report_interval == 0:
            elapsed = time.perf_counter() - t_start
            steps_done = t + 1
            us_per_step = elapsed / steps_done * 1e6
            eta = (n_steps - steps_done) / steps_done * elapsed
            fwd_pct = t_forward_total / elapsed * 100.0
            rec_pct = t_record_total / elapsed * 100.0
            print(
                f"  {label} {steps_done:>6d}/{n_steps}"
                f"  {us_per_step:7.1f} µs/step"
                f"  fwd {fwd_pct:4.1f}%  rec {rec_pct:4.1f}%"
                f"  ETA {eta:6.2f} s"
            )

    elapsed = time.perf_counter() - t_start
    return PerformanceMetrics(
        wall_clock_s=elapsed,
        forward_s=t_forward_total,
        record_s=t_record_total,
        n_timesteps=n_steps,
    )
