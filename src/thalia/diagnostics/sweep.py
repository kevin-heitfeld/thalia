"""Sweep over multiple sensory input patterns, running a full diagnostics pass for each, then compare."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .analysis import analyze
from .analysis_tuning import compute_tuning, print_tuning_report
from .calibration_advisor import compute_calibration_advice, print_calibration_advice
from .diagnostics_io import print_report, save_report, save_snapshot
from .diagnostics_plots import plot, plot_sweep_comparison
from .diagnostics_types import DiagnosticsReport
from .sensory_patterns import make_sensory_input

if TYPE_CHECKING:
    from thalia.brain import Brain
    from .diagnostics_recorder import DiagnosticsRecorder


DEFAULT_SWEEP_PATTERNS: Tuple[str, ...] = ("random", "rhythmic", "burst", "slow_wave", "none")


# =============================================================================
# SIMULATION LOOP
# =============================================================================


def _simulate(
    brain: Brain,
    recorder: DiagnosticsRecorder,
    n_steps: int,
    pattern: str,
    *,
    report_interval: int,
    label: str,
    spike_accumulator: Optional[Dict[str, int]] = None,
) -> float:
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
    float
        Elapsed wall-clock seconds for the loop.
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

    return time.perf_counter() - t_start


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================


def run_single(
    brain: Brain,
    recorder: DiagnosticsRecorder,
    pattern: str,
    timesteps: int,
    output_dir: str,
    *,
    report_interval: int,
    no_plots: bool = False,
    triage_fn: Optional[Callable[[Brain, DiagnosticsReport], None]] = None,
    detailed: bool = True,
) -> DiagnosticsReport:
    """Simulate one recording pass, then analyse, report, optionally triage, save, and plot.

    The caller is responsible for resetting the recorder between passes
    (``recorder.reset()``).
    """
    recorder.config.sensory_pattern = pattern
    elapsed_sim = _simulate(
        brain, recorder, timesteps, pattern,
        report_interval=report_interval, label="step",
    )
    print(
        f"\n  \u2713 Simulation complete: {elapsed_sim:.2f} s"
        f"  ({timesteps / elapsed_sim:.1f} steps/s)"
    )

    recorder_snapshot = recorder.to_snapshot()
    save_snapshot(recorder_snapshot, os.path.join(output_dir, "snapshot"))

    print(f"\n{'\u2550'*80}")
    print("ANALYSING")
    print(f"{'\u2550'*80}\n")
    t_analyse = time.perf_counter()
    report = analyze(recorder_snapshot)
    print(f"  \u2713 Analysis complete in {time.perf_counter() - t_analyse:.3f} s")

    print_report(report, detailed=detailed)

    # Tuning guidance for out-of-range populations
    tuning = compute_tuning(recorder_snapshot, report)
    print_tuning_report(tuning)

    # Calibration advisor: actionable parameter recommendations
    advice = compute_calibration_advice(recorder_snapshot, report, tuning)
    print_calibration_advice(advice)

    if triage_fn is not None:
        triage_fn(brain, report)

    print(f"\n{'\u2550'*80}")
    print(f"SAVING  \u2192  {output_dir}")
    print(f"{'\u2550'*80}\n")
    save_report(report, output_dir)
    print("  \u2713 Snapshot saved")

    if not no_plots:
        print(f"\n{'\u2550'*80}")
        print("GENERATING PLOTS")
        print(f"{'\u2550'*80}\n")
        plot(recorder_snapshot, report, output_dir)

    return report


def run_sweep(
    brain: Brain,
    recorder: DiagnosticsRecorder,
    *,
    timesteps: int,
    output_dir: str,
    patterns: Sequence[str] = DEFAULT_SWEEP_PATTERNS,
    no_plots: bool = False,
    report_interval: int = 200,
    triage_fn: Optional[Callable[[Brain, DiagnosticsReport], None]] = None,
) -> Dict[str, DiagnosticsReport]:
    """Run *patterns* in sequence and return per-pattern reports.

    Each pattern is run for *timesteps* steps.  The network starts from its
    current state (no shared warmup); onset transients are automatically
    detected and excluded by the analyzer using
    :func:`~.analysis.detect_transient_step`.

    Per-pattern reports are saved to ``<output_dir>/sweep_<pattern>/``.
    A cross-pattern JSON summary is written to ``<output_dir>/sweep_summary.json``.
    A comparison figure is saved to ``<output_dir>/sweep_comparison.png`` unless
    *no_plots* is ``True``.

    Parameters
    ----------
    brain:
        Live ``Brain`` instance.
    recorder:
        ``DiagnosticsRecorder`` configured for *brain*.  Its ``reset()``
        method is called between patterns.
    timesteps:
        Number of simulation steps per pattern.
    output_dir:
        Root directory for all sweep outputs.
    patterns:
        Ordered sequence of input-pattern keys to sweep over.
    no_plots:
        If ``True``, skip per-pattern and comparison plot generation.
    report_interval:
        Print a progress line every *report_interval* steps.
    triage_fn:
        Optional ``(brain, report) -> None`` called after each pattern's
        health report.  Pass the ``_run_triage`` function from
        ``comprehensive_diagnostics.py`` to isolate silent regions per pattern.

    Returns
    -------
    Dict[str, DiagnosticsReport]
        Map from pattern name to its :class:`DiagnosticsReport`.
    """
    sweep_reports: Dict[str, DiagnosticsReport] = {}

    for pat in patterns:
        print(f"\n{'═'*80}")
        print(f"SWEEP PATTERN: {pat!r}  ({timesteps} timesteps)")
        print(f"{'═'*80}\n")
        recorder.reset()
        pat_dir = os.path.join(output_dir, f"sweep_{pat}")
        report = run_single(
            brain, recorder, pat, timesteps, pat_dir,
            report_interval=report_interval,
            no_plots=no_plots,
            triage_fn=triage_fn,
            detailed=False,
        )
        sweep_reports[pat] = report

    # Cross-pattern comparison table.
    print(f"\n{'═'*80}")
    print("SWEEP COMPARISON TABLE  (mean firing rate Hz per region)")
    print(f"{'═'*80}")
    all_regions = sorted(
        {rn for rep in sweep_reports.values() for rn in rep.regions.keys()}
    )
    _COL_CAP = 8
    cols = all_regions[:_COL_CAP]
    n_hidden = len(all_regions) - len(cols)
    pat_w, fr_w = 12, 16
    header = f"{'pattern':<{pat_w}}" + "".join(f"{rn:<{fr_w}}" for rn in cols)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for pat, rep in sweep_reports.items():
        row = f"{pat:<{pat_w}}"
        for rn in cols:
            rs = rep.regions.get(rn)
            row += f"{rs.mean_fr_hz:<{fr_w}.2f}" if rs is not None else f"{'—':<{fr_w}}"
        print(f"  {row}")
    if n_hidden:
        print(f"  ({n_hidden} more region{'s' if n_hidden != 1 else ''} not shown: "
              f"{', '.join(all_regions[_COL_CAP:])})")

    # Save aggregated sweep summary JSON.
    summary: Dict[str, object] = {}
    for pat, rep in sweep_reports.items():
        pop_stats_out: Dict[str, object] = {}
        for rn, rs in rep.regions.items():
            for pn, ps in rs.populations.items():
                pop_stats_out[f"{rn}:{pn}"] = {
                    "mean_fr_hz":   None if np.isnan(ps.mean_fr_hz) else round(ps.mean_fr_hz, 4),
                    "total_spikes": ps.total_spikes,
                    "isi_cv":       None if np.isnan(ps.isi_cv) else round(ps.isi_cv, 4),
                    "bio_range_hz": list(ps.bio_range_hz) if ps.bio_range_hz else None,
                    "status":       ps.bio_plausibility,
                }
        summary[pat] = {
            "stability_score": round(rep.health.stability_score, 4),
            "n_critical":      len(rep.health.critical_issues),
            "n_warnings":      len(rep.health.warnings),
            "populations":     pop_stats_out,
        }
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  ✓ Sweep summary saved: {summary_path}")

    # Comparison figure.
    if not no_plots:
        plot_sweep_comparison(sweep_reports, output_dir)

    return sweep_reports
