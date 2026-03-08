"""Sweep-mode diagnostics — run all input patterns from a shared warmup state.

Public API
----------
simulate
    Core simulation loop; runs *n_steps* timesteps with optional recording.
run_sweep
    Run a sequence of input patterns and return per-pattern reports.
plot_sweep_comparison
    Save a multi-panel bar chart comparing key metrics across patterns.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics_types import DiagnosticsReport
from .sensory_patterns import make_sensory_input

if TYPE_CHECKING:
    from thalia.brain import Brain
    from .diagnostics_recorder import DiagnosticsRecorder


DEFAULT_SWEEP_PATTERNS: tuple[str, ...] = ("random", "rhythmic", "burst", "slow_wave", "none")


# =============================================================================
# SIMULATION LOOP
# =============================================================================


def simulate(
    brain: "Brain",
    recorder: Optional["DiagnosticsRecorder"],
    n_steps: int,
    pattern: str,
    *,
    report_interval: int,
    label: str,
    spike_accumulator: Optional[Dict[str, int]] = None,
) -> float:
    """Run the core simulation loop for *n_steps* timesteps.

    If *recorder* is ``None`` the loop runs ``brain.forward()`` only (warmup
    mode — no data is recorded).  If a recorder is provided, each forward
    output is additionally passed to ``recorder.record()``.

    Parameters
    ----------
    brain:
        The live ``Brain`` instance.
    recorder:
        ``DiagnosticsRecorder`` to record into, or ``None`` for warmup.
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
        if recorder is not None:
            _tr0 = time.perf_counter()
            recorder.record(t, outputs)
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


def run_warmup(
    brain: "Brain",
    n_steps: int,
    pattern: str,
    *,
    report_interval: int = 100,
) -> None:
    """Run *n_steps* warmup timesteps without recording and print a status summary.

    Call this before :func:`run_sweep` or a recording loop to bring the network
    to a homeostatic steady state.  After the loop completes a
    ``[post-warmup]`` line is printed listing active and silent regions.

    Parameters
    ----------
    brain:
        The live ``Brain`` instance.
    n_steps:
        Number of warmup timesteps (must be > 0).
    pattern:
        Sensory input pattern key used during warmup.
    report_interval:
        Print a progress line every *report_interval* steps.
    """
    print(f"\n{'─'*80}")
    print(f"WARMUP  {n_steps} timesteps  (pattern={pattern!r}, not recorded)")
    print(f"{'─'*80}\n")
    _warmup_spikes: Dict[str, int] = {}
    elapsed = simulate(
        brain, None, n_steps, pattern,
        report_interval=report_interval, label="warmup",
        spike_accumulator=_warmup_spikes,
    )
    print(f"  \u2713 Warmup complete: {elapsed:.2f} s")
    _all_regions = list(brain.regions.keys())
    _silent = [rn for rn in _all_regions if _warmup_spikes.get(rn, 0) == 0]
    _n_active = len(_all_regions) - len(_silent)
    # Per-region spike-count summary for instant calibration feedback.
    # Sort active regions descending by count so the most-driven regions appear first.
    _sorted_regions = sorted(
        _all_regions,
        key=lambda rn: _warmup_spikes.get(rn, 0),
        reverse=True,
    )
    _col_width = max((len(rn) for rn in _all_regions), default=10)
    if _silent:
        print(
            f"\n  [post-warmup]  active={_n_active}  silent={len(_silent)}"
            f"\n  {'Region':<{_col_width}}  {'Spikes':>12}"
        )
        for _rn in _sorted_regions:
            _cnt = _warmup_spikes.get(_rn, 0)
            _note = "  \u2190 check connectivity before recording" if _cnt == 0 else ""
            print(f"  {_rn:<{_col_width}}  {_cnt:>12,}{_note}")
    else:
        print(
            f"\n  [post-warmup]  active={_n_active}  silent=0"
            f"\n  {'Region':<{_col_width}}  {'Spikes':>12}"
        )
        for _rn in _sorted_regions:
            _cnt = _warmup_spikes.get(_rn, 0)
            print(f"  {_rn:<{_col_width}}  {_cnt:>12,}")


def run_single(
    brain: "Brain",
    recorder: "DiagnosticsRecorder",
    pattern: str,
    timesteps: int,
    output_dir: str,
    *,
    report_interval: int,
    no_plots: bool = False,
    triage_fn: Optional[Callable[["Brain", DiagnosticsReport], None]] = None,
    detailed: bool = True,
) -> DiagnosticsReport:
    """Simulate one recording pass, then analyse, report, optionally triage, save, and plot.

    The caller is responsible for any warmup (:func:`run_warmup`) and for
    resetting the recorder between passes (``recorder.reset()``).
    """
    elapsed_sim = simulate(
        brain, recorder, timesteps, pattern,
        report_interval=report_interval, label="step",
    )
    print(
        f"\n  \u2713 Simulation complete: {elapsed_sim:.2f} s"
        f"  ({timesteps / elapsed_sim:.1f} steps/s)"
    )

    print(f"\n{'\u2550'*80}")
    print("ANALYSING")
    print(f"{'\u2550'*80}\n")
    recorder.config.sensory_pattern = pattern
    t_analyse = time.perf_counter()
    report = recorder.analyze()
    print(f"  \u2713 Analysis complete in {time.perf_counter() - t_analyse:.3f} s")

    recorder.print_report(report, detailed=detailed)

    if triage_fn is not None:
        triage_fn(brain, report)

    print(f"\n{'\u2550'*80}")
    print(f"SAVING  \u2192  {output_dir}")
    print(f"{'\u2550'*80}\n")
    recorder.save(report, output_dir)

    if not no_plots:
        print(f"\n{'\u2550'*80}")
        print("GENERATING PLOTS")
        print(f"{'\u2550'*80}\n")
        recorder.plot(report, output_dir)

    return report


def plot_sweep_comparison(
    sweep_reports: Dict[str, DiagnosticsReport],
    output_dir: str,
) -> None:
    """Save a multi-panel bar chart comparing key metrics across sweep patterns.

    Panels:

    1. Mean firing rate (Hz) per region+population, grouped by pattern.
    2. E/I ratio per region, grouped by pattern.
    3. Stability score per pattern.

    The figure is written to ``<output_dir>/sweep_comparison.png``.
    """
    patterns = list(sweep_reports.keys())
    n_pat = len(patterns)
    colours = matplotlib.colormaps["tab10"].resampled(n_pat)
    pat_colours = {p: colours(i) for i, p in enumerate(patterns)}

    # Collect all (region, population) keys across all reports.
    pop_keys: List[str] = sorted(
        {
            pk
            for rep in sweep_reports.values()
            for pk in rep.health.population_status
        }
    )
    region_keys: List[str] = sorted(
        {rn for rep in sweep_reports.values() for rn in rep.regions}
    )

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(pop_keys) * 0.7 + 2), 14))
    fig.suptitle("Sweep Pattern Comparison", fontsize=13)

    # Panel 1: mean FR per population.
    ax0 = axes[0]
    x0 = np.arange(len(pop_keys))
    bar_w = 0.8 / max(n_pat, 1)
    for i, pat in enumerate(patterns):
        rep = sweep_reports[pat]
        heights = []
        for pk in pop_keys:
            rn, pn = pk.split(":", 1)
            fr_val = 0.0
            rs = rep.regions.get(rn)
            if rs is not None:
                pop_s = rs.populations.get(pn)
                if pop_s is not None:
                    fr_val = pop_s.mean_fr_hz if not np.isnan(pop_s.mean_fr_hz) else 0.0
            heights.append(fr_val)
        ax0.bar(x0 + i * bar_w, heights, width=bar_w, label=pat, color=pat_colours[pat], alpha=0.85)
    ax0.set_xticks(x0 + bar_w * (n_pat - 1) / 2)
    ax0.set_xticklabels(pop_keys, rotation=45, ha="right", fontsize=7)
    ax0.set_ylabel("Mean FR (Hz)")
    ax0.set_title("Firing Rates by Population")
    ax0.legend(fontsize=8)

    # Panel 2: E/I ratio per region.
    ax1 = axes[1]
    x1 = np.arange(len(region_keys))
    for i, pat in enumerate(patterns):
        rep = sweep_reports[pat]
        heights = [
            (rep.regions[rn].ei_ratio if rn in rep.regions and not np.isnan(rep.regions[rn].ei_ratio) else 0.0)
            for rn in region_keys
        ]
        ax1.bar(x1 + i * bar_w, heights, width=bar_w, label=pat, color=pat_colours[pat], alpha=0.85)
    ax1.set_xticks(x1 + bar_w * (n_pat - 1) / 2)
    ax1.set_xticklabels(region_keys, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("E/I ratio")
    ax1.set_title("E/I Conductance Ratio by Region")
    ax1.legend(fontsize=8)

    # Panel 3: stability score per pattern.
    ax2 = axes[2]
    scores = [sweep_reports[p].health.stability_score for p in patterns]
    bar_colours = [pat_colours[p] for p in patterns]
    ax2.bar(patterns, scores, color=bar_colours, alpha=0.85)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Stability score")
    ax2.set_title("Overall Stability Score by Pattern")
    ax2.axhline(0.7, color="orange", linestyle="--", linewidth=0.8, label="OK threshold (0.7)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sweep_comparison.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Sweep comparison figure saved: {out_path}")


def run_sweep(
    brain: "Brain",
    recorder: "DiagnosticsRecorder",
    *,
    timesteps: int,
    output_dir: str,
    warmup: int = 0,
    warmup_pattern: str = "background",
    patterns: Sequence[str] = DEFAULT_SWEEP_PATTERNS,
    no_plots: bool = False,
    report_interval: int = 200,
    triage_fn: Optional[Callable[["Brain", DiagnosticsReport], None]] = None,
) -> Dict[str, DiagnosticsReport]:
    """Run *patterns* in sequence from a shared warmup state.

    Each pattern is run for *timesteps* steps.  A shared warmup of *warmup*
    steps is run first (if > 0) so all patterns start from the same
    STP/homeostatic state rather than the STP-naïve cold state.

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
    warmup:
        Number of warmup steps before the first pattern.
    warmup_pattern:
        Input pattern used during the warmup phase.
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

    # Shared warmup — all patterns start from the same homeostatic state.
    if warmup > 0:
        run_warmup(brain, warmup, warmup_pattern, report_interval=100)

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
