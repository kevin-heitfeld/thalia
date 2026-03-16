"""Comprehensive Brain Diagnostics — thin driver script."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO

import torch

from thalia import GlobalConfig
from thalia.brain import BrainBuilder
from thalia.diagnostics import (
    DiagnosticsConfig,
    DiagnosticsRecorder,
    SENSORY_PATTERNS,
    print_brain_config,
    print_neuron_populations,
    print_synaptic_weights,
    run_sweep,
    run_triage,
)
from thalia.diagnostics.sweep import run_single, DEFAULT_SWEEP_PATTERNS

from scripts.compare_runs import compare
from scripts.tee_writer import TeeWriter

# ── Configure line buffering so progress prints appear immediately ─────────────
sys.stdout.reconfigure(line_buffering=True)


def _run(args: argparse.Namespace, output_dir: str) -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("THALIA  ·  COMPREHENSIVE BRAIN DIAGNOSTICS")
    print("═" * 80)

    print("\n  GlobalConfig:")
    for field in GlobalConfig.__dataclass_fields__.values():
        value = getattr(GlobalConfig, field.name)
        print(f"    {field.name:25s}: {value}")

    print(f"\n  Run parameters:")
    print(f"    timesteps           : {args.timesteps}")
    print(f"    input-pattern       : {args.input_pattern}")
    print(f"    mode                : {args.mode}")
    print(f"    output-dir          : {output_dir}")
    print(f"    plots               : {'no' if args.no_plots else 'yes'}")
    print(f"    sweep               : {args.sweep}")
    if args.sweep_patterns:
        print(f"    sweep-patterns      : {args.sweep_patterns}")
    if args.report_interval is not None:
        print(f"    report-interval     : {args.report_interval}")
    print(f"    voltage-sample-size : {args.voltage_sample_size}")
    print(f"    conductance-sample  : {args.conductance_sample_size}")
    print(f"    cond-sample-interval: {args.conductance_sample_interval} timesteps")
    print(f"    rate-bin-ms         : {args.rate_bin_ms}")
    print(f"    device              : {args.device or '(pytorch default)'}")

    # ── Build brain ───────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("BUILDING BRAIN")
    print(f"{'═'*80}\n")
    # Set PyTorch default device before building so all tensors land there
    if args.device is not None:
        torch.set_default_device(args.device)
    t_build = time.perf_counter()
    brain = BrainBuilder.preset("default")
    print(f"  ✓ Brain built in {time.perf_counter() - t_build:.3f} s")

    print_brain_config(brain)
    print_neuron_populations(brain)
    print_synaptic_weights(brain, heading="INITIAL SYNAPTIC WEIGHTS")

    # ── Create recorder ───────────────────────────────────────────────────────
    # SFA health check is unreliable with a ramping input: FR rises monotonically
    # so all populations appear adapted regardless of cellular SFA properties.
    # Onset transients are handled automatically by detect_transient_step().
    recorder = DiagnosticsRecorder(
        brain=brain,
        config=DiagnosticsConfig(
            n_timesteps=args.timesteps,
            dt_ms=brain.dt_ms,
            mode=args.mode,
            voltage_sample_size=args.voltage_sample_size,
            conductance_sample_size=args.conductance_sample_size,
            conductance_sample_interval_steps=args.conductance_sample_interval,
            gain_sample_interval_ms=10,
            rate_bin_ms=args.rate_bin_ms,
            compute_avalanches=(args.timesteps >= 2000),
            skip_sfa_health_check=(args.input_pattern == "ramp"),
            sensory_pattern=args.input_pattern,
        ),
    )

    # Warn if voltage sample size is low relative to population sizes
    if recorder.config.voltage_sample_size < 20:
        large_pops = [
            f"{rn}:{pn}"
            for rn, region in brain.regions.items()
            for pn, pop in region.neuron_populations.items()
            if pop.n_neurons > 100
        ]
        if large_pops:
            print(
                f"\n  NOTE: voltage_sample_size={recorder.config.voltage_sample_size} < 20 while "
                f"{len(large_pops)} population(s) have >100 neurons. "
                f"Consider increasing --voltage-sample-size for richer subpopulation analysis. "
                f"voltage_bimodality (up/down state detection) is suppressed below 20 sampled "
                f"neurons — use --voltage-sample-size ≥20 to enable it."
            )

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if args.sweep:
        if args.sweep_patterns:
            _sweep_patterns = [p.strip() for p in args.sweep_patterns.split(",")]
            invalid = [p for p in _sweep_patterns if p not in SENSORY_PATTERNS]
            if invalid:
                raise SystemExit(
                    f"--sweep-patterns contains unknown pattern(s): {invalid}. "
                    f"Valid: {sorted(SENSORY_PATTERNS)}"
                )
        else:
            _sweep_patterns = DEFAULT_SWEEP_PATTERNS
        _sweep_report_interval = args.report_interval if args.report_interval is not None else 200
        run_sweep(
            brain, recorder,
            timesteps=args.timesteps,
            output_dir=output_dir,
            patterns=_sweep_patterns,
            no_plots=args.no_plots,
            report_interval=_sweep_report_interval,
            triage_fn=run_triage if args.triage else None,
        )
        print(f"\n{'═'*80}\nDONE (sweep)\n{'═'*80}\n")
        return

    # ── Single-run ────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(
        f"RUNNING {args.timesteps} TIMESTEPS"
        f"  input={args.input_pattern!r}  mode={args.mode!r}"
    )
    print(f"{'═'*80}\n")

    _single_report_interval = (
        args.report_interval if args.report_interval is not None
        else max(50, args.timesteps // 20)
    )
    run_single(
        brain, recorder, args.input_pattern, args.timesteps, output_dir,
        report_interval=_single_report_interval,
        no_plots=args.no_plots,
        triage_fn=run_triage if args.triage else None,
    )
    print_synaptic_weights(brain, heading="FINAL SYNAPTIC WEIGHTS")

    print(f"\n{'═'*80}")
    print("DONE")
    print(f"{'═'*80}\n")


def _run_comparison(base_dir: str, current_stamp: str) -> None:
    """Compare the current run with the most recent previous run, if one exists."""
    stamp_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}$")

    try:
        entries = sorted(
            (d for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d)) and stamp_re.match(d)),
            reverse=True,
        )
    except OSError:
        return

    if len(entries) < 2 or entries[0] != current_stamp:
        print("  No previous timestamped run found — skipping comparison.")
        return

    new_stamp, old_stamp = entries[0], entries[1]
    old_report = os.path.join(base_dir, old_stamp, "diagnostics_report.json")
    new_report = os.path.join(base_dir, new_stamp, "diagnostics_report.json")

    if not os.path.exists(old_report) or not os.path.exists(new_report):
        print("  Previous run missing diagnostics_report.json — skipping comparison.")
        return

    with open(old_report, encoding="utf-8") as f:
        old_data = json.load(f)
    with open(new_report, encoding="utf-8") as f:
        new_data = json.load(f)

    # Capture output so we can write it to file AND print to console/log
    buf = StringIO()
    with redirect_stdout(buf):
        compare(old_data, new_data)
    output = buf.getvalue()

    print(output, end="")

    comparison_dir = os.path.join(base_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_path = os.path.join(comparison_dir, f"{old_stamp}_vs_{new_stamp}.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"  \u2713 Comparison saved to {comparison_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive Thalia brain diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Device selection
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help=(
            "PyTorch device to build the brain on.  Defaults to the PyTorch global "
            "default (usually CPU).  Use 'cuda' for NVIDIA GPU or 'mps' for Apple "
            "Silicon.  The brain and all tensors are created on this device."
        ),
    )

    # Run configuration
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "stats"],
        help=(
            "'full' records spike times, voltages, and conductances. "
            "'stats' records only spike counts and gains (lighter, suitable "
            "for long training-loop checks)."
        ),
    )
    parser.add_argument(
        "--timesteps", type=int, default=1500,
        help="Number of simulation timesteps (1 step = dt_ms).",
    )
    parser.add_argument(
        "--input-pattern", type=str, default="random",
        choices=list(SENSORY_PATTERNS),
        help=(
            "Sensory input pattern fed to the brain.\n" +
            "\n".join(f"  {k!r}: {v.__doc__}" for k, v in SENSORY_PATTERNS.items())
        ).replace("%", "%%"),
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help=(
            "Run a sequence of input patterns back-to-back, each for --timesteps steps. "
            "Saves a report per pattern and prints a cross-pattern comparison table of "
            "region firing rates.  Default patterns: random, rhythmic, burst, slow_wave, none. "
            "Override with --sweep-patterns."
        ),
    )
    parser.add_argument(
        "--sweep-patterns", type=str, default=None,
        help=(
            "Comma-separated list of input patterns to use in sweep mode. "
            "Overrides the default pattern set. "
            f"Valid names: {', '.join(sorted(SENSORY_PATTERNS))}. "
            "Example: --sweep-patterns random,burst,slow_wave"
        ),
    )
    parser.add_argument(
        "--report-interval", type=int, default=None,
        help=(
            "Print a simulation progress line every N timesteps. "
            "Defaults to max(50, timesteps // 20) for single-run mode "
            "and 200 for sweep mode.  Set to a large value (e.g. 99999) "
            "for quiet / CI runs."
        ),
    )

    # Sampling / binning fidelity knobs
    parser.add_argument(
        "--voltage-sample-size", type=int, default=20,
        help=(
            "Number of neurons sampled per population for voltage recording (full mode). "
            "Increase for larger populations (>50 neurons) to get reliable per-neuron "
            "ISI and bimodality statistics.  Each extra neuron costs one float tensor "
            "row per timestep."
        ),
    )
    parser.add_argument(
        "--conductance-sample-size", type=int, default=8,
        help=(
            "Number of neurons sampled per population for conductance recording (full mode). "
            "Increase alongside --voltage-sample-size for E/I-ratio accuracy."
        ),
    )
    parser.add_argument(
        "--rate-bin-ms", type=float, default=10.0,
        help=(
            "Width of the rate-estimation bin in milliseconds. "
            "Smaller values improve temporal resolution but degrade spectral "
            "frequency resolution (freq_resolution_hz = 1000 / (n_bins * rate_bin_ms)). "
            "Default 10 ms gives 1 Hz resolution at 1000 timesteps."
        ),
    )
    parser.add_argument(
        "--conductance-sample-interval", type=int, default=1,
        help=(
            "How often (every N timesteps) to snapshot conductance values in full mode. "
            "1 samples every timestep (highest fidelity, highest memory). "
            "Increase to 5 or 10 for long runs (>5000 timesteps) with large brains "
            "to avoid excessive memory usage. The E/I ratio is computed as the mean "
            "over all samples taken."
        ),
    )

    # Automatic failure triage
    parser.add_argument(
        "--triage", action="store_true",
        help=(
            "After the main diagnostic run, automatically isolate every CRITICAL-SILENT "
            "region in a RegionTestRunner (single region, Poisson drive at 15 Hz) and "
            "print its per-population firing rates.  This makes it immediately clear "
            "whether silence is intrinsic to the region (still silent in isolation) or "
            "caused by missing upstream drive in the full-brain context.  Ignored in "
            "--sweep mode."
        ),
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="data/diagnostics",
        help="Directory for diagnostic outputs (JSON, NPZ, PNG).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation.",
    )
    parser.add_argument(
        "--no-comparison", action="store_true",
        help="Skip comparison with previous run.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # ── Create timestamped output directory ────────────────────────────────────
    run_stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_dir = os.path.join(args.output_dir, run_stamp)
    os.makedirs(output_dir, exist_ok=True)

    # ── Tee stdout/stderr to a log file ───────────────────────────────────────
    log_path = os.path.join(output_dir, "diagnostics.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        try:
            TeeWriter.patch_stdout_and_stderr(log_file)

            _run(args, output_dir)

            # ── Compare with previous run ─────────────────────────────────
            if not args.no_comparison:
                print(f"\n{'═'*80}")
                print("COMPARISON WITH PREVIOUS RUN")
                print(f"{'═'*80}\n")
                _run_comparison(args.output_dir, run_stamp)

        finally:
            TeeWriter.restore_original_stdout_and_stderr()
