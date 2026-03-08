"""
Comprehensive Brain Diagnostics — thin driver script.

This script owns the simulation loop.  All recording and analysis is
delegated to ``DiagnosticsRecorder``, which can equally be plugged into a
training loop with a single ``recorder.record(t, outputs)`` call.
"""

from __future__ import annotations

import argparse
import sys
import time

import matplotlib
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
from thalia.diagnostics.sweep import run_warmup, run_single, DEFAULT_SWEEP_PATTERNS

matplotlib.use("Agg")

# ── Configure line buffering so progress prints appear immediately ─────────────
sys.stdout.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive Thalia brain diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Number of simulation timesteps (1 step = dt_ms).",
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help=(
            "Number of warmup timesteps run before recording begins. "
            "Allows STP and adaptation to reach near-equilibrium so the "
            "measurement window reflects steady-state dynamics, not startup "
            "transients. At dt=1ms and 30 Hz thalamic input the thalamocortical "
            "STP effective time constant is ~67 ms; 300 ms ≈ 4.5τ. "
            "WARNING: circuits without STP-compensated weights (BLA, LHb, "
            "hippocampus) will be starved and appear silent with warmup>0."
        ),
    )
    parser.add_argument(
        "--warmup-pattern", type=str, default="background",
        choices=list(SENSORY_PATTERNS),
        help=(
            "Input pattern used during the warmup phase. "
            "Defaults to 'background' (low-rate Poisson broadcast to all regions) "
            "which is a neutral drive that settles STP without biasing any "
            "particular pathway.  In sweep mode this pattern is used for the "
            "shared pre-sweep warmup; in single-run mode it is used instead of "
            "--input-pattern so the warmup and recording phases can differ."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/diagnostics",
        help="Directory for diagnostic outputs (JSON, NPZ, PNG).",
    )
    _pattern_descriptions = {
        "random":               "sparse Poisson to thalamus relay (~3%% neurons, 20%% prob).",
        "rhythmic":             "8 Hz theta-burst to thalamus relay.",
        "burst":               "single 50 ms synchronous burst at t=100 ms (30%% of relay).",
        "sustained_burst":     "repeating 50 ms on / 450 ms off cycle to thalamus relay.",
        "background":          "low-rate (≈2 Hz) Poisson to every external-input synapse in all regions.",
        "none":                "no external input (spontaneous activity only).",
        "gamma":               "sinusoidal 40 Hz drive (5–15%% relay per step) — tests thalamocortical gamma chain.",
        "correlated_background": "half relay shares a common Poisson driver — tests common-input vs. local synchrony.",
        "ramp":                "linearly ramping relay activation 0→30%% over --timesteps — tests rate coding and neural gain.",
        "slow_wave":           "600 ms up/down cycle (up: 40 Hz relay, down: silence) — stress-tests cortical bistability and voltage bimodality.",
    }
    _pattern_help = "Sensory input pattern fed to the brain.\n" + "\n".join(
        f"  {k!r}: {v}" for k, v in _pattern_descriptions.items() if k in SENSORY_PATTERNS
    )
    parser.add_argument(
        "--input-pattern", type=str, default="random",
        choices=list(SENSORY_PATTERNS),
        help=_pattern_help,
    )
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
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation.",
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
        "--voltage-sample-size", type=int, default=8,
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

    # Avalanche analysis option (requires long runs for enough events)
    parser.add_argument(
        "--compute-avalanches", action="store_true",
        help=(
            "Fit a power-law to the spike avalanche size distribution (Beggs & Plenz 2003). "
            "Requires ≥200 avalanche events; add --timesteps ≥2000."
        ),
    )

    args = parser.parse_args()

    # Warmup validation — the entire recording window would be consumed.
    if args.warmup >= args.timesteps:
        parser.error(
            f"--warmup ({args.warmup}) must be strictly less than "
            f"--timesteps ({args.timesteps}).  "
            f"Increase --timesteps or reduce --warmup."
        )

    # Avalanche analysis validation — too few events for a reliable fit.
    if args.compute_avalanches and args.timesteps < 2000:
        parser.error(
            f"--compute-avalanches requires at least 200 avalanche events for a reliable fit. "
            f"With typical parameters this means --timesteps should be at least 2000."
        )

    return args


def main() -> None:
    args = parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("THALIA  ·  COMPREHENSIVE BRAIN DIAGNOSTICS")
    print("═" * 80)

    print("\n  GlobalConfig:")
    print(f"    DEFAULT_DT_MS            : {GlobalConfig.DEFAULT_DT_MS}")
    print(f"    HOMEOSTASIS_DISABLED     : {GlobalConfig.HOMEOSTASIS_DISABLED}")
    print(f"    LEARNING_DISABLED        : {GlobalConfig.LEARNING_DISABLED}")
    print(f"    NEUROMODULATION_DISABLED : {GlobalConfig.NEUROMODULATION_DISABLED}")
    print(f"    SYNAPTIC_WEIGHT_SCALE    : {GlobalConfig.SYNAPTIC_WEIGHT_SCALE}")

    print(f"\n  Run parameters:")
    print(f"    timesteps           : {args.timesteps}")
    print(f"    warmup              : {args.warmup}")
    print(f"    warmup-pattern      : {args.warmup_pattern}")
    print(f"    input-pattern       : {args.input_pattern}")
    print(f"    mode                : {args.mode}")
    print(f"    output-dir          : {args.output_dir}")
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
    # 3.4.2: set PyTorch default device before building so all tensors land there
    if args.device is not None:
        torch.set_default_device(args.device)
    t_build = time.perf_counter()
    brain = BrainBuilder.preset("default")
    print(f"  ✓ Brain built in {time.perf_counter() - t_build:.3f} s")

    print_brain_config(brain)
    print_neuron_populations(brain)
    print_synaptic_weights(brain, heading="INITIAL SYNAPTIC WEIGHTS")

    # ── Create recorder ───────────────────────────────────────────────────────
    # SFA health check is unreliable when using a ramping input (FR rises
    # monotonically regardless of cellular adaptation) or when warmup==0 with
    # a short recording (transient onset dynamics dominate the early window).
    _skip_sfa = (args.input_pattern == "ramp") or (
        args.warmup == 0 and args.timesteps < 2000
    )
    config = DiagnosticsConfig(
        n_timesteps=args.timesteps,
        dt_ms=brain.dt_ms,
        mode=args.mode,
        voltage_sample_size=args.voltage_sample_size,
        conductance_sample_size=args.conductance_sample_size,
        conductance_sample_interval_steps=args.conductance_sample_interval,
        gain_sample_interval_ms=10,
        rate_bin_ms=args.rate_bin_ms,
        compute_avalanches=args.compute_avalanches,
        skip_sfa_health_check=_skip_sfa,
        sensory_pattern=args.input_pattern,
    )
    recorder = DiagnosticsRecorder(brain, config)

    if _skip_sfa and args.input_pattern != "ramp":
        print(
            f"\n  NOTE: SFA health checks suppressed (warmup={args.warmup}, "
            f"timesteps={args.timesteps} < 2000).  "
            "SFA results reflect network onset transients, not cellular adaptation.  "
            "Re-run with --warmup ≥300 or --timesteps ≥2000 for reliable SFA."
        )

    # Warn if voltage sample size is low relative to population sizes
    if config.voltage_sample_size < 20:
        large_pops = [
            f"{rn}:{pn}"
            for rn, region in brain.regions.items()
            for pn, pop in region.neuron_populations.items()
            if pop.n_neurons > 100
        ]
        if large_pops:
            print(
                f"\n  NOTE: voltage_sample_size={config.voltage_sample_size} < 20 while "
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
            output_dir=args.output_dir,
            warmup=args.warmup,
            warmup_pattern=args.warmup_pattern,
            patterns=_sweep_patterns,
            no_plots=args.no_plots,
            report_interval=_sweep_report_interval,
            triage_fn=run_triage if args.triage else None,
        )
        print(f"\n{'═'*80}\nDONE (sweep)\n{'═'*80}\n")
        return

    # ── Warmup (single-pattern mode only) ────────────────────────────────────
    if args.warmup > 0:
        run_warmup(brain, args.warmup, args.warmup_pattern, report_interval=100)

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
        brain, recorder, args.input_pattern, args.timesteps, args.output_dir,
        report_interval=_single_report_interval,
        no_plots=args.no_plots,
        triage_fn=run_triage if args.triage else None,
    )
    print_synaptic_weights(brain, heading="FINAL SYNAPTIC WEIGHTS")

    print(f"\n{'═'*80}")
    print("DONE")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()
