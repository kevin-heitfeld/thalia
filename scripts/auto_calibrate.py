"""
Auto-calibrate — automated parameter sweeps for isolated region tuning.

Wraps RegionTestRunner to systematically sweep neuron parameters and/or
config parameters, measuring firing rates and scoring against targets.
Each parameter combination runs an isolated region simulation (~3 s),
so a 50-point sweep takes ~2.5 min instead of the ~60 s full-brain diagnostic.

Supports two kinds of swept parameters:

  Neuron parameters  (format: ``pop.attr=min:max:n``)
      Override a registered buffer on the neuron module **after** region
      creation.  Examples: ``tan.v_threshold=0.8:2.0:12``,
      ``pv.noise_std=0.002:0.01:5``.

  Config parameters  (format: ``attr=min:max:n``)
      Set an attribute on the region config **before** region creation.
      Examples: ``tan_baseline_drive=0.002:0.01:6``, ``tonic_drive=0.001:0.01:5``.

Usage examples::

    # Single-parameter sweep: find optimal TAN v_threshold
    python scripts/auto_calibrate.py striatum \\
        --sweep tan.v_threshold=0.8:2.0:12 --target tan=7.5

    # Two-parameter sweep (grid)
    python scripts/auto_calibrate.py striatum \\
        --sweep tan.v_threshold=0.8:2.0:8 \\
        --sweep tan.noise_std=0.002:0.015:6 \\
        --target tan=7.5

    # Config-level sweep
    python scripts/auto_calibrate.py striatum \\
        --sweep tan_baseline_drive=0.002:0.010:8 --target tan=7.5

    # Multiple target populations
    python scripts/auto_calibrate.py subiculum \\
        --sweep principal.v_threshold=0.8:1.5:8 \\
        --target principal=3.0 --target pv=15.0

    # List available populations in a region
    python scripts/auto_calibrate.py striatum --list-populations
"""

from __future__ import annotations

import argparse
import copy
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from thalia.diagnostics.region_test_runner import RegionTestRunner
from scripts.tee_writer import TeeWriter

# Realistic firing rates for source populations that would otherwise default
# to 5 Hz in the Poisson input generator.
RATE_OVERRIDES: Dict[str, float] = {
    "globus_pallidus_interna:principal": 75.0,
    "globus_pallidus_externa:prototypic": 75.0,
    "medial_septum:gaba": 8.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SweepParameter:
    """One axis of the parameter grid."""

    name: str                # Display name (e.g. "tan.v_threshold" or "tan_baseline_drive")
    values: np.ndarray       # Values to evaluate
    is_neuron_param: bool    # True → neuron buffer override; False → config attribute
    population: str          # Population key (only for neuron params)
    attribute: str           # Attribute name on the neuron module or config

    @classmethod
    def parse(cls, spec: str) -> SweepParameter:
        """Parse ``pop.attr=min:max:n`` or ``config_attr=min:max:n``.

        Raises:
            ValueError: If the format is invalid.
        """
        if "=" not in spec:
            raise ValueError(
                f"Invalid sweep spec '{spec}'. Expected format: "
                "pop.attr=min:max:n  or  config_attr=min:max:n"
            )
        name, range_part = spec.split("=", 1)
        parts = range_part.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid range '{range_part}' in sweep spec '{spec}'. "
                "Expected min:max:n (e.g. 0.8:2.0:12)"
            )
        vmin, vmax, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n < 1:
            raise ValueError(f"Number of sweep points must be ≥ 1, got {n}")
        values = np.linspace(vmin, vmax, n)

        if "." in name:
            pop, attr = name.split(".", 1)
            return cls(name=name, values=values, is_neuron_param=True,
                       population=pop, attribute=attr)
        return cls(name=name, values=values, is_neuron_param=False,
                   population="", attribute=name)


@dataclass
class Target:
    """Desired firing rate for a population."""

    population: str
    rate_hz: float

    @classmethod
    def parse(cls, spec: str) -> Target:
        """Parse ``pop=rate``."""
        if "=" not in spec:
            raise ValueError(f"Invalid target spec '{spec}'. Expected pop=rate_hz")
        pop, rate = spec.split("=", 1)
        return cls(population=pop, rate_hz=float(rate))


@dataclass
class SweepResult:
    """Result of one parameter combination."""

    combo: tuple[float, ...]
    rates: Dict[str, float]
    score: float


# ─────────────────────────────────────────────────────────────────────────────
# Override helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_neuron_override(overrides: Sequence[tuple[SweepParameter, float]]) -> Any:
    """Return a region-override callback that sets neuron buffer values."""
    captured = list(overrides)  # snapshot

    def _apply(region: Any) -> None:
        for param, value in captured:
            neuron = region.get_neuron_population(param.population)
            if neuron is None:
                raise ValueError(
                    f"Population '{param.population}' not found in region. "
                    f"Available: {list(region.neuron_populations.keys())}"
                )
            buf = getattr(neuron, param.attribute, None)
            if buf is None or not isinstance(buf, torch.Tensor):
                raise ValueError(
                    f"'{param.attribute}' is not a tensor buffer on "
                    f"population '{param.population}'"
                )
            buf.fill_(value)

    return _apply


# ─────────────────────────────────────────────────────────────────────────────
# Sweep engine
# ─────────────────────────────────────────────────────────────────────────────


def run_sweep(
    region_name: str,
    params: List[SweepParameter],
    targets: List[Target],
    duration_ms: float = 3000.0,
    warmup_ms: float = 2000.0,
) -> List[SweepResult]:
    """Execute a grid sweep and return scored results.

    A template :class:`RegionTestRunner` is built once from the default preset,
    then deep-copied for each parameter combination so that config mutations
    cannot leak between runs.
    """
    # Build the template runner once (avoids re-running preset_builder per combo).
    template = RegionTestRunner.from_preset("default", region_name)
    template.add_preset_inputs("default", rate_overrides=RATE_OVERRIDES)

    combos = list(itertools.product(*(p.values for p in params)))

    print(f"\n{'═' * 72}")
    print(f"  AUTO-CALIBRATE: {region_name}")
    print(f"{'═' * 72}")
    print(f"  Parameters : {', '.join(p.name for p in params)}")
    print(f"  Grid points: {len(combos)}")
    print(f"  Targets    : {', '.join(f'{t.population}={t.rate_hz} Hz' for t in targets) or '(none)'}")
    print(f"  Simulation : {duration_ms:.0f} ms measurement + {warmup_ms:.0f} ms warmup")
    print()

    results: List[SweepResult] = []
    t_start = time.perf_counter()

    for i, combo in enumerate(combos):
        runner = copy.deepcopy(template)

        # Config-level overrides (applied before region creation)
        for param, value in zip(params, combo):
            if not param.is_neuron_param:
                setattr(runner.config, param.attribute, float(value))

        # Neuron-level overrides (applied after region creation via hook)
        neuron_overrides = [
            (param, float(value))
            for param, value in zip(params, combo)
            if param.is_neuron_param
        ]
        if neuron_overrides:
            runner.add_region_override(_make_neuron_override(neuron_overrides))

        result = runner.run(duration_ms=duration_ms, warmup_ms=warmup_ms)

        # Score: sum of normalised absolute errors across all targets.
        score = 0.0
        for t in targets:
            actual = result.rates_hz.get(t.population, 0.0)
            score += abs(actual - t.rate_hz) / max(t.rate_hz, 0.1)

        entry = SweepResult(combo=combo, rates=result.rates_hz, score=score)
        results.append(entry)

        # Progress line
        pct = (i + 1) / len(combos) * 100
        param_str = "  ".join(
            f"{p.name}={v:.4f}" for p, v in zip(params, combo)
        )
        rate_str = "  ".join(
            f"{t.population}={entry.rates.get(t.population, 0.0):.1f}Hz"
            for t in targets
        ) if targets else "  ".join(
            f"{k}={v:.1f}Hz" for k, v in sorted(entry.rates.items())
        )
        marker = ""
        if targets and entry.score < 0.1:
            marker = "  ✓"
        print(
            f"  [{i + 1:>4d}/{len(combos)}] ({pct:5.1f}%)  "
            f"{param_str}  →  {rate_str}  score={entry.score:.3f}{marker}"
        )

    elapsed = time.perf_counter() - t_start
    print(f"\n  Completed {len(combos)} runs in {elapsed:.1f}s "
          f"({elapsed / max(len(combos), 1):.1f}s/run)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_top_results(
    params: List[SweepParameter],
    targets: List[Target],
    results: List[SweepResult],
    n: int = 10,
) -> None:
    """Print the top-N parameter combinations ranked by score."""
    if not targets:
        return
    ranked = sorted(results, key=lambda r: r.score)

    print(f"\n{'═' * 72}")
    print(f"  TOP {min(n, len(ranked))} COMBINATIONS (lowest score = best)")
    print(f"{'═' * 72}")

    for rank, entry in enumerate(ranked[:n], start=1):
        param_str = "  ".join(
            f"{p.name}={v:.4f}" for p, v in zip(params, entry.combo)
        )
        rate_str = "  ".join(
            f"{t.population}={entry.rates.get(t.population, 0.0):.1f}Hz"
            for t in targets
        )
        print(f"  #{rank:<3d} {param_str}  →  {rate_str}  score={entry.score:.4f}")

    best = ranked[0]
    print(f"\n  ★ OPTIMAL (score={best.score:.4f}):")
    for p, v in zip(params, best.combo):
        print(f"      {p.name} = {v}")
    for t in targets:
        actual = best.rates.get(t.population, 0.0)
        delta = actual - t.rate_hz
        print(f"      {t.population}: {actual:.1f} Hz  (target {t.rate_hz:.1f} Hz, Δ={delta:+.1f})")


def print_sensitivity(
    params: List[SweepParameter],
    targets: List[Target],
    results: List[SweepResult],
) -> None:
    """Print marginal sensitivity: mean firing-rate vs each swept parameter."""
    if not targets or not params:
        return

    print(f"\n{'═' * 72}")
    print("  SENSITIVITY ANALYSIS  (∂rate / ∂param, averaged over other axes)")
    print(f"{'═' * 72}")

    for target in targets:
        for pidx, param in enumerate(params):
            # Group by this param's value, averaging over all other axes.
            buckets: Dict[float, List[float]] = {}
            for entry in results:
                v = entry.combo[pidx]
                rate = entry.rates.get(target.population, 0.0)
                buckets.setdefault(v, []).append(rate)

            sorted_vals = sorted(buckets.keys())
            mean_rates = [float(np.mean(buckets[v])) for v in sorted_vals]

            if len(sorted_vals) < 2:
                continue

            # Overall sensitivity (slope over full range)
            dp = sorted_vals[-1] - sorted_vals[0]
            dr = mean_rates[-1] - mean_rates[0]
            sens = dr / dp if dp != 0 else 0.0

            print(f"\n  {target.population} firing rate  vs  {param.name}")
            print(f"    Sensitivity ≈ {sens:+.2f} Hz per unit change in {param.name}")
            print(f"    Range: {sorted_vals[0]:.4f} → {sorted_vals[-1]:.4f}")
            print(f"    Effect: {mean_rates[0]:.1f} Hz → {mean_rates[-1]:.1f} Hz")
            print()

            # ASCII bar chart
            max_rate = max(max(mean_rates), 0.1)
            bar_width = 40
            for v, r in zip(sorted_vals, mean_rates):
                bar_len = int(r / max_rate * bar_width)
                bar = "█" * bar_len
                at_target = (
                    " ◄ TARGET"
                    if abs(r - target.rate_hz) <= max(target.rate_hz * 0.15, 0.5)
                    else ""
                )
                print(f"    {v:>10.4f} │ {r:>6.1f} Hz │{bar}{at_target}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated region calibration via parameter sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Sweep TAN v_threshold (12 points from 0.8 to 2.0), target 7.5 Hz
  python scripts/auto_calibrate.py striatum \\
      --sweep tan.v_threshold=0.8:2.0:12 --target tan=7.5

  # Two-parameter grid sweep
  python scripts/auto_calibrate.py striatum \\
      --sweep tan.v_threshold=0.8:2.0:8 \\
      --sweep tan.noise_std=0.002:0.015:6 \\
      --target tan=7.5

  # Config-level parameter
  python scripts/auto_calibrate.py striatum \\
      --sweep tan_baseline_drive=0.002:0.010:8 --target tan=7.5

  # List available populations
  python scripts/auto_calibrate.py striatum --list-populations
""",
    )
    parser.add_argument("region", help="Region name (e.g. striatum, subiculum)")
    parser.add_argument(
        "--sweep", action="append", default=[],
        help="Parameter sweep: pop.attr=min:max:n  or  config_attr=min:max:n  (repeatable)",
    )
    parser.add_argument(
        "--target", action="append", default=[],
        help="Target firing rate: pop=rate_hz  (repeatable)",
    )
    parser.add_argument(
        "--duration", type=float, default=3000.0,
        help="Measurement duration in ms (default: 3000)",
    )
    parser.add_argument(
        "--warmup", type=float, default=2000.0,
        help="Warmup duration in ms (default: 2000)",
    )
    parser.add_argument(
        "--list-populations", action="store_true",
        help="List populations in the region and exit",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/calibration_runs",
        help="Directory to save sweep results (default: data/calibration_runs)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H%M%S")
    output_path = output_dir / f"{stamp}_sweep_{args.region}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        try:
            TeeWriter.patch_stdout_and_stderr(f)

            # --list-populations mode
            if args.list_populations:
                runner = RegionTestRunner.from_preset("default", args.region)
                print(f"\nPopulations in '{args.region}':")
                for pop, size in sorted(runner.population_sizes.items()):
                    print(f"  {pop}: {size} neurons")
                return

            if not args.sweep:
                parser.error("at least one --sweep is required")

            params = [SweepParameter.parse(s) for s in args.sweep]
            targets = [Target.parse(t) for t in args.target]

            total_combos = 1
            for p in params:
                total_combos *= len(p.values)

            results = run_sweep(
                args.region, params, targets,
                duration_ms=args.duration, warmup_ms=args.warmup,
            )

            print_top_results(params, targets, results)
            print_sensitivity(params, targets, results)

            print(f"Auto-Calibrate Sweep: {args.region}")
            print(f"Parameters: {', '.join(p.name for p in params)}")
            print(f"Targets: {', '.join(f'{t.population}={t.rate_hz}Hz' for t in targets)}\n\n")

            ranked = sorted(results, key=lambda r: r.score) if targets else results
            for entry in ranked:
                param_str = "  ".join(
                    f"{p.name}={v:.4f}" for p, v in zip(params, entry.combo)
                )
                rate_str = "  ".join(
                    f"{k}={v:.1f}Hz" for k, v in sorted(entry.rates.items())
                )
                print(f"{param_str}  →  {rate_str}  score={entry.score:.4f}")

        finally:
            TeeWriter.restore_original_stdout_and_stderr()


if __name__ == "__main__":
    main()
