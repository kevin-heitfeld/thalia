"""
Brain Performance Profiler
==========================
Measures time spent per region, per method, and highlights top bottlenecks.
Run with: python scripts/profile_brain.py
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import thalia.brain.regions.neural_region as _nr_mod
from thalia.brain import BrainBuilder
from thalia.diagnostics import SENSORY_PATTERNS

from scripts.tee_writer import TeeWriter


def _run_profile(n_steps_warmup: int, n_steps_profile: int, print_top_k: int):
    # ── Build brain ───────────────────────────────────────────────────────────────
    print("Building brain...")
    t0 = time.perf_counter()
    brain = BrainBuilder.preset("default")
    print(f"  Built in {time.perf_counter()-t0:.2f}s\n")

    # ── Prepare a fixed sensory input (random pattern) ───────────────────────────
    pattern_fn = SENSORY_PATTERNS["random"]

    # ── PART 1: Per-region timing ─────────────────────────────────────────────────
    print("=" * 70)
    print(f"PART 1: Per-region timing (forward method) — {n_steps_profile} steps")
    print("=" * 70)

    original_forward = _nr_mod.NeuralRegion.forward

    region_times: dict[str, float] = defaultdict(float)
    region_calls: dict[str, int] = defaultdict(int)

    def timed_forward(self, synaptic_inputs, neuromodulator_inputs):
        t = time.perf_counter()
        result = original_forward(self, synaptic_inputs, neuromodulator_inputs)
        region_times[self.region_name] += time.perf_counter() - t
        region_calls[self.region_name] += 1
        return result

    _nr_mod.NeuralRegion.forward = timed_forward

    # Warmup
    for t in range(n_steps_warmup):
        brain.forward(pattern_fn(brain, t))

    # Reset counters
    region_times.clear()
    region_calls.clear()

    # Measure
    t_total_start = time.perf_counter()
    for t in range(n_steps_profile):
        brain.forward(pattern_fn(brain, t))
    t_total = time.perf_counter() - t_total_start

    _nr_mod.NeuralRegion.forward = original_forward  # restore

    print(f"\n  Total wall time: {t_total:.3f}s  ({n_steps_profile/t_total:.2f} steps/s)")
    print(f"  Time accounted in regions: {sum(region_times.values()):.3f}s\n")

    # Sort by time descending
    sorted_regions = sorted(region_times.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Region':<35} {'Total(ms)':>10} {'Per-step(ms)':>14} {'% of total':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*14} {'-'*12}")
    for region, t in sorted_regions:
        pct = 100.0 * t / t_total
        per_step = 1000.0 * t / n_steps_profile
        print(f"  {region:<35} {1000*t:>10.1f} {per_step:>14.2f} {pct:>11.1f}%")

    # ── PART 2: cProfile of one timestep (post-warmup) ────────────────────────────
    print("\n" + "=" * 70)
    print(f"PART 2: cProfile of {n_steps_profile} timesteps (post-warmup) — top {print_top_k} functions")
    print("=" * 70)

    pr = cProfile.Profile()
    pr.enable()
    for t in range(n_steps_profile):
        brain.forward(pattern_fn(brain, t))
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(print_top_k)
    print(stream.getvalue())

    # ── PART 3: Drill into the top region ─────────────────────────────────────────
    top_region_name, _top_region_time = sorted_regions[0]
    print("=" * 70)
    print(f"PART 3: Per-method breakdown for top region: '{top_region_name}'")
    print("=" * 70)

    top_region = brain.regions[top_region_name]
    pr2 = cProfile.Profile()
    pr2.enable()
    for t in range(n_steps_profile):
        inp = pattern_fn(brain, t)
        synaptic_inputs = {k: v for k, v in inp.items() if k.target_region == top_region_name} if inp else {}
        # Provide precomputed STP efficacy, matching Brain.forward()
        stp_eff = brain._stp_batch.step({}, brain._last_brain_output)
        top_region._precomputed_stp_efficacy = stp_eff
        top_region.forward(synaptic_inputs, {})
        top_region._precomputed_stp_efficacy = None
    pr2.disable()

    stream2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=stream2).sort_stats("cumulative")
    ps2.print_stats(print_top_k)
    print(stream2.getvalue())

    # ── PART 4: Tensor operation timing in ConductanceLIF ─────────────────────────
    print("=" * 70)
    print(f"PART 4: Full cProfile — top {print_top_k} by tottime (pure self-time)")
    print("=" * 70)

    pr3 = cProfile.Profile()
    pr3.enable()
    for t in range(n_steps_profile):
        brain.forward(pattern_fn(brain, t))
    pr3.disable()

    stream3 = io.StringIO()
    ps3 = pstats.Stats(pr3, stream=stream3).sort_stats("tottime")
    ps3.print_stats(print_top_k)
    print(stream3.getvalue())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thalia brain profiling script. Measures time spent per region, per method, and highlights top bottlenecks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Profiling configuration
    parser.add_argument(
        "--n-steps-warmup", type=int, default=20,
        help="Number of warmup steps to run before profiling (to allow JIT compilation, caching, etc).",
    )
    parser.add_argument(
        "--n-steps-profile", type=int, default=100,
        help="Number of steps to run while profiling.",
    )
    parser.add_argument(
        "--print-top-k", type=int, default=10,
        help="Number of top entries to print in cProfile outputs.",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="data/profiling",
        help="Directory for profiling outputs.",
    )
    parser.add_argument(
        "--out-file-name-suffix", type=str, default=None,
        help="Optional suffix to add to the output file name (after timestamp).",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_file_name = f"{run_stamp}.txt" if args.out_file_name_suffix is None else f"{run_stamp}_{args.out_file_name_suffix}.txt"

    out_file = os.path.join(args.output_dir, out_file_name)
    with open(out_file, "w", encoding="utf-8") as log_file:
        try:
            TeeWriter.patch_stdout_and_stderr(log_file)

            _run_profile(
                n_steps_warmup=args.n_steps_warmup,
                n_steps_profile=args.n_steps_profile,
                print_top_k=args.print_top_k,
            )

        finally:
            TeeWriter.restore_original_stdout_and_stderr()
