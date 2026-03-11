"""
Brain Performance Profiler
==========================
Measures time spent per region, per method, and highlights top bottlenecks.
Run with: python scripts/profile_brain.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from collections import defaultdict

import thalia.brain.regions.neural_region as _nr_mod
from thalia.brain import BrainBuilder
from thalia.diagnostics import SENSORY_PATTERNS

# ── Build brain ───────────────────────────────────────────────────────────────
print("Building brain...")
t0 = time.perf_counter()
brain = BrainBuilder.preset("default")
print(f"  Built in {time.perf_counter()-t0:.2f}s\n")

# ── Prepare a fixed sensory input (random pattern) ───────────────────────────
pattern_fn = SENSORY_PATTERNS["random"]

# ── PART 1: Per-region timing ─────────────────────────────────────────────────
print("=" * 70)
print("PART 1: Per-region timing (10 warmup + 20 measured steps)")
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
for i in range(10):
    brain.forward(pattern_fn(brain, i))

# Reset counters
region_times.clear()
region_calls.clear()

# Measure
N_STEPS = 20
t_total_start = time.perf_counter()
for i in range(N_STEPS):
    brain.forward(pattern_fn(brain, i))
t_total = time.perf_counter() - t_total_start

_nr_mod.NeuralRegion.forward = original_forward  # restore

print(f"\n  Total wall time: {t_total:.3f}s  ({N_STEPS/t_total:.2f} steps/s)")
print(f"  Time accounted in regions: {sum(region_times.values()):.3f}s\n")

# Sort by time descending
sorted_regions = sorted(region_times.items(), key=lambda x: x[1], reverse=True)
print(f"  {'Region':<35} {'Total(ms)':>10} {'Per-step(ms)':>14} {'% of total':>12}")
print(f"  {'-'*35} {'-'*10} {'-'*14} {'-'*12}")
total_region_time = sum(region_times.values())
for region, t in sorted_regions:
    pct = 100.0 * t / t_total
    per_step = 1000.0 * t / N_STEPS
    print(f"  {region:<35} {1000*t:>10.1f} {per_step:>14.2f} {pct:>11.1f}%")

# ── PART 2: cProfile of one timestep (post-warmup) ────────────────────────────
print("\n" + "=" * 70)
print("PART 2: cProfile of 5 timesteps (post-warmup) — top 40 functions")
print("=" * 70)

pr = cProfile.Profile()
pr.enable()
for i in range(5):
    brain.forward(pattern_fn(brain, i))
pr.disable()

stream = io.StringIO()
ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
ps.print_stats(40)
print(stream.getvalue())

# ── PART 3: Drill into the top region ─────────────────────────────────────────
top_region_name, top_region_time = sorted_regions[0]
print("=" * 70)
print(f"PART 3: Per-method breakdown for top region: '{top_region_name}'")
print("=" * 70)

top_region = brain.regions[top_region_name]
pr2 = cProfile.Profile()
pr2.enable()
for i in range(20):
    inp = pattern_fn(brain, i)
    synaptic_inputs = {k: v for k, v in inp.items() if k.target_region == top_region_name} if inp else {}
    top_region.forward(synaptic_inputs, {})
pr2.disable()

stream2 = io.StringIO()
ps2 = pstats.Stats(pr2, stream=stream2).sort_stats("cumulative")
ps2.print_stats(30)
print(stream2.getvalue())

# ── PART 4: Tensor operation timing in ConductanceLIF ─────────────────────────
print("=" * 70)
print("PART 4: Full cProfile — top 50 by tottime (pure self-time)")
print("=" * 70)

pr3 = cProfile.Profile()
pr3.enable()
for i in range(10):
    brain.forward(pattern_fn(brain, i))
pr3.disable()

stream3 = io.StringIO()
ps3 = pstats.Stats(pr3, stream=stream3).sort_stats("tottime")
ps3.print_stats(50)
print(stream3.getvalue())
