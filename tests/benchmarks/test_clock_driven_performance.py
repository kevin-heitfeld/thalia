"""
Benchmark suite for clock-driven execution performance.

Measures the impact of Phase 1 optimizations:
1. Pre-computed connection topology
2. Reusable component_inputs dict
3. Pre-allocated output cache
4. GPU spike tensor tracking (no sync in hot loop)

Run with: pytest tests/benchmarks/test_clock_driven_performance.py -v
"""

import time

import pytest
import torch

from thalia.config import GlobalConfig
from thalia.core.brain_builder import BrainBuilder


@pytest.fixture
def global_config():
    """Standard config for benchmarks."""
    return GlobalConfig(
        device="cpu",  # Use CPU for consistent benchmarking
        dt_ms=1.0,
        theta_frequency_hz=8.0,
        alpha_frequency_hz=10.0,
        gamma_frequency_hz=40.0,
    )


class TestClockDrivenPerformance:
    """Benchmark clock-driven execution performance."""

    @pytest.mark.slow
    def test_benchmark_default_brain_1000_steps(self, global_config):
        """Benchmark 1000 timesteps of default brain execution."""
        brain = BrainBuilder.preset("default", global_config)

        # Warm-up run (JIT compilation, cache warming)
        sensory_input = torch.rand(128, device=global_config.device) > 0.5
        brain.forward(sensory_input, n_timesteps=10)
        brain.reset_state()

        # Timed run
        start = time.perf_counter()
        for _ in range(10):
            sensory_input = torch.rand(128, device=global_config.device) > 0.5
            brain.forward(sensory_input, n_timesteps=100)
        elapsed = time.perf_counter() - start

        timesteps_per_sec = 1000 / elapsed
        ms_per_timestep = (elapsed / 1000) * 1000

        print(f"\n{'='*60}")
        print(f"Default Brain (1000 timesteps)")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Throughput: {timesteps_per_sec:.1f} timesteps/sec")
        print(f"Latency: {ms_per_timestep:.3f} ms/timestep")
        print(f"{'='*60}\n")

        # Sanity check: should be at least 100 timesteps/sec on CPU
        assert timesteps_per_sec > 100, f"Too slow: {timesteps_per_sec:.1f} timesteps/sec"

    @pytest.mark.slow
    def test_benchmark_minimal_brain_10000_steps(self, global_config):
        """Benchmark 10000 timesteps of minimal brain (fewer components)."""
        brain = BrainBuilder.preset("minimal", global_config)

        # Warm-up
        sensory_spikes = torch.rand(64, device=global_config.device) > 0.5
        brain.forward({"input": sensory_spikes}, n_timesteps=10)
        brain.reset_state()

        # Timed run
        start = time.perf_counter()
        for _ in range(10):
            sensory_spikes = torch.rand(64, device=global_config.device) > 0.5
            brain.forward({"input": sensory_spikes}, n_timesteps=1000)
        elapsed = time.perf_counter() - start

        timesteps_per_sec = 10000 / elapsed
        ms_per_timestep = (elapsed / 10000) * 1000

        print(f"\n{'='*60}")
        print("Minimal Brain (10000 timesteps)")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Throughput: {timesteps_per_sec:.1f} timesteps/sec")
        print(f"Latency: {ms_per_timestep:.3f} ms/timestep")
        print(f"{'='*60}\n")

        # Minimal brain should be faster
        assert timesteps_per_sec > 200, f"Too slow: {timesteps_per_sec:.1f} timesteps/sec"


if __name__ == "__main__":
    import sys

    config = GlobalConfig(device="cpu", dt_ms=1.0)
    suite = TestClockDrivenPerformance()

    print("\n" + "=" * 60)
    print("CLOCK-DRIVEN EXECUTION BENCHMARKS")
    print("=" * 60 + "\n")

    try:
        suite.test_benchmark_default_brain_1000_steps(config)
        suite.test_benchmark_minimal_brain_10000_steps(config)

        print("\n" + "=" * 60)
        print("ALL BENCHMARKS PASSED")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ Benchmark failed: {e}\n")
        sys.exit(1)
