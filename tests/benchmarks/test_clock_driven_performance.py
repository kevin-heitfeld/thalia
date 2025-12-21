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
import torch
import pytest
from thalia.core.brain_builder import BrainBuilder
from thalia.config import GlobalConfig


@pytest.fixture
def global_config():
    """Standard config for benchmarks."""
    return GlobalConfig(
        device="cpu",  # Use CPU for consistent benchmarking
        dt_ms=1.0,
        theta_freq=8.0,
        alpha_freq=10.0,
        gamma_freq=40.0,
    )


class TestClockDrivenPerformance:
    """Benchmark clock-driven execution performance."""

    def test_benchmark_sensorimotor_brain_1000_steps(self, global_config):
        """Benchmark 1000 timesteps of sensorimotor brain execution."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

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
        print(f"Sensorimotor Brain (1000 timesteps)")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Throughput: {timesteps_per_sec:.1f} timesteps/sec")
        print(f"Latency: {ms_per_timestep:.3f} ms/timestep")
        print(f"{'='*60}\n")

        # Sanity check: should be at least 100 timesteps/sec on CPU
        assert timesteps_per_sec > 100, f"Too slow: {timesteps_per_sec:.1f} timesteps/sec"

    def test_benchmark_minimal_brain_10000_steps(self, global_config):
        """Benchmark 10000 timesteps of minimal brain (fewer components)."""
        brain = BrainBuilder.preset("minimal", global_config)

        # Warm-up
        sensory_input = torch.rand(128, device=global_config.device) > 0.5
        brain.forward(sensory_input, n_timesteps=10)
        brain.reset_state()

        # Timed run
        start = time.perf_counter()
        for _ in range(10):
            sensory_input = torch.rand(128, device=global_config.device) > 0.5
            brain.forward(sensory_input, n_timesteps=1000)
        elapsed = time.perf_counter() - start

        timesteps_per_sec = 10000 / elapsed
        ms_per_timestep = (elapsed / 10000) * 1000

        print(f"\n{'='*60}")
        print(f"Minimal Brain (10000 timesteps)")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.3f}s")
        print(f"Throughput: {timesteps_per_sec:.1f} timesteps/sec")
        print(f"Latency: {ms_per_timestep:.3f} ms/timestep")
        print(f"{'='*60}\n")

        # Minimal brain should be faster
        assert timesteps_per_sec > 200, f"Too slow: {timesteps_per_sec:.1f} timesteps/sec"

    def test_benchmark_connection_lookup_overhead(self, global_config):
        """Measure connection lookup overhead (pre-computed vs linear scan)."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Access pre-computed topology
        start = time.perf_counter()
        for _ in range(100000):
            _ = brain._component_connections.get("cortex", [])
        lookup_time = time.perf_counter() - start

        # Simulate old linear scan
        start = time.perf_counter()
        for _ in range(100000):
            matches = []
            for (src, tgt), pathway in brain.connections.items():
                if tgt == "cortex":
                    matches.append((src, pathway))
        scan_time = time.perf_counter() - start

        speedup = scan_time / lookup_time

        print(f"\n{'='*60}")
        print(f"Connection Lookup Overhead")
        print(f"{'='*60}")
        print(f"Pre-computed lookup: {lookup_time*1000:.2f} ms (100k lookups)")
        print(f"Linear scan: {scan_time*1000:.2f} ms (100k lookups)")
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"{'='*60}\n")

        # Pre-computed should be at least 5x faster
        assert speedup > 5.0, f"Pre-computed lookup not fast enough: {speedup:.1f}x"

    def test_benchmark_dict_reuse_overhead(self, global_config):
        """Measure dict allocation overhead (reuse vs new allocation)."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Reuse dict
        reusable = {}
        start = time.perf_counter()
        for _ in range(100000):
            reusable.clear()
            reusable["input"] = torch.zeros(128)
            reusable["feedback"] = torch.zeros(64)
        reuse_time = time.perf_counter() - start

        # New allocation each time
        start = time.perf_counter()
        for _ in range(100000):
            new_dict = {}
            new_dict["input"] = torch.zeros(128)
            new_dict["feedback"] = torch.zeros(64)
        alloc_time = time.perf_counter() - start

        speedup = alloc_time / reuse_time

        print(f"\n{'='*60}")
        print(f"Dict Allocation Overhead")
        print(f"{'='*60}")
        print(f"Dict reuse: {reuse_time*1000:.2f} ms (100k ops)")
        print(f"New allocation: {alloc_time*1000:.2f} ms (100k ops)")
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"{'='*60}\n")

        # Reuse should be at least 1.2x faster
        assert speedup > 1.2, f"Dict reuse not faster: {speedup:.1f}x"

    def test_benchmark_spike_counting_gpu_sync(self, global_config):
        """Measure GPU sync overhead (accumulate on GPU vs sync each step)."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        device = "cuda"
        spikes = torch.rand(128, device=device) > 0.5

        # Accumulate on GPU (optimized)
        counter = torch.tensor(0, dtype=torch.int64, device=device)
        start = time.perf_counter()
        for _ in range(10000):
            counter += spikes.sum()
        final_count = int(counter.item())  # Sync once at end
        gpu_time = time.perf_counter() - start

        # Sync each step (old way)
        total = 0
        start = time.perf_counter()
        for _ in range(10000):
            total += int(spikes.sum().item())  # Sync every iteration
        cpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time

        print(f"\n{'='*60}")
        print(f"GPU Spike Counting Overhead")
        print(f"{'='*60}")
        print(f"GPU accumulate: {gpu_time*1000:.2f} ms (10k steps)")
        print(f"CPU sync each step: {cpu_time*1000:.2f} ms (10k steps)")
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"{'='*60}\n")

        # GPU accumulation should be at least 5x faster
        assert speedup > 5.0, f"GPU optimization not effective: {speedup:.1f}x"


if __name__ == "__main__":
    """Run benchmarks directly."""
    import sys

    config = GlobalConfig(device="cpu", dt_ms=1.0)
    suite = TestClockDrivenPerformance()

    print("\n" + "="*60)
    print("CLOCK-DRIVEN EXECUTION BENCHMARKS")
    print("="*60 + "\n")

    try:
        suite.test_benchmark_sensorimotor_brain_1000_steps(config)
        suite.test_benchmark_minimal_brain_10000_steps(config)
        suite.test_benchmark_connection_lookup_overhead(config)
        suite.test_benchmark_dict_reuse_overhead(config)

        if torch.cuda.is_available():
            config.device = "cuda"
            suite.test_benchmark_spike_counting_gpu_sync(config)
        else:
            print("\nSkipping GPU benchmark (CUDA not available)\n")

        print("\n" + "="*60)
        print("ALL BENCHMARKS PASSED")
        print("="*60 + "\n")

    except AssertionError as e:
        print(f"\n❌ Benchmark failed: {e}\n")
        sys.exit(1)
