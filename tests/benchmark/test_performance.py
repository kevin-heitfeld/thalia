"""
Performance benchmark tests to prevent regression.

These tests track performance metrics over time and fail if
performance degrades significantly.
"""

import pytest
import torch
import time
from typing import Callable

from thalia.core.neuron import LIFNeuron
from thalia.core.dendritic import DendriticNeuron, DendriticNeuronConfig
from thalia.regions import LayeredCortex, LayeredCortexConfig
from thalia.learning.ei_balance import EIBalanceRegulator


# Performance thresholds (in seconds)
PERFORMANCE_THRESHOLDS = {
    "lif_forward_small": 0.001,      # 1ms for 100 neurons
    "lif_forward_large": 0.01,       # 10ms for 10k neurons
    "cortex_forward_medium": 0.02,   # 20ms for medium cortex
    "cortex_forward_large": 0.1,     # 100ms for large cortex
    "dendritic_forward": 0.005,      # 5ms for dendritic neuron
    "ei_balance_update": 0.001,      # 1ms for E/I balance update
}


def benchmark_function(func: Callable, n_runs: int = 100) -> dict:
    """Benchmark a function over multiple runs.

    Args:
        func: Function to benchmark (no arguments)
        n_runs: Number of times to run

    Returns:
        Dict with timing statistics
    """
    times = []

    # Warmup run (JIT compilation, cache warming)
    func()

    # Timed runs
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    times_tensor = torch.tensor(times)

    return {
        "mean": times_tensor.mean().item(),
        "std": times_tensor.std().item(),
        "min": times_tensor.min().item(),
        "max": times_tensor.max().item(),
        "median": times_tensor.median().item(),
    }


@pytest.mark.benchmark
@pytest.mark.slow
class TestLIFPerformance:
    """Benchmark LIF neuron performance."""

    def test_lif_forward_small(self):
        """Benchmark small LIF forward pass (100 neurons)."""
        neuron = LIFNeuron(n_neurons=100)
        neuron.reset_state(batch_size=32)
        input_current = torch.randn(32, 100)

        def forward():
            neuron(input_current)

        stats = benchmark_function(forward, n_runs=100)

        print(f"\nLIF Forward (100 neurons, batch=32):")
        print(f"  Mean: {stats['mean']*1000:.3f} ms")
        print(f"  Std:  {stats['std']*1000:.3f} ms")

        threshold = PERFORMANCE_THRESHOLDS["lif_forward_small"]
        assert stats["mean"] < threshold, \
            f"LIF forward pass too slow: {stats['mean']:.4f}s > {threshold}s threshold"

    def test_lif_forward_large(self):
        """Benchmark large LIF forward pass (10k neurons)."""
        neuron = LIFNeuron(n_neurons=10000)
        neuron.reset_state(batch_size=32)
        input_current = torch.randn(32, 10000)

        def forward():
            neuron(input_current)

        stats = benchmark_function(forward, n_runs=50)

        print(f"\nLIF Forward (10k neurons, batch=32):")
        print(f"  Mean: {stats['mean']*1000:.3f} ms")
        print(f"  Std:  {stats['std']*1000:.3f} ms")

        threshold = PERFORMANCE_THRESHOLDS["lif_forward_large"]
        assert stats["mean"] < threshold, \
            f"Large LIF forward too slow: {stats['mean']:.4f}s > {threshold}s"

    def test_lif_batch_scaling(self):
        """Test that LIF scales reasonably with batch size."""
        neuron = LIFNeuron(n_neurons=1000)

        timings = {}
        for batch_size in [1, 8, 32, 128]:
            neuron.reset_state(batch_size=batch_size)
            input_current = torch.randn(batch_size, 1000)

            def forward():
                neuron(input_current)

            stats = benchmark_function(forward, n_runs=50)
            timings[batch_size] = stats["mean"]

        print(f"\nLIF Batch Scaling (1000 neurons):")
        for bs, t in timings.items():
            print(f"  Batch {bs:3d}: {t*1000:.3f} ms")

        # Should scale sublinearly (batching is efficient)
        # Time for batch=128 should be < 10x time for batch=1
        assert timings[128] < timings[1] * 10, \
            "Batch processing not efficient enough"


@pytest.mark.benchmark
@pytest.mark.slow
class TestCortexPerformance:
    """Benchmark cortex performance."""

    @pytest.mark.skip(reason="LayeredCortex only supports batch_size=1 (single continuous brain state)")
    def test_cortex_forward_medium(self):
        """Benchmark medium cortex forward pass."""
        config = LayeredCortexConfig(n_input=256, n_output=128)
        cortex = LayeredCortex(config)
        cortex.reset_state()
        input_data = torch.randn(32, 256)

        def forward():
            cortex.forward(input_data)

        stats = benchmark_function(forward, n_runs=50)

        print(f"\nCortex Forward (256→128, batch=32):")
        print(f"  Mean: {stats['mean']*1000:.3f} ms")
        print(f"  Std:  {stats['std']*1000:.3f} ms")

        threshold = PERFORMANCE_THRESHOLDS["cortex_forward_medium"]
        assert stats["mean"] < threshold, \
            f"Cortex forward too slow: {stats['mean']:.4f}s > {threshold}s"

    @pytest.mark.skip(reason="LayeredCortex only supports batch_size=1 (single continuous brain state)")
    def test_cortex_forward_large(self):
        """Benchmark large cortex forward pass."""
        config = LayeredCortexConfig(n_input=1024, n_output=512)
        cortex = LayeredCortex(config)
        cortex.reset_state()
        input_data = torch.randn(32, 1024)

        def forward():
            cortex.forward(input_data)

        stats = benchmark_function(forward, n_runs=30)

        print(f"\nCortex Forward (1024→512, batch=32):")
        print(f"  Mean: {stats['mean']*1000:.3f} ms")
        print(f"  Std:  {stats['std']*1000:.3f} ms")

        threshold = PERFORMANCE_THRESHOLDS["cortex_forward_large"]
        assert stats["mean"] < threshold, \
            f"Large cortex too slow: {stats['mean']:.4f}s > {threshold}s"

    @pytest.mark.skip(reason="LayeredCortex only supports batch_size=1 (single continuous brain state)")
    def test_cortex_with_robustness_overhead(self):
        """Measure overhead of robustness mechanisms."""
        from thalia.config import RobustnessConfig

        # Cortex without robustness
        config_no_rob = LayeredCortexConfig(
            n_input=256,
            n_output=128,
            robustness=None,
        )
        cortex_no_rob = LayeredCortex(config_no_rob)
        cortex_no_rob.reset_state()

        # Cortex with full robustness
        config_with_rob = LayeredCortexConfig(
            n_input=256,
            n_output=128,
            robustness=RobustnessConfig.full(),
        )
        cortex_with_rob = LayeredCortex(config_with_rob)
        cortex_with_rob.reset_state()

        input_data = torch.randn(32, 256)

        # Benchmark both
        def forward_no_rob():
            cortex_no_rob.forward(input_data)

        def forward_with_rob():
            cortex_with_rob.forward(input_data)

        stats_no_rob = benchmark_function(forward_no_rob, n_runs=50)
        stats_with_rob = benchmark_function(forward_with_rob, n_runs=50)

        overhead = (stats_with_rob["mean"] - stats_no_rob["mean"]) / stats_no_rob["mean"]

        print(f"\nRobustness Overhead:")
        print(f"  Without: {stats_no_rob['mean']*1000:.3f} ms")
        print(f"  With:    {stats_with_rob['mean']*1000:.3f} ms")
        print(f"  Overhead: {overhead*100:.1f}%")

        # Overhead should be reasonable (<100%)
        assert overhead < 1.0, \
            f"Robustness overhead too high: {overhead*100:.1f}%"


@pytest.mark.benchmark
@pytest.mark.slow
class TestDendriticPerformance:
    """Benchmark dendritic neuron performance."""

    def test_dendritic_forward(self):
        """Benchmark dendritic neuron forward pass."""
        config = DendriticNeuronConfig(
            n_branches=5,
            inputs_per_branch=20,
        )
        neuron = DendriticNeuron(n_neurons=100, config=config)
        neuron.reset_state(batch_size=32)
        input_spikes = torch.randn(32, 100)

        def forward():
            neuron(input_spikes)

        stats = benchmark_function(forward, n_runs=50)

        print(f"\nDendritic Forward (100 neurons, 5 branches):")
        print(f"  Mean: {stats['mean']*1000:.3f} ms")
        print(f"  Std:  {stats['std']*1000:.3f} ms")

        threshold = PERFORMANCE_THRESHOLDS["dendritic_forward"]
        assert stats["mean"] < threshold, \
            f"Dendritic forward too slow: {stats['mean']:.4f}s > {threshold}s"


@pytest.mark.benchmark
class TestLearningPerformance:
    """Benchmark learning mechanism performance."""

    def test_ei_balance_update(self):
        """Benchmark E/I balance regulator update."""
        regulator = EIBalanceRegulator()
        exc_spikes = torch.randn(32, 100)
        inh_spikes = torch.randn(32, 20)

        def update():
            regulator.update(exc_spikes, inh_spikes)

        stats = benchmark_function(update, n_runs=100)

        print(f"\nE/I Balance Update:")
        print(f"  Mean: {stats['mean']*1000:.3f} ms")
        print(f"  Std:  {stats['std']*1000:.3f} ms")

        threshold = PERFORMANCE_THRESHOLDS["ei_balance_update"]
        assert stats["mean"] < threshold, \
            f"E/I update too slow: {stats['mean']:.4f}s > {threshold}s"


@pytest.mark.benchmark
class TestMemoryUsage:
    """Test memory efficiency."""

    @pytest.mark.cuda
    @pytest.mark.skip(reason="LayeredCortex needs proper device handling for all subcomponents")
    def test_gpu_memory_usage(self):
        """Test GPU memory usage is reasonable."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = "cuda"

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # Create cortex on GPU
        config = LayeredCortexConfig(n_input=1024, n_output=512, device=device)
        cortex = LayeredCortex(config)
        cortex.reset_state()

        # Run forward pass with batch_size=1 (THALIA single-instance architecture)
        input_data = torch.randn(1, 1024, device=device)
        output = cortex.forward(input_data)

        # Check memory usage
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        print(f"\nGPU Memory Usage (1024→512 cortex):")
        print(f"  Peak: {memory_mb:.2f} MB")

        # Should fit in reasonable memory budget
        MAX_MEMORY_MB = 1000  # 1 GB
        assert memory_mb < MAX_MEMORY_MB, \
            f"Memory usage too high: {memory_mb:.2f} MB > {MAX_MEMORY_MB} MB"

        # Cleanup
        torch.cuda.empty_cache()

    def test_no_memory_leaks(self):
        """Test that repeated forward passes don't leak memory."""
        import gc

        config = LayeredCortexConfig(n_input=256, n_output=128)
        cortex = LayeredCortex(config)

        # Initial memory state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_mem = torch.cuda.memory_allocated()
        else:
            initial_mem = 0

        # Run many forward passes with batch_size=1 (THALIA single-instance architecture)
        for _ in range(100):
            cortex.reset_state()
            input_data = torch.randn(1, 256)
            output = cortex.forward(input_data)
            del output

        # Final memory state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_mem = torch.cuda.memory_allocated()
            mem_increase_mb = (final_mem - initial_mem) / (1024**2)

            print(f"\nMemory leak check (100 iterations):")
            print(f"  Memory increase: {mem_increase_mb:.2f} MB")

            # Small increase is OK (caching), but not huge leak
            MAX_INCREASE_MB = 100  # 100 MB
            assert mem_increase_mb < MAX_INCREASE_MB, \
                f"Possible memory leak: {mem_increase_mb:.2f} MB increase"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
