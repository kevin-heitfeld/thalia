"""Simple timing test to check if optimizations helped or hurt."""

import time
import torch
from thalia.core.brain_builder import BrainBuilder
from thalia.config import GlobalConfig


def measure_throughput(n_timesteps=100, n_trials=5):
    """Measure throughput over multiple trials."""
    config = GlobalConfig(device="cpu", dt_ms=1.0)
    brain = BrainBuilder.preset("default", config)

    # Warm-up
    sensory_input = torch.rand(128, device=config.device) > 0.5
    brain.forward(sensory_input, n_timesteps=10)

    # Timed trials
    times = []
    for trial in range(n_trials):
        brain.reset_state()
        sensory_input = torch.rand(128, device=config.device) > 0.5

        start = time.perf_counter()
        brain.forward(sensory_input, n_timesteps=n_timesteps)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        throughput = n_timesteps / elapsed
        print(f"Trial {trial+1}: {elapsed:.3f}s  ({throughput:.1f} timesteps/sec)")

    # Stats
    avg_time = sum(times) / len(times)
    avg_throughput = n_timesteps / avg_time

    print(f"\nAverage: {avg_time:.3f}s ({avg_throughput:.1f} timesteps/sec)")
    print(f"Per timestep: {(avg_time / n_timesteps) * 1000:.3f} ms\n")

    return avg_throughput


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CLOCK-DRIVEN THROUGHPUT TEST")
    print("="*60 + "\n")

    throughput = measure_throughput(n_timesteps=100, n_trials=5)

    # Baseline expectation: should be at least 50 timesteps/sec on CPU
    # With optimizations: ideally 100+ timesteps/sec
    if throughput < 50:
        print(f"⚠️  WARNING: Throughput is low ({throughput:.1f} timesteps/sec)")
        print("   Expected: >50 timesteps/sec")
        print("   Possible issue: Performance regression or slow components\n")
    elif throughput < 100:
        print(f"✓  Throughput is acceptable ({throughput:.1f} timesteps/sec)")
        print("   Room for improvement with further optimizations\n")
    else:
        print(f"✓✓ Throughput is good ({throughput:.1f} timesteps/sec)")
        print("   Optimizations are working well\n")
