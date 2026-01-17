"""Quick profiling to identify bottlenecks in forward pass."""

import time

import torch

from thalia.config import GlobalConfig
from thalia.core.brain_builder import BrainBuilder


def profile_forward_pass():
    """Profile a single forward pass to find bottlenecks."""
    config = GlobalConfig(device="cpu", dt_ms=1.0)
    brain = BrainBuilder.preset("default", config)

    # Warm-up
    sensory_input = torch.rand(128, device=config.device) > 0.5
    brain.forward(sensory_input, n_timesteps=10)
    brain.reset_state()

    # Profile 100 timesteps
    print("\nProfiling 100 timesteps...")
    start = time.perf_counter()
    sensory_input = torch.rand(128, device=config.device) > 0.5
    brain.forward(sensory_input, n_timesteps=100)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Time per timestep: {(elapsed/100)*1000:.3f} ms")
    print(f"Throughput: {100/elapsed:.1f} timesteps/sec")

    # Check component counts
    print(f"\nBrain structure:")
    print(f"  Components: {len(brain.components)}")
    print(f"  Connections: {len(brain.connections)}")
    print(f"  Component names: {list(brain.components.keys())}")

    # Check spike counts
    spike_counts = brain.get_spike_counts()
    print(f"\nSpike activity over 100 timesteps:")
    for comp_name, count in sorted(spike_counts.items()):
        print(f"  {comp_name}: {count} spikes")


if __name__ == "__main__":
    profile_forward_pass()
