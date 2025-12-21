"""Profile component-level execution to find bottlenecks."""

import time
import torch
from thalia.core.brain_builder import BrainBuilder
from thalia.config import GlobalConfig


def profile_component_timing():
    """Profile individual component execution times."""
    config = GlobalConfig(device="cpu", dt_ms=1.0)
    brain = BrainBuilder.preset("sensorimotor", config)

    # Warm-up
    sensory_input = torch.rand(128, device=config.device) > 0.5
    brain.forward(sensory_input, n_timesteps=5)
    brain.reset_state()

    # Manually execute one timestep and time each component
    print("\nProfiling component execution (single timestep):\n")

    sensory_input = torch.rand(128, device=config.device) > 0.5
    brain._output_cache["thalamus"] = sensory_input

    total_time = 0.0
    for comp_name in brain._get_execution_order():
        component = brain.components[comp_name]

        # Prepare inputs (simplified version of main loop)
        component_inputs = {}
        for src, pathway in brain._component_connections.get(comp_name, []):
            if src in brain._output_cache and brain._output_cache[src] is not None:
                delayed = pathway.forward({src: brain._output_cache[src]})
                component_inputs.update(delayed)

        # Time this component
        start = time.perf_counter()
        output = component.forward(component_inputs)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        brain._output_cache[comp_name] = output
        total_time += elapsed

        print(f"  {comp_name:15s}: {elapsed:.3f} ms")

    print(f"\n  {'TOTAL':15s}: {total_time:.3f} ms\n")

    # Compare with actual forward pass
    brain.reset_state()
    start = time.perf_counter()
    brain.forward(sensory_input, n_timesteps=1)
    actual_time = (time.perf_counter() - start) * 1000

    print(f"Actual forward(1 timestep): {actual_time:.3f} ms")
    print(f"Overhead: {actual_time - total_time:.3f} ms ({((actual_time - total_time) / actual_time * 100):.1f}%)\n")


if __name__ == "__main__":
    profile_component_timing()
