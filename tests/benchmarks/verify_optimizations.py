"""Verify optimization infrastructure is set up correctly."""

from thalia.core.brain_builder import BrainBuilder
from thalia.config import GlobalConfig


def verify_optimizations():
    """Check that optimization attributes exist and are populated."""
    config = GlobalConfig(device="cpu", dt_ms=1.0)
    brain = BrainBuilder.preset("default", config)

    print("\nOptimization Infrastructure Check:\n")

    # Check _component_connections
    has_conn = hasattr(brain, '_component_connections')
    print(f"1. _component_connections exists: {has_conn}")
    if has_conn:
        print(f"   - Type: {type(brain._component_connections)}")
        print(f"   - Components with incoming connections: {len(brain._component_connections)}")
        for comp_name, conns in brain._component_connections.items():
            print(f"     - {comp_name}: {len(conns)} incoming connections")

    # Check _reusable_component_inputs
    has_reusable = hasattr(brain, '_reusable_component_inputs')
    print(f"\n2. _reusable_component_inputs exists: {has_reusable}")
    if has_reusable:
        print(f"   - Type: {type(brain._reusable_component_inputs)}")

    # Check _output_cache
    has_cache = hasattr(brain, '_output_cache')
    print(f"\n3. _output_cache exists: {has_cache}")
    if has_cache:
        print(f"   - Type: {type(brain._output_cache)}")
        print(f"   - Pre-allocated entries: {len(brain._output_cache)}")

    # Check _spike_tensors
    has_spike_tensors = hasattr(brain, '_spike_tensors')
    print(f"\n4. _spike_tensors exists: {has_spike_tensors}")
    if has_spike_tensors:
        print(f"   - Type: {type(brain._spike_tensors)}")
        print(f"   - Components: {len(brain._spike_tensors)}")

    # Check _spike_counts
    has_spike_counts = hasattr(brain, '_spike_counts')
    print(f"\n5. _spike_counts exists: {has_spike_counts}")
    if has_spike_counts:
        print(f"   - Type: {type(brain._spike_counts)}")
        print(f"   - Components: {len(brain._spike_counts)}")

    print("\n" + "="*60)
    if all([has_conn, has_reusable, has_cache, has_spike_tensors, has_spike_counts]):
        print("✓ All optimization infrastructure is in place")
    else:
        print("⚠️  Some optimization attributes are missing")
    print("="*60 + "\n")


if __name__ == "__main__":
    verify_optimizations()
