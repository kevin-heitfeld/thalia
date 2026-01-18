"""Profile pathway overhead."""

from thalia.config import BrainConfig
from thalia.core.brain_builder import BrainBuilder


def count_delay_buffers():
    """Count total delay buffers in the brain."""
    config = BrainConfig(device="cpu", dt_ms=1.0)
    brain = BrainBuilder.preset("default", config)

    total_buffers = 0
    for (src, tgt), pathway in brain.connections.items():
        if hasattr(pathway, "_delay_buffers"):
            n_buffers = len(pathway._delay_buffers)
            total_buffers += n_buffers
            print(f"  {src} -> {tgt}: {n_buffers} delay buffers")

    print(f"\nTotal delay buffers: {total_buffers}")
    print(f"Components: {len(brain.components)}")
    print(f"Connections: {len(brain.connections)}\n")


if __name__ == "__main__":
    print("\nDelay Buffer Count:\n")
    count_delay_buffers()
