"""
Integration module for multi-region brain systems.

This module provides:
- SpikingPathway: Learnable inter-region connections with spike-based plasticity
- EventDrivenBrain: Event-driven brain with input buffering and parallel execution

The key insight is that inter-region connections in the brain are not static -
they are plastic and learn according to spike-timing dependent rules.

Example usage:
    from thalia.core.brain import EventDrivenBrain
    from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

    config = ThaliaConfig(
        global_=GlobalConfig(device="cpu"),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=256,
                cortex_size=128,
                hippocampus_size=64,
                pfc_size=32,
                n_actions=2,
            ),
        ),
    )

    brain = EventDrivenBrain.from_thalia_config(config)

    # Process input (encoding, maintenance, or retrieval)
    result = brain.forward(sample_pattern, n_timesteps=15)
    result = brain.forward(None, n_timesteps=10)  # Maintenance period
    result = brain.forward(test_pattern, n_timesteps=15)

    # Action selection and learning
    action, confidence = brain.select_action()
    brain.deliver_reward(external_reward=1.0)  # Combines with intrinsic rewards

    # Sleep consolidation
    brain.consolidate(n_cycles=5)
"""

from .spiking_pathway import SpikingPathway

# Note: EventDrivenBrain should be imported from thalia.core.brain directly
# to avoid circular imports

__all__ = [
    "SpikingPathway",
]
