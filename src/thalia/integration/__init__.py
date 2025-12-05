"""
Integration module for multi-region brain systems.

This module provides:
- SpikingPathway: Learnable inter-region connections with spike-based plasticity
- EventDrivenBrain: Event-driven brain with input buffering and parallel execution

The key insight is that inter-region connections in the brain are not static -
they are plastic and learn according to spike-timing dependent rules.

Example usage:
    from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
    
    brain = EventDrivenBrain(EventDrivenBrainConfig(
        input_size=256,
        cortex_size=128,
        hippocampus_size=64,
        pfc_size=32,
        n_actions=2,
    ))
    
    result = brain.process_sample(sample_pattern)
    result = brain.delay(10)
    result = brain.process_test(test_pattern)
    action, confidence = brain.select_action()
    brain.deliver_reward(reward=1.0)
    
    # Sleep consolidation
    brain.sleep_epoch(n_cycles=5)
"""

from .spiking_pathway import SpikingPathway, SpikingPathwayConfig

# Note: EventDrivenBrain should be imported from thalia.core.brain directly
# to avoid circular imports

__all__ = [
    "SpikingPathway",
    "SpikingPathwayConfig",
]
