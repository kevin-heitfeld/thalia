"""
Pathway submodule for specialized inter-region connections.

Each pathway type implements spike-based learning appropriate
for its biological function.

Available pathways:
- SpikingAttentionPathway: Fully spiking attention with temporal coding
- SpikingReplayPathway: Fully spiking replay with phase coding
"""

from .spiking_attention import SpikingAttentionPathway, SpikingAttentionPathwayConfig
from .spiking_replay import SpikingReplayPathway, SpikingReplayPathwayConfig

__all__ = [
    "SpikingAttentionPathway",
    "SpikingAttentionPathwayConfig",
    "SpikingReplayPathway",
    "SpikingReplayPathwayConfig",
]
