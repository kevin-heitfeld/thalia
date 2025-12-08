"""
Pathway submodule for specialized inter-region connections.

Each pathway type implements spike-based learning appropriate
for its biological function.

Available pathways:
- SpikingAttentionPathway: Fully spiking attention with temporal coding
- SpikingReplayPathway: Fully spiking replay with phase coding
- AttentionMechanisms: Enhanced attention with bottom-up/top-down integration
- CrossModalGammaBinding: Multimodal binding via gamma synchrony
"""

from .spiking_attention import SpikingAttentionPathway, SpikingAttentionPathwayConfig
from .spiking_replay import SpikingReplayPathway, SpikingReplayPathwayConfig
from .attention import (
    AttentionMechanisms,
    AttentionMechanismsConfig,
    AttentionStage,
)
from .crossmodal_binding import CrossModalGammaBinding, CrossModalBindingConfig

__all__ = [
    "SpikingAttentionPathway",
    "SpikingAttentionPathwayConfig",
    "SpikingReplayPathway",
    "SpikingReplayPathwayConfig",
    "AttentionMechanisms",
    "AttentionMechanismsConfig",
    "AttentionStage",
    "CrossModalGammaBinding",
    "CrossModalBindingConfig",
]
