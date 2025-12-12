"""Attention Pathways.

Specialized pathways for attention mechanisms and crossmodal binding.
"""

from thalia.pathways.attention.attention import (
    AttentionMechanisms,
    AttentionMechanismsConfig,
    AttentionStage,
)
from thalia.pathways.attention.spiking_attention import (
    SpikingAttentionPathway,
    SpikingAttentionPathwayConfig,
)
from thalia.pathways.attention.crossmodal_binding import (
    CrossModalGammaBinding,
    CrossModalBindingConfig,
)

__all__ = [
    "AttentionMechanisms",
    "AttentionMechanismsConfig",
    "AttentionStage",
    "SpikingAttentionPathway",
    "SpikingAttentionPathwayConfig",
    "CrossModalGammaBinding",
    "CrossModalBindingConfig",
]
