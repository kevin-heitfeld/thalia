"""Neural Pathways Package.

All inter-region and sensory pathways for the Thalia brain architecture.

Pathways ARE mini-regions with neurons, synapses, and learning rules.
They actively transform information and learn continuously via STDP/BCM.

v2.0 Architecture:
- AxonalProjection: Pure spike routing, NO weights (new in v2.0)
- SpikingPathway: Traditional pathway with weights (legacy, v1.x)
"""

# Protocol
from thalia.pathways.protocol import NeuralPathway

# Core pathway types
from thalia.pathways.axonal_projection import AxonalProjection  # v2.0
from thalia.pathways.spiking_pathway import (
    SpikingPathway,
    SpikingLearningRule,
    TemporalCoding,
)
from thalia.pathways.spiking_replay import (
    SpikingReplayPathway,
    SpikingReplayPathwayConfig,
)

# Sensory pathways
from thalia.pathways.sensory_pathways import (
    SensoryPathway,
    VisualPathway,
    AuditoryPathway,
    LanguagePathway,
)

# Attention pathways (from attention submodule)
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
    # Protocol
    "NeuralPathway",
    # Core pathways
    "AxonalProjection",
    "SpikingPathway",
    "SpikingLearningRule",
    "TemporalCoding",
    "SpikingReplayPathway",
    "SpikingReplayPathwayConfig",
    # Sensory pathways
    "SensoryPathway",
    "VisualPathway",
    "AuditoryPathway",
    "LanguagePathway",
    # Attention pathways
    "AttentionMechanisms",
    "AttentionMechanismsConfig",
    "AttentionStage",
    "SpikingAttentionPathway",
    "SpikingAttentionPathwayConfig",
    "CrossModalGammaBinding",
    "CrossModalBindingConfig",
]
