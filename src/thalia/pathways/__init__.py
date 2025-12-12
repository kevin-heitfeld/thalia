"""Neural Pathways Package.

All inter-region and sensory pathways for the Thalia brain architecture.

Pathways ARE mini-regions with neurons, synapses, and learning rules.
They actively transform information and learn continuously via STDP/BCM.
"""

# Protocol
from thalia.pathways.protocol import NeuralPathway

# Manager
from thalia.pathways.manager import PathwayManager

# Core pathway types
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
    # Manager
    "PathwayManager",
    # Core pathways
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
