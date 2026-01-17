"""Neural Pathways Package.

All inter-region and sensory pathways for the Thalia brain architecture.

Pathways ARE mini-regions with neurons, synapses, and learning rules.
They actively transform information and learn continuously via STDP/BCM.

v3.0 Architecture:
- AxonalProjection: Pure spike routing, NO weights
- All learning happens at dendrites (NeuralRegion pattern)
"""

from __future__ import annotations

# Core pathway types
from thalia.pathways.axonal_projection import AxonalProjection

# Protocol
from thalia.pathways.protocol import NeuralPathway

# Sensory pathways
from thalia.pathways.sensory_pathways import (
    AuditoryPathway,
    LanguagePathway,
    SensoryPathway,
    VisualPathway,
)

__all__ = [
    # Protocol
    "NeuralPathway",
    # Core pathways
    "AxonalProjection",
    # Sensory pathways
    "SensoryPathway",
    "VisualPathway",
    "AuditoryPathway",
    "LanguagePathway",
]
