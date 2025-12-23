"""Neural Pathways Package.

All inter-region and sensory pathways for the Thalia brain architecture.

Pathways ARE mini-regions with neurons, synapses, and learning rules.
They actively transform information and learn continuously via STDP/BCM.

v3.0 Architecture:
- AxonalProjection: Pure spike routing, NO weights
- All learning happens at dendrites (NeuralRegion pattern)
"""

# Protocol
from thalia.pathways.protocol import NeuralPathway

# Core pathway types
from thalia.pathways.axonal_projection import AxonalProjection

# Sensory pathways
from thalia.pathways.sensory_pathways import (
    SensoryPathway,
    VisualPathway,
    AuditoryPathway,
    LanguagePathway,
)

# Sensory processing constants
from thalia.pathways.sensory_constants import (
    AUDITORY_NERVE_ADAPTATION_DECAY,
    AUDITORY_NERVE_ADAPTATION_RATE,
    COCHLEA_INTEGRATION_WINDOW_MS,
    COCHLEA_MAX_FREQ_HZ,
    COCHLEA_MIN_FREQ_HZ,
    DOG_FILTER_SIZE,
    DOG_SIGMA_CENTER,
    DOG_SIGMA_SURROUND,
    HAIR_CELL_ADAPTATION_SUPPRESSION,
    HAIR_CELL_COMPRESSION_EXPONENT,
    LATENCY_EPSILON,
    RETINA_ADAPTATION_DECAY,
    RETINA_ADAPTATION_RATE,
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
    # Sensory constants
    "RETINA_ADAPTATION_DECAY",
    "RETINA_ADAPTATION_RATE",
    "DOG_FILTER_SIZE",
    "DOG_SIGMA_CENTER",
    "DOG_SIGMA_SURROUND",
    "COCHLEA_MIN_FREQ_HZ",
    "COCHLEA_MAX_FREQ_HZ",
    "COCHLEA_INTEGRATION_WINDOW_MS",
    "HAIR_CELL_COMPRESSION_EXPONENT",
    "HAIR_CELL_ADAPTATION_SUPPRESSION",
    "AUDITORY_NERVE_ADAPTATION_DECAY",
    "AUDITORY_NERVE_ADAPTATION_RATE",
    "LATENCY_EPSILON",
]
