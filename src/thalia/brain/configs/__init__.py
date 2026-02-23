"""Configuration classes for brain regions and overall brain architecture."""

from .neural_region import NeuralRegionConfig
from .amygdala import (
    BasolateralAmygdalaConfig,
    CentralAmygdalaConfig,
)
from .basal_ganglia import (
    GlobusPallidusExternaConfig,
    LateralHabenulaConfig,
    RostromedialTegmentumConfig,
    StriatumConfig,
    SubstantiaNigraCompactaConfig,
    SubstantiaNigraConfig,
    SubthalamicNucleusConfig,
    VTAConfig,
)
from .brainstem import (
    CerebellumConfig,
    DorsalRapheNucleusConfig,
    LocusCoeruleusConfig,
    NucleusBasalisConfig,
)
from .cortex import (
    CortexLayer,
    CortexConfig,
    PrefrontalConfig,
)
from .hippocampus import (
    EntorhinalCortexConfig,
    HippocampusConfig,
    MedialSeptumConfig,
)
from .thalamus import ThalamusConfig
from .brain import BrainConfig


__all__ = [
    # General region config
    "NeuralRegionConfig",
    # Amygdala configs
    "BasolateralAmygdalaConfig",
    "CentralAmygdalaConfig",
    # Basal ganglia configs
    "GlobusPallidusExternaConfig",
    "LateralHabenulaConfig",
    "RostromedialTegmentumConfig",
    "StriatumConfig",
    "SubstantiaNigraCompactaConfig",
    "SubstantiaNigraConfig",
    "SubthalamicNucleusConfig",
    "VTAConfig",
    # Brainstem configs
    "CerebellumConfig",
    "DorsalRapheNucleusConfig",
    "LocusCoeruleusConfig",
    "NucleusBasalisConfig",
    # Cortex configs
    "CortexLayer",
    "CortexConfig",
    "PrefrontalConfig",
    # Hippocampus configs
    "EntorhinalCortexConfig",
    "HippocampusConfig",
    "MedialSeptumConfig",
    # Thalamus config
    "ThalamusConfig",
    # Overall brain config
    "BrainConfig",
]
