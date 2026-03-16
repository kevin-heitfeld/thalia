"""Configuration classes for brain regions and overall brain architecture."""

from .neural_region import NeuralRegionConfig
from .amygdala import (
    AmygdalaNucleusConfig,
    BasolateralAmygdalaConfig,
    CentralAmygdalaConfig,
)
from .basal_ganglia import (
    TonicPacemakerConfig,
    DopaminePacemakerConfig,
    StriatumConfig,
    VTAConfig,
)
from .brainstem import (
    CerebellumConfig,
    DorsalRapheNucleusConfig,
    LocusCoeruleusConfig,
)
from .cortical_column import (
    CorticalColumnConfig,
    CorticalPopulationConfig,
)
from .hippocampus import (
    EntorhinalCortexConfig,
    EntorhinalCortexPopulationConfig,
    HippocampusConfig,
    HippocampalPopulationConfig,
    MedialSeptumConfig,
    SubiculumConfig,
)
from .thalamus import ThalamusConfig
from .brain import BrainConfig


__all__ = [
    # General region config
    "NeuralRegionConfig",
    # Amygdala configs
    "AmygdalaNucleusConfig",
    "BasolateralAmygdalaConfig",
    "CentralAmygdalaConfig",
    # Basal ganglia configs
    "TonicPacemakerConfig",
    "DopaminePacemakerConfig",
    "StriatumConfig",
    "VTAConfig",
    # Brainstem configs
    "CerebellumConfig",
    "DorsalRapheNucleusConfig",
    "LocusCoeruleusConfig",
    # Cortical column configs
    "CorticalColumnConfig",
    "CorticalPopulationConfig",
    # Hippocampus configs
    "EntorhinalCortexConfig",
    "EntorhinalCortexPopulationConfig",
    "HippocampusConfig",
    "HippocampalPopulationConfig",
    "MedialSeptumConfig",
    "SubiculumConfig",
    # Thalamus config
    "ThalamusConfig",
    # Overall brain config
    "BrainConfig",
]
