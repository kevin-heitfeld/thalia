"""Configuration classes for brain regions and overall brain architecture."""

from .neural_region import (
    HomeostaticGainConfig,
    HomeostaticThresholdConfig,
    NMReceptorConfig,
    NeuralRegionConfig,
    SynapticScalingConfig,
)
from .amygdala import (
    AmygdalaNucleusConfig,
    BasolateralAmygdalaConfig,
    BLAPopulationConfig,
    CeAPopulationConfig,
    CentralAmygdalaConfig,
)
from .basal_ganglia import (
    BGOutputConfig,
    BGPopulationConfig,
    GPeConfig,
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
    LayerFractions,
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
    # General region config + sub-configs
    "NMReceptorConfig",
    "NeuralRegionConfig",
    "HomeostaticGainConfig",
    "HomeostaticThresholdConfig",
    "SynapticScalingConfig",
    # Amygdala configs
    "AmygdalaNucleusConfig",
    "BasolateralAmygdalaConfig",
    "BLAPopulationConfig",
    "CeAPopulationConfig",
    "CentralAmygdalaConfig",
    # Basal ganglia configs
    "BGOutputConfig",
    "BGPopulationConfig",
    "GPeConfig",
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
    "LayerFractions",
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
