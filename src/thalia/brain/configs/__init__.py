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
    SubstantiaNigraCompactaConfig,
    VTAConfig,
    get_default_gpe_config,
    get_default_gpi_config,
    get_default_lhb_config,
    get_default_rmtg_config,
    get_default_snr_config,
    get_default_stn_config,
)
from .brainstem import (
    CerebellumConfig,
    DorsalRapheNucleusConfig,
    LocusCoeruleusConfig,
    NucleusBasalisConfig,
)
from .cortical_column import (
    CorticalColumnConfig,
    PrefrontalCortexConfig,
)
from .hippocampus import (
    EntorhinalCortexConfig,
    HippocampusConfig,
    MedialSeptumConfig,
    SubiculumConfig,
    get_default_dg_layer_config,
    get_default_ca3_layer_config,
    get_default_ca2_layer_config,
    get_default_ca1_layer_config,
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
    "SubstantiaNigraCompactaConfig",
    "VTAConfig",
    # Basal ganglia config factories
    "get_default_gpe_config",
    "get_default_gpi_config",
    "get_default_lhb_config",
    "get_default_rmtg_config",
    "get_default_snr_config",
    "get_default_stn_config",
    # Brainstem configs
    "CerebellumConfig",
    "DorsalRapheNucleusConfig",
    "LocusCoeruleusConfig",
    "NucleusBasalisConfig",
    # Cortical column configs
    "CorticalColumnConfig",
    "PrefrontalCortexConfig",
    # Hippocampus configs
    "EntorhinalCortexConfig",
    "HippocampusConfig",
    "MedialSeptumConfig",
    "SubiculumConfig",
    "get_default_dg_layer_config",
    "get_default_ca3_layer_config",
    "get_default_ca2_layer_config",
    "get_default_ca1_layer_config",
    # Thalamus config
    "ThalamusConfig",
    # Overall brain config
    "BrainConfig",
]
