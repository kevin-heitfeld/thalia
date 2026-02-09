"""Neural Regions for Dynamic Brain Modeling."""

from .axonal_projection import (
    AxonalProjection,
    AxonalProjectionSourceSpec,
)
from .brain_builder import (
    BrainBuilder,
)
from .brain import (
    DynamicBrain,
)
from .configs import (
    BrainConfig,
    RegionType,
    CortexLayer,
    NeuralRegionConfig,
    MedialSeptumConfig,
    CerebellumConfig,
    CortexConfig,
    HippocampusConfig,
    PrefrontalConfig,
    StriatumConfig,
    ThalamusConfig,
)
from .regions import (
    NeuralRegion,
    Cerebellum,
    Cortex,
    Hippocampus,
    Prefrontal,
    Striatum,
    StriatumStateTracker,
    Thalamus,
    StimulusGating,
    NeuralRegionRegistry,
    register_region,
)

__all__ = [
    # Axonal Projections
    "AxonalProjection",
    "AxonalProjectionSourceSpec",
    # Configurations
    "BrainConfig",
    "RegionType",
    "CortexLayer",
    "NeuralRegionConfig",
    "MedialSeptumConfig",
    "CerebellumConfig",
    "CortexConfig",
    "HippocampusConfig",
    "PrefrontalConfig",
    "StriatumConfig",
    "ThalamusConfig",
    # Regions
    "NeuralRegion",
    "Cerebellum",
    "Cortex",
    "Hippocampus",
    "Prefrontal",
    "Striatum",
    "StriatumStateTracker",
    "Thalamus",
    "StimulusGating",
    "NeuralRegionRegistry",
    "register_region",
    # Brain Builder
    "BrainBuilder",
    # Dynamic Brain
    "DynamicBrain",
]
