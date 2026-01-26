"""
THALIA Unified Configuration System.

This module provides a centralized configuration system that:
1. Eliminates parameter duplication across modules
2. Provides clear inheritance of global parameters
3. Validates configuration consistency
4. Shows effective values being used

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from .base import BaseConfig
from .brain_config import (
    BrainConfig,
    CortexType,
    NeuromodulationConfig,
)
from .curriculum_growth import (
    CurriculumStage,
    get_curriculum_growth_config,
)
from .learning_config import (
    BaseLearningConfig,
    HebbianLearningConfig,
    ModulatedLearningConfig,
    STDPLearningConfig,
)
from .region_configs import (
    CerebellumConfig,
    HippocampusConfig,
    LayeredCortexConfig,
    MultimodalIntegrationConfig,
    PredictiveCortexConfig,
    PrefrontalConfig,
    StriatumConfig,
    ThalamicRelayConfig,
)
from .size_calculator import BiologicalRatios, LayerSizeCalculator

__all__ = [
    # Base configs
    "BaseConfig",
    "BaseLearningConfig",
    "ModulatedLearningConfig",
    "STDPLearningConfig",
    "HebbianLearningConfig",
    # Brain configs
    "BrainConfig",
    "CortexType",
    "NeuromodulationConfig",
    # Curriculum growth
    "CurriculumStage",
    "get_curriculum_growth_config",
    # Region configs
    "LayeredCortexConfig",
    "PredictiveCortexConfig",
    "HippocampusConfig",
    "StriatumConfig",
    "PrefrontalConfig",
    "CerebellumConfig",
    "ThalamicRelayConfig",
    "MultimodalIntegrationConfig",
    # Size calculator
    "LayerSizeCalculator",
    "BiologicalRatios",
]
