"""
THALIA Unified Configuration System.

This module provides a centralized configuration system that:
1. Eliminates parameter duplication across modules
2. Provides clear inheritance of global parameters
3. Validates configuration consistency
4. Shows effective values being used

Usage:
======

    from thalia.config import ThaliaConfig, BrainConfig

    # Create configuration with customizations
    config = ThaliaConfig(
        brain=BrainConfig(device="cuda", vocab_size=10000, cortex_size=256, hippocampus_size=128),
    )

    # Show what values are actually being used
    config.summary()

    # Create brain with resolved config
    brain = DynamicBrain.from_config(config)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export component configs from core/
from thalia.core.base.component_config import (
    LearningComponentConfig,
    NeuralComponentConfig,
    PathwayConfig,
)

if TYPE_CHECKING:
    # Re-export region configs from canonical locations
    from thalia.regions.cerebellum import CerebellumConfig
    from thalia.regions.cortex.config import LayeredCortexConfig
    from thalia.regions.cortex.predictive_cortex import PredictiveCortexConfig
    from thalia.regions.cortex.robustness_config import RobustnessConfig
    from thalia.regions.hippocampus.config import HippocampusConfig
    from thalia.regions.prefrontal import PrefrontalConfig
    from thalia.regions.striatum.config import StriatumConfig

from .base import BaseConfig
from .brain_config import (
    BrainConfig,
    CortexType,
    NeuromodulationConfig,
    RegionSizes,
)
from .language_config import (
    DecodingConfig,
    EncodingConfig,
    LanguageConfig,
    PositionConfig,
    SequenceMemoryConfig,
)
from .learning_config import (
    BaseLearningConfig,
    HebbianLearningConfig,
    ModulatedLearningConfig,
    STDPLearningConfig,
)
from .size_calculator import BiologicalRatios, LayerSizeCalculator
from .thalia_config import ThaliaConfig, print_config
from .training_config import TrainingConfig
from .validation import (
    ConfigValidationError,
    ValidatedConfig,
    ValidatorRegistry,
    check_config_and_warn,
    validate_brain_config,
    validate_global_consistency,
    validate_region_sizes,
    validate_thalia_config,
)

__all__ = [
    # Main config
    "ThaliaConfig",
    "print_config",
    # Validation
    "validate_thalia_config",
    "validate_brain_config",
    "validate_global_consistency",
    "validate_region_sizes",
    "check_config_and_warn",
    "ConfigValidationError",
    "ValidatedConfig",
    "ValidatorRegistry",
    # Base configs
    "BaseConfig",
    "BaseLearningConfig",
    "ModulatedLearningConfig",
    "STDPLearningConfig",
    "HebbianLearningConfig",
    "NeuralComponentConfig",
    "LearningComponentConfig",
    "PathwayConfig",
    # Brain
    "BrainConfig",
    "RegionSizes",
    "CortexType",
    "NeuromodulationConfig",
    # Training
    "TrainingConfig",
    # Region configs (canonical locations)
    "LayeredCortexConfig",
    "PredictiveCortexConfig",
    "RobustnessConfig",
    "HippocampusConfig",
    "StriatumConfig",
    "PrefrontalConfig",
    "CerebellumConfig",
    # Language
    "LanguageConfig",
    "EncodingConfig",
    "DecodingConfig",
    "PositionConfig",
    "SequenceMemoryConfig",
    # Size calculator
    "LayerSizeCalculator",
    "BiologicalRatios",
]


# Lazy imports for region configs to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import region configs to break circular dependencies."""
    _region_config_map = {
        "CerebellumConfig": ("thalia.regions.cerebellum", "CerebellumConfig"),
        "LayeredCortexConfig": ("thalia.regions.cortex.config", "LayeredCortexConfig"),
        "PredictiveCortexConfig": (
            "thalia.regions.cortex.predictive_cortex",
            "PredictiveCortexConfig",
        ),
        "RobustnessConfig": ("thalia.regions.cortex.robustness_config", "RobustnessConfig"),
        "HippocampusConfig": ("thalia.regions.hippocampus.config", "HippocampusConfig"),
        "PrefrontalConfig": ("thalia.regions.prefrontal", "PrefrontalConfig"),
        "StriatumConfig": ("thalia.regions.striatum.config", "StriatumConfig"),
    }

    if name in _region_config_map:
        module_name, attr_name = _region_config_map[name]
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
