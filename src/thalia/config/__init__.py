"""
THALIA Unified Configuration System.

This module provides a centralized configuration system that:
1. Eliminates parameter duplication across modules
2. Provides clear inheritance of global parameters
3. Validates configuration consistency
4. Shows effective values being used

Usage:
======

    from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig

    # Create configuration with customizations
    config = ThaliaConfig(
        global_=GlobalConfig(device="cuda", vocab_size=10000),
        brain=BrainConfig(cortex_size=256, hippocampus_size=128),
    )

    # Show what values are actually being used
    config.summary()

    # Create brain with resolved config
    brain = DynamicBrain.from_config(config)

Author: Thalia Project
Date: December 2025
"""

# Re-export component configs from core/
from thalia.core.base.component_config import (
    NeuralComponentConfig,
    LearningComponentConfig,
    PathwayConfig,
)

# Re-export region configs from canonical locations
from thalia.regions.cerebellum import CerebellumConfig
from thalia.regions.cortex.config import LayeredCortexConfig
from thalia.regions.cortex.predictive_cortex import PredictiveCortexConfig
from thalia.regions.cortex.robustness_config import RobustnessConfig
from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.regions.prefrontal import PrefrontalConfig
from thalia.regions.striatum.config import StriatumConfig

from .base import BaseConfig
from .learning_config import (
    BaseLearningConfig,
    ModulatedLearningConfig,
    STDPLearningConfig,
    HebbianLearningConfig,
)
from .global_config import GlobalConfig
from .brain_config import (
    BrainConfig,
    RegionSizes,
    CortexType,
    NeuromodulationConfig,
)
from .training_config import TrainingConfig
from .language_config import (
    LanguageConfig,
    EncodingConfig,
    DecodingConfig,
    PositionConfig,
    SequenceMemoryConfig,
)
from .thalia_config import ThaliaConfig, print_config
from .validation import (
    validate_thalia_config,
    validate_brain_config,
    validate_global_consistency,
    validate_region_sizes,
    check_config_and_warn,
    ConfigValidationError,
    ValidatedConfig,
    ValidatorRegistry,
)
from .region_sizes import (
    # Utility functions (DEPRECATED - use LayerSizeCalculator instead)
    compute_hippocampus_sizes,
    compute_cortex_layer_sizes,
    compute_striatum_sizes,
    compute_thalamus_sizes,
    compute_multisensory_sizes,
    compute_cerebellum_sizes,
)
from .size_calculator import LayerSizeCalculator, BiologicalRatios

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
    # Global
    "GlobalConfig",
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
    # Size calculator (NEW - preferred)
    "LayerSizeCalculator",
    "BiologicalRatios",
    # DEPRECATED size functions (use LayerSizeCalculator instead)
    "compute_hippocampus_sizes",
    "compute_cortex_layer_sizes",
    "compute_striatum_sizes",
    "compute_thalamus_sizes",
    "compute_multisensory_sizes",
    "compute_cerebellum_sizes",
]
