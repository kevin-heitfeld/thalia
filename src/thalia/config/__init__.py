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

from .base import BaseConfig
from .brain_config import (
    BrainConfig,
    CortexType,
    NeuromodulationConfig,
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
    # Main configs
    "ThaliaConfig",
    "print_config",
    # Validation configs
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
    # Brain configs
    "BrainConfig",
    "CortexType",
    "NeuromodulationConfig",
    # Training configs
    "TrainingConfig",
    # Region configs
    "LayeredCortexConfig",
    "PredictiveCortexConfig",
    "HippocampusConfig",
    "StriatumConfig",
    "PrefrontalConfig",
    "CerebellumConfig",
    "ThalamicRelayConfig",
    "MultimodalIntegrationConfig",
    # Language configs
    "LanguageConfig",
    "EncodingConfig",
    "DecodingConfig",
    "PositionConfig",
    "SequenceMemoryConfig",
    # Size calculator
    "LayerSizeCalculator",
    "BiologicalRatios",
]
