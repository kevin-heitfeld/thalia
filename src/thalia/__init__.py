"""
THALIA - Thinking Architecture via Learning Integrated Attractors

A framework for building genuinely thinking spiking neural networks.

Quick Start (External Users):
=============================

    from thalia import ThaliaConfig, Brain
    from thalia import Striatum, Hippocampus, LayeredCortex
    from thalia import ConductanceLIF, WeightInitializer
    
    # Create regions directly
    striatum = Striatum(StriatumConfig(n_input=256, n_output=64))
    
    # Or use full brain system
    config = ThaliaConfig(...)
    brain = Brain.from_thalia_config(config)

Internal Development:
====================

Internal code should use explicit imports for clarity:
    
    from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig
    from thalia.regions.striatum import Striatum, StriatumConfig
    from thalia.learning.bcm import BCMRule, BCMConfig
    
This helps with IDE navigation and makes dependencies explicit.
"""

__version__ = "0.2.0"

# ============================================================================
# PUBLIC API - For notebooks, experiments, and external users
# ============================================================================

# Configuration
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

# Brain System
from thalia.core.brain import EventDrivenBrain as Brain

# Brain Regions (most commonly used)
from thalia.regions import (
    BrainRegion,
    RegionConfig,
    Striatum,
    StriatumConfig,
    LayeredCortex,
    LayeredCortexConfig,
    PredictiveCortex,
    PredictiveCortexConfig,
    Cerebellum,
    CerebellumConfig,
    Prefrontal,
    PrefrontalConfig,
    TrisynapticHippocampus as Hippocampus,
    TrisynapticConfig as HippocampusConfig,
)

# Core Components (frequently needed)
from thalia.core import (
    # Neurons
    LIFNeuron,
    ConductanceLIF,
    # Weight Initialization
    WeightInitializer,
    InitStrategy,
    # Short-term plasticity
    ShortTermPlasticity,
    STPConfig,
    STPType,
    # Utilities
    ensure_batch_dim,
    clamp_weights,
)

# Learning Rules (common)
from thalia.learning import (
    BCMRule,
    STDPStrategy,
    HebbianStrategy,
    UnifiedHomeostasis,
)

# Pathways
from thalia.integration import SpikingPathway, SpikingPathwayConfig

# Namespaces for topic-level imports (advanced users)
from thalia import regions  # noqa: F401
from thalia import core  # noqa: F401  
from thalia import learning  # noqa: F401

__all__ = [
    # Version
    "__version__",
    # Configuration
    "ThaliaConfig",
    "GlobalConfig",
    "BrainConfig",
    "RegionSizes",
    # Brain System
    "Brain",
    # Brain Regions
    "BrainRegion",
    "RegionConfig",
    "Striatum",
    "StriatumConfig",
    "LayeredCortex",
    "LayeredCortexConfig",
    "PredictiveCortex",
    "PredictiveCortexConfig",
    "Cerebellum",
    "CerebellumConfig",
    "Prefrontal",
    "PrefrontalConfig",
    "Hippocampus",
    "HippocampusConfig",
    # Core Components
    "LIFNeuron",
    "ConductanceLIF",
    "WeightInitializer",
    "InitStrategy",
    "ShortTermPlasticity",
    "STPConfig",
    "STPType",
    "ensure_batch_dim",
    "clamp_weights",
    # Learning
    "BCMRule",
    "STDPStrategy",
    "HebbianStrategy",
    "UnifiedHomeostasis",
    # Pathways
    "SpikingPathway",
    "SpikingPathwayConfig",
    # Namespaces
    "regions",
    "core",
    "learning",
]
