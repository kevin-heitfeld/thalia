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

    # Neuron Models
    from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
    from thalia.regions.striatum import Striatum, StriatumConfig
    from thalia.learning.rules.bcm import BCMRule, BCMConfig

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

from thalia.core.component_config import NeuralComponentConfig

# Brain Regions (most commonly used)
from thalia.regions import (
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
    Hippocampus,
    HippocampusConfig,
)

# Core Components (frequently needed)
from thalia.core import (
    # Neurons
    ConductanceLIF,
    # Weight Initialization
    WeightInitializer,
    InitStrategy,
    # Short-term plasticity
    ShortTermPlasticity,
    STPConfig,
    STPType,
    # STP Presets (new in Tier 2)
    STP_PRESETS,
    STPPreset,
    get_stp_config,
    # Utilities
    clamp_weights,
)

# Oscillators (for working memory, consolidation)
from thalia.coordination.oscillator import (
    SinusoidalOscillator,
    OscillatorManager,
    OSCILLATOR_DEFAULTS,
)

# Neuromodulator Systems (centralized)
from thalia.neuromodulation.systems.vta import (
    VTADopamineSystem,
    VTAConfig,
)
from thalia.neuromodulation.systems.locus_coeruleus import (
    LocusCoeruleusSystem,
    LocusCoeruleusConfig,
)
from thalia.neuromodulation.systems.nucleus_basalis import (
    NucleusBasalisSystem,
    NucleusBasalisConfig,
)
from thalia.learning.homeostasis.homeostatic_regulation import (
    HomeostaticRegulator,
    HomeostaticConfig,
    NeuromodulatorCoordination,
    inverted_u_function,
)

# Learning Rules (common)
from thalia.learning import (
    BCMRule,
    STDPStrategy,
    HebbianStrategy,
    UnifiedHomeostasis,
)

# Pathways
from thalia.integration import SpikingPathway

# Visualization (optional - requires manim)
# try:
#     from thalia.visualization import BrainActivityVisualization, MANIM_AVAILABLE
# except ImportError:
#     BrainActivityVisualization = None
#     MANIM_AVAILABLE = False
MANIM_AVAILABLE = False

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
    "NeuralComponentConfig",
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
    "ConductanceLIF",
    "WeightInitializer",
    "InitStrategy",
    "ShortTermPlasticity",
    "STPConfig",
    "STPType",
    "STP_PRESETS",
    "STPPreset",
    "get_stp_config",
    "clamp_weights",
    # Learning
    "BCMRule",
    "STDPStrategy",
    "HebbianStrategy",
    "UnifiedHomeostasis",
    # Pathways
    "SpikingPathway",
    # Oscillators
    "SinusoidalOscillator",
    "OscillatorManager",
    "OSCILLATOR_DEFAULTS",
    # Neuromodulator Systems
    "VTADopamineSystem",
    "VTAConfig",
    "LocusCoeruleusSystem",
    "LocusCoeruleusConfig",
    "NucleusBasalisSystem",
    "NucleusBasalisConfig",
    "HomeostaticRegulator",
    "HomeostaticConfig",
    "NeuromodulatorCoordination",
    "inverted_u_function",
    # Visualization
    "MANIM_AVAILABLE",
    # Namespaces
    "regions",
    "core",
    "learning",
]
