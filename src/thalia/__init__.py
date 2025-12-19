"""
THALIA - Thinking Architecture via Learning Integrated Attractors

A framework for building genuinely thinking spiking neural networks.

Quick Start (External Users):
=============================

    from thalia import ThaliaConfig, DynamicBrain
    from thalia import Striatum, Hippocampus, LayeredCortex
    from thalia import ConductanceLIF, WeightInitializer

    # Create regions directly
    striatum = Striatum(StriatumConfig(n_input=256, n_output=64))

    # Or use full brain system (DynamicBrain)
    config = ThaliaConfig(...)
    brain = DynamicBrain.from_thalia_config(config)

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

# Brain System (DynamicBrain is now the primary API)
from thalia.core.dynamic_brain import DynamicBrain

from thalia.core.base.component_config import NeuralComponentConfig

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
from thalia.components.neurons.neuron import ConductanceLIF
from thalia.components.synapses.weight_init import WeightInitializer, InitStrategy
from thalia.components.synapses.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.components.synapses.stp_presets import STP_PRESETS, STPPreset, get_stp_config
from thalia.utils.core_utils import clamp_weights

# Diagnostics
from thalia.core.diagnostics_keys import DiagnosticKeys

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

# Pathways (SpikingPathway deprecated - use AxonalProjection)
import warnings
from thalia.pathways.spiking_pathway import SpikingPathway

# Warn users on first import
warnings.warn(
    "SpikingPathway is deprecated and will be removed in v3.0. "
    "All regions now use NeuralRegion base class with synaptic_weights at dendrites. "
    "Use AxonalProjection for connections between regions.",
    DeprecationWarning,
    stacklevel=2
)

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
    "DynamicBrain",
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
    # Diagnostics
    "DiagnosticKeys",
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
