"""
THALIA - Thinking Architecture via Learning Integrated Attractors

A framework for building genuinely thinking spiking neural networks.

Quick Start (External Users):
=============================

    from thalia import ThaliaConfig, DynamicBrain
    from thalia import Striatum, TrisynapticHippocampus, LayeredCortex
    from thalia import ConductanceLIF, WeightInitializer

    # Create regions directly
    striatum = Striatum(StriatumConfig(n_input=256, n_output=64))

    # Or use full brain system (DynamicBrain)
    config = ThaliaConfig(...)
    brain = BrainBuilder.preset("default", config)

Internal Development:
====================

Internal code should use explicit imports for clarity:

    # Neuron Models
    from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
    from thalia.regions.striatum import Striatum, StriatumConfig
    from thalia.learning.rules.bcm import BCMRule, BCMConfig

This helps with IDE navigation and makes dependencies explicit.
"""

from __future__ import annotations

__version__ = "0.2.0"

# ============================================================================
# PUBLIC API - For notebooks, experiments, and external users
# ============================================================================

# Core Components
from thalia.components.neurons.neuron import ConductanceLIF
from thalia.components.synapses.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.components.synapses.stp_presets import STP_PRESETS, STPPreset, get_stp_config
from thalia.components.synapses import InitStrategy, WeightInitializer

# Configuration
# Configuration
from thalia.config import (
    BrainConfig,
    CerebellumConfig,
    HippocampusConfig,
    LayeredCortexConfig,
    PredictiveCortexConfig,
    PrefrontalConfig,
    StriatumConfig,
    ThaliaConfig,
)

# Oscillators
from thalia.coordination.oscillator import (
    OSCILLATOR_DEFAULTS,
    OscillatorManager,
    SinusoidalOscillator,
)
from thalia.core.base.component_config import NeuralComponentConfig

# Diagnostics
from thalia.core.diagnostics_keys import DiagnosticKeys

# Brain System
from thalia.core.dynamic_brain import DynamicBrain

# Learning Rules
from thalia.learning import (
    BCMRule,
    HebbianStrategy,
    STDPStrategy,
    UnifiedHomeostasis,
)

# Neuromodulator Systems
from thalia.neuromodulation.homeostasis import (
    NeuromodulatorCoordination,
    NeuromodulatorHomeostasis,
    NeuromodulatorHomeostasisConfig,
    inverted_u_function,
)
from thalia.neuromodulation.systems.locus_coeruleus import (
    LocusCoeruleusConfig,
    LocusCoeruleusSystem,
)
from thalia.neuromodulation.systems.nucleus_basalis import (
    NucleusBasalisConfig,
    NucleusBasalisSystem,
)
from thalia.neuromodulation.systems.vta import (
    VTAConfig,
    VTADopamineSystem,
)

# Brain Regions
from thalia.regions import (
    Cerebellum,
    LayeredCortex,
    PredictiveCortex,
    Prefrontal,
    Striatum,
    TrisynapticHippocampus,
)

# Type Aliases (for better type hints)
from thalia.typing import (  # DiagnosticsDict removed - use specific TypedDict subclasses instead
    BatchData,
    CheckpointMetadata,
    ComponentGraph,
    ConnectionGraph,
    InputSizes,
    LearningStrategies,
    NeuromodulatorLevels,
    SourceOutputs,
    SourcePort,
    SourceSpec,
    StateDict,
    SynapticWeights,
    TargetPort,
    TopologyGraph,
)
from thalia.utils.core_utils import clamp_weights

# Visualization (optional - requires manim)
# try:
#     from thalia.visualization import BrainActivityVisualization, MANIM_AVAILABLE
# except ImportError:
#     BrainActivityVisualization = None
#     MANIM_AVAILABLE = False
MANIM_AVAILABLE = False

# Namespaces for topic-level imports
from thalia import core  # noqa: E402, F401
from thalia import learning  # noqa: E402, F401
from thalia import regions  # noqa: E402, F401

__all__ = [
    # Version
    "__version__",
    # Configuration
    "ThaliaConfig",
    "BrainConfig",
    # Type Aliases
    "ComponentGraph",
    "ConnectionGraph",
    "TopologyGraph",
    "SourceSpec",
    "SourcePort",
    "TargetPort",
    "SourceOutputs",
    "InputSizes",
    "SynapticWeights",
    "LearningStrategies",
    "StateDict",
    "CheckpointMetadata",
    # "DiagnosticsDict",  # Removed - use specific TypedDict subclasses
    "NeuromodulatorLevels",
    "BatchData",
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
    "TrisynapticHippocampus",
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
    # Oscillators
    "SinusoidalOscillator",
    "OscillatorManager",
    "OSCILLATOR_DEFAULTS",
    # Neuromodulator Systems
    "NeuromodulatorHomeostasis",
    "NeuromodulatorHomeostasisConfig",
    "NeuromodulatorCoordination",
    "inverted_u_function",
    "VTADopamineSystem",
    "VTAConfig",
    "LocusCoeruleusSystem",
    "LocusCoeruleusConfig",
    "NucleusBasalisSystem",
    "NucleusBasalisConfig",
    # Visualization
    "MANIM_AVAILABLE",
    # Namespaces
    "regions",
    "core",
    "learning",
]
