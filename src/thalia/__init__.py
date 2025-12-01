"""
THALIA - Thinking Architecture via Learning Integrated Attractors

A framework for building genuinely thinking spiking neural networks.
"""

__version__ = "0.1.0"

from thalia.core import LIFNeuron, SNNLayer

# Diagnostic tools
from thalia.diagnostics import (
    DiagnosticConfig,
    DiagnosticLevel,
    MechanismConfig,
    ExperimentDiagnostics,
)

# Brain region modules (biologically-specialized learning)
from thalia import regions
