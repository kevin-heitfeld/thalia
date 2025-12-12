"""Core Protocols.

Protocol definitions for brain components and neural elements.
"""

from thalia.core.protocols.component import (
    BrainComponent,
    BrainComponentBase,
    BrainComponentMixin,
)
from thalia.core.protocols.neural import (
    Resettable,
    Learnable,
    Forwardable,
    Diagnosable,
    WeightContainer,
    Configurable,
    NeuralComponentProtocol,
)

__all__ = [
    # Component Protocol
    "BrainComponent",
    "BrainComponentBase",
    "BrainComponentMixin",
    # Neural Protocols
    "Resettable",
    "Learnable",
    "Forwardable",
    "Diagnosable",
    "WeightContainer",
    "Configurable",
    "NeuralComponentProtocol",
]
