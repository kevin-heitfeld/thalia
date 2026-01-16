"""Core Protocols.

Protocol definitions for brain components and neural elements.
"""

from __future__ import annotations

from thalia.core.protocols.component import (
    BrainComponent,
    BrainComponentBase,
    BrainComponentMixin,
    LearnableComponent,
    RoutingComponent,
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
from thalia.core.protocols.checkpoint import (
    Checkpointable,
    CheckpointableWithNeuromorphic,
)

__all__ = [
    # Component Protocol
    "BrainComponent",
    "BrainComponentBase",
    "BrainComponentMixin",
    "LearnableComponent",
    "RoutingComponent",
    # Neural Protocols
    "Resettable",
    "Learnable",
    "Forwardable",
    "Diagnosable",
    "WeightContainer",
    "Configurable",
    "NeuralComponentProtocol",
    # Checkpoint Protocols
    "Checkpointable",
    "CheckpointableWithNeuromorphic",
]
