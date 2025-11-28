"""
High-level cognitive functions: thinking, simulation, language.

The cognition module provides the highest-level abstractions in THALIA:
- ThinkingSNN: The main integrated "thinking machine"
- ThoughtState: Snapshot of cognitive state at each timestep
- ThinkingConfig: Configuration for the thinking architecture
- DaydreamNetwork: Spontaneous thought generation without input
- DaydreamIntegration: Add daydream capability to ThinkingSNN
"""

from .thinking import ThinkingSNN, ThinkingConfig, ThoughtState
from .daydream import (
    DaydreamNetwork,
    DaydreamConfig,
    DaydreamState,
    DaydreamMode,
    DaydreamIntegration,
)

__all__ = [
    "ThinkingSNN",
    "ThinkingConfig",
    "ThoughtState",
    "DaydreamNetwork",
    "DaydreamConfig",
    "DaydreamState",
    "DaydreamMode",
    "DaydreamIntegration",
]
