"""
High-level cognitive functions: thinking, simulation, language.

The cognition module provides the highest-level abstractions in THALIA:
- ThinkingSNN: The main integrated "thinking machine"
- ThoughtState: Snapshot of cognitive state at each timestep
- ThinkingConfig: Configuration for the thinking architecture
"""

from .thinking import ThinkingSNN, ThinkingConfig, ThoughtState

__all__ = [
    "ThinkingSNN",
    "ThinkingConfig",
    "ThoughtState",
]
