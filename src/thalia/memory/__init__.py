"""
Working memory systems.

Provides reverberating circuit-based working memory with:
- Limited capacity (7±2 slots)
- Gated loading and clearing
- Decay without attention
- Pattern maintenance through recurrent activity
"""

from .working_memory import (
    WorkingMemoryConfig,
    MemorySlot,
    WorkingMemory,
    WorkingMemorySNN,
)

__all__ = [
    "WorkingMemoryConfig",
    "MemorySlot",
    "WorkingMemory",
    "WorkingMemorySNN",
]
