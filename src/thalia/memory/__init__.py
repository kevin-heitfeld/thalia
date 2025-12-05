"""
Memory Module for THALIA.

This module provides higher-level memory abstractions built on the
hippocampal circuit for sequence and context memory.

Components:
- SequenceMemory: Token sequence storage and recall
- ContextBuffer: Working memory for recent context
- EpisodicStore: Long-term episodic memory storage

Configuration:
- Use ``thalia.config.SequenceMemoryConfig`` for new code
- Legacy ``SequenceMemoryConfig`` here is deprecated
"""

from thalia.memory.sequence import (
    SequenceMemory,
    SequenceContext,
)
from thalia.memory.context import (
    ContextBuffer,
    ContextBufferConfig,
)

# Re-export from canonical location for backwards compatibility
from thalia.config import SequenceMemoryConfig

__all__ = [
    "SequenceMemory",
    "SequenceMemoryConfig",
    "SequenceContext",
    "ContextBuffer",
    "ContextBufferConfig",
]
