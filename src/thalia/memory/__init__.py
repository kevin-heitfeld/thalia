"""
Memory Module for THALIA.

This module provides higher-level memory abstractions built on the
hippocampal circuit for sequence and context memory.

Components:
- SequenceMemory: Token sequence storage and recall
- ContextBuffer: Working memory for recent context
- EpisodicStore: Long-term episodic memory storage
"""

from thalia.memory.sequence import (
    SequenceMemory,
    SequenceMemoryConfig,
    SequenceContext,
)
from thalia.memory.context import (
    ContextBuffer,
    ContextBufferConfig,
)

__all__ = [
    "SequenceMemory",
    "SequenceMemoryConfig",
    "SequenceContext",
    "ContextBuffer",
    "ContextBufferConfig",
]
