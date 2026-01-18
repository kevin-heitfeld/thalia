"""
Language Configuration - Settings for language processing pipeline.

This module defines configuration for encoding, decoding, and
language-brain integration.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .brain_config import BrainConfig


class EncodingType(Enum):
    """Types of spike encoding strategies."""

    RATE = "rate"
    TEMPORAL = "temporal"
    PHASE = "phase"
    BURST = "burst"
    SDR = "sdr"


class DecodingType(Enum):
    """Types of spike decoding strategies."""

    RATE = "rate"
    TEMPORAL = "temporal"
    POPULATION = "population"
    ATTENTION = "attention"


@dataclass
class EncodingConfig:
    """Configuration for token-to-spike encoding.

    Note: vocab_size, device, and timing come from BrainConfig.
    """

    # SDR parameters
    sparsity: Optional[float] = None  # If None, use global default
    sdr_on_bits: Optional[int] = None  # If None, computed from sparsity
    sdr_overlap: float = 0.3

    # Encoding type
    encoding_type: EncodingType = EncodingType.SDR

    # Timesteps per token
    n_timesteps: int = 20

    # Learnable embedding
    learnable_embedding: bool = True

    # Embedding dimension (intermediate, before SDR projection)
    embedding_dim: int = 256

    def get_sparsity(self, brain_config: BrainConfig) -> float:
        """Get effective sparsity, falling back to global default."""
        return self.sparsity if self.sparsity is not None else brain_config.default_sparsity

    def get_sdr_on_bits(self, n_neurons: int, brain_config: BrainConfig) -> int:
        """Get number of active bits in SDR."""
        if self.sdr_on_bits is not None:
            return self.sdr_on_bits
        sparsity = self.get_sparsity(brain_config)
        return int(n_neurons * sparsity)


@dataclass
class DecodingConfig:
    """Configuration for spike-to-token decoding.

    Note: vocab_size and device come from BrainConfig.
    """

    # Decoding type
    decoding_type: DecodingType = DecodingType.POPULATION

    # Timesteps to accumulate
    n_timesteps: int = 20

    # Temperature for output softmax
    temperature: float = 1.0

    # Top-k and top-p for sampling
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@dataclass
class PositionConfig:
    """Configuration for position encoding."""

    # Max sequence length
    max_positions: int = 1024

    # Timesteps per position
    n_timesteps: int = 20

    # Position encoding size (relative to neuron count)
    size_ratio: float = 0.25  # 1/4 of content neurons


@dataclass
class SequenceMemoryConfig:
    """Configuration for hippocampus-based sequence memory."""

    # Context length
    context_length: int = 128

    # Memory parameters
    association_strength: float = 0.3
    retrieval_threshold: float = 0.2

    # Storage limits
    max_stored_contexts: int = 1000

    # Learning
    learning_rate: float = 0.1


@dataclass
class LanguageConfig:
    """Complete language processing configuration.

    Combines encoding, decoding, position, and memory settings.
    Global parameters come from BrainConfig.
    """

    # Sub-configurations
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    sequence_memory: SequenceMemoryConfig = field(default_factory=SequenceMemoryConfig)

    # Generation parameters
    max_new_tokens: int = 100
    stop_tokens: Optional[list] = None

    def summary(self) -> str:
        """Return formatted summary of language configuration."""
        lines = [
            "=== Language Configuration ===",
            "--- Encoding ---",
            f"  Type: {self.encoding.encoding_type.value}",
            f"  Timesteps: {self.encoding.n_timesteps}",
            f"  Embedding dim: {self.encoding.embedding_dim}",
            f"  Learnable: {self.encoding.learnable_embedding}",
            "",
            "--- Decoding ---",
            f"  Type: {self.decoding.decoding_type.value}",
            f"  Timesteps: {self.decoding.n_timesteps}",
            f"  Temperature: {self.decoding.temperature}",
            "",
            "--- Position ---",
            f"  Max positions: {self.position.max_positions}",
            f"  Size ratio: {self.position.size_ratio}",
            "",
            "--- Sequence Memory ---",
            f"  Context length: {self.sequence_memory.context_length}",
            f"  Max contexts: {self.sequence_memory.max_stored_contexts}",
        ]
        return "\n".join(lines)
