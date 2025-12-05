"""
Language Interface Module for THALIA.

This module provides the bridge between text and spike domain,
enabling the SNN to process and generate natural language.

Components:
- SpikeEncoder: Token embeddings → Sparse spike patterns
- SpikeDecoder: Spike patterns → Token probabilities
- PositionEncoder: Sequence position via oscillatory phase
- SpikingLanguageModel: Full language model wrapper
"""

from thalia.language.encoder import (
    SpikeEncoder,
    SpikeEncoderConfig,
    SparseDistributedRepresentation,
)
from thalia.language.decoder import (
    SpikeDecoder,
    SpikeDecoderConfig,
)
from thalia.language.position import (
    OscillatoryPositionEncoder,
    PositionEncoderConfig,
)
from thalia.language.model import (
    SpikingLanguageModel,
    SpikingLanguageModelConfig,
)

__all__ = [
    # Encoder
    "SpikeEncoder",
    "SpikeEncoderConfig",
    "SparseDistributedRepresentation",
    # Decoder
    "SpikeDecoder",
    "SpikeDecoderConfig",
    # Position
    "OscillatoryPositionEncoder",
    "PositionEncoderConfig",
    # Model
    "SpikingLanguageModel",
    "SpikingLanguageModelConfig",
]
