"""
Language Interface Module for THALIA.

This module provides the bridge between text and spike domain,
enabling the SNN to process and generate natural language.

Components:
- SpikeEncoder: Token embeddings → Sparse spike patterns
- SpikeDecoder: Spike patterns → Token probabilities
- PositionEncoder: Sequence position via oscillatory phase
- LanguageBrain: Language-enabled brain (integrates with DynamicBrain)
"""

from __future__ import annotations

from thalia.language.decoder import (
    SpikeDecoder,
    SpikeDecoderConfig,
)
from thalia.language.encoder import (
    SparseDistributedRepresentation,
    SpikeEncoder,
    SpikeEncoderConfig,
)
from thalia.language.model import (
    LanguageBrainInterface,
    LanguageInterfaceConfig,
)
from thalia.language.position import (
    OscillatoryPositionEncoder,
    PositionEncoderConfig,
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
    # Model (Brain Integration)
    "LanguageBrainInterface",
    "LanguageInterfaceConfig",
]
