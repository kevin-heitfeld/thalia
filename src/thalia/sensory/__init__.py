"""
Sensory Module - Multimodal input encoding for the spiking brain.

This module provides sensory pathways that convert raw inputs
(images, audio, text, etc.) into spike patterns for brain processing.

Key Concept:
===========
The brain doesn't care about input modality - once information is
converted to spikes, all processing is uniform. The sensory pathways
handle the modality-specific encoding.

Architecture:
============

    Vision (Image)      Audio (Waveform)     Text (Tokens)
         │                    │                   │
         ▼                    ▼                   ▼
    ┌─────────┐         ┌──────────┐       ┌──────────┐
    │ Retinal │         │ Cochlear │       │   SDR    │
    │ Encoder │         │ Encoder  │       │ Encoder  │
    └────┬────┘         └────┬─────┘       └────┬─────┘
         │                   │                  │
         └───────────────────┼──────────────────┘
                             │
                             ▼
                    Spike Patterns
                    [batch, time, neurons]
                             │
                             ▼
                    ┌────────────────┐
                    │ EventDrivenBrain │
                    │  (Unified SNN)   │
                    └────────────────┘

Usage:
======

    from thalia.sensory import (
        create_visual_pathway,
        create_auditory_pathway,
        create_language_pathway,
        create_multimodal_pathway,
    )

    # Create pathways
    vision = create_visual_pathway(output_size=256)
    audio = create_auditory_pathway(output_size=256)
    language = create_language_pathway(output_size=256)

    # Encode inputs to spikes using forward() (ADR-007)
    # Callable syntax (preferred):
    image_spikes, _ = vision(image_tensor)
    audio_spikes, _ = audio(audio_tensor)
    text_spikes, _ = language(token_ids)

    # Feed to brain
    brain.process_sample(image_spikes[:, 0, :])  # One timestep at a time

Author: Thalia Project
Date: December 2025
"""

from thalia.sensory.pathways import (
    # Base classes
    Modality,
    SensoryPathwayConfig,
    SensoryPathway,

    # Visual pathway
    VisualConfig,
    RetinalEncoder,
    VisualPathway,

    # Auditory pathway
    AuditoryConfig,
    CochlearEncoder,
    AuditoryPathway,

    # Language pathway
    LanguageConfig,
    LanguagePathway,

    # Multimodal integration
    MultimodalPathway,

    # Factory functions
    create_visual_pathway,
    create_auditory_pathway,
    create_language_pathway,
    create_multimodal_pathway,
)

__all__ = [
    # Enums and base
    "Modality",
    "SensoryPathwayConfig",
    "SensoryPathway",

    # Visual
    "VisualConfig",
    "RetinalEncoder",
    "VisualPathway",

    # Auditory
    "AuditoryConfig",
    "CochlearEncoder",
    "AuditoryPathway",

    # Language
    "LanguageConfig",
    "LanguagePathway",

    # Multimodal
    "MultimodalPathway",

    # Factories
    "create_visual_pathway",
    "create_auditory_pathway",
    "create_language_pathway",
    "create_multimodal_pathway",
]
