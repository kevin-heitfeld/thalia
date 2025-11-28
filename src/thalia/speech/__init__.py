"""
THALIA Inner Speech Module.

This module implements self-dialogue and verbal reasoning capabilities,
allowing the network to generate internal verbal sequences for reasoning,
planning, and metacognition.

The inner speech system consists of:
- TokenVocabulary: Token storage with embeddings
- InnerVoice: Generates verbal sequences through spiking activity
- DialogueManager: Multi-voice system for internal debate
- ReasoningChain: Multi-step verbal reasoning
- InnerSpeechNetwork: Complete integrated system
"""

from thalia.speech.inner_speech import (
    Token,
    VoiceType,
    TokenVocabulary,
    InnerVoice,
    Utterance,
    ReasoningStep,
    ReasoningChain,
    DialogueManager,
    InnerSpeechConfig,
    InnerSpeechNetwork,
)

__all__ = [
    "Token",
    "VoiceType",
    "TokenVocabulary",
    "InnerVoice",
    "Utterance",
    "ReasoningStep",
    "ReasoningChain",
    "DialogueManager",
    "InnerSpeechConfig",
    "InnerSpeechNetwork",
]
