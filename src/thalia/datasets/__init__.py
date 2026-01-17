"""
Datasets for Thalia training.

Provides task-specific datasets for each developmental stage:
- Stage 0: Phonological tasks (English, German, Spanish)
- Stage 0: Temporal sequence prediction (A-B-C patterns)
- Stage 1: CIFAR-10 with spike encoding
- Stage 2: Grammar learning (subject-verb agreement, word order)
- Stage 3: Reading comprehension (phoneme→word, simple QA)
"""

from __future__ import annotations

from thalia.datasets.cifar_wrapper import (
    CIFARConfig,
    CIFARForThalia,
    create_stage1_cifar_datasets,
)
from thalia.datasets.grammar import (
    AgreementType,
    GrammarConfig,
    GrammarDataset,
    GrammarRule,
    GrammarVocabulary,
)
from thalia.datasets.grammar import Language as GrammarLanguage
from thalia.datasets.grammar import (
    create_stage2_grammar_dataset,
)
from thalia.datasets.phonology import (
    LANGUAGE_CONTRASTS,
    LANGUAGE_PHONEMES,
    PHONEME_FEATURES,
    Language,
    PhonemeCategory,
    PhonemeFeatures,
    PhonologicalConfig,
    PhonologicalDataset,
)
from thalia.datasets.reading import Language as ReadingLanguage
from thalia.datasets.reading import (
    ReadingConfig,
    ReadingDataset,
    ReadingTask,
    ReadingVocabulary,
    create_stage3_reading_dataset,
)
from thalia.datasets.temporal_sequences import (
    PatternType,
    SequenceConfig,
    TemporalSequenceDataset,
    create_stage0_temporal_dataset,
)

__all__ = [
    # Phonology (Stage 0)
    "PhonologicalDataset",
    "PhonologicalConfig",
    "PhonemeCategory",
    "PhonemeFeatures",
    "PHONEME_FEATURES",
    "Language",
    "LANGUAGE_PHONEMES",
    "LANGUAGE_CONTRASTS",
    # Temporal sequences (Stage 0)
    "TemporalSequenceDataset",
    "SequenceConfig",
    "PatternType",
    "create_stage0_temporal_dataset",
    # CIFAR-10 (Stage 1)
    "CIFARForThalia",
    "CIFARConfig",
    "create_stage1_cifar_datasets",
    # Grammar (Stage 2)
    "GrammarDataset",
    "GrammarConfig",
    "GrammarVocabulary",
    "GrammarRule",
    "AgreementType",
    "GrammarLanguage",
    "create_stage2_grammar_dataset",
    # Reading (Stage 3)
    "ReadingDataset",
    "ReadingConfig",
    "ReadingVocabulary",
    "ReadingTask",
    "ReadingLanguage",
    "create_stage3_reading_dataset",
]
