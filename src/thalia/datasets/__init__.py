"""
Datasets for Thalia training.

Provides task-specific datasets for each developmental stage:
- Phonological tasks (Stage 0)
- Visual object recognition (Stage 1+)
- Language tasks (Stage 2+)
- Social interaction (Stage 3+)
"""

from thalia.datasets.phonology import (
    PhonologicalDataset,
    PhonologicalConfig,
    PhonemeCategory,
    PhonemeFeatures,
    PHONEME_FEATURES,
)

__all__ = [
    "PhonologicalDataset",
    "PhonologicalConfig",
    "PhonemeCategory",
    "PhonemeFeatures",
    "PHONEME_FEATURES",
]
