"""
Temporal Sequence Dataset for Stage 0 (Phonology)

Generates sequential patterns for testing temporal prediction:
- A-B-C patterns (predictable sequences)
- A-B-A patterns (repetition with interleaving)
- A-A-B patterns (repetition followed by change)
- Violations of learned patterns (for prediction error)

Biologically relevant:
- Tests hippocampal sequence learning
- Engages theta oscillations for temporal binding
- Critical for speech sound sequence learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class PatternType(Enum):
    """Types of sequential patterns."""

    ABC = "abc"  # Linear sequence A→B→C
    ABA = "aba"  # Repetition with gap A→B→A
    AAB = "aab"  # Immediate repetition A→A→B
    ABAC = "abac"  # Hierarchical A→B→A→C
    RANDOM = "random"  # No structure (control)


@dataclass
class SequenceConfig:
    """Configuration for temporal sequence generation."""

    n_symbols: int = 5  # Number of distinct symbols
    sequence_length: int = 10  # Length of each sequence
    pattern_types: List[PatternType] = field(
        default_factory=lambda: [PatternType.ABC, PatternType.ABA, PatternType.AAB]
    )
    violation_probability: float = 0.1  # Probability of pattern violation
    encoding: str = "one_hot"  # "one_hot" or "distributed"
    device: torch.device = torch.device("cpu")


class TemporalSequenceDataset:
    """
    Temporal sequence dataset for Stage 0 (Phonology).

    Generates sequences with learnable patterns for testing:
    - Temporal prediction (hippocampus)
    - Pattern completion (cortex)
    - Violation detection (prediction error)
    """

    def __init__(self, config: SequenceConfig):
        self.config = config
        self.pattern_generators = {
            PatternType.ABC: self._generate_abc,
            PatternType.ABA: self._generate_aba,
            PatternType.AAB: self._generate_aab,
            PatternType.ABAC: self._generate_abac,
            PatternType.RANDOM: self._generate_random,
        }

    def generate_sequence(
        self,
        pattern_type: Optional[PatternType] = None,
        include_violation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, PatternType]:
        """
        Generate a single sequence.

        Args:
            pattern_type: Specific pattern to generate (random if None)
            include_violation: Whether to include pattern violation

        Returns:
            sequence: Input sequence (length, n_symbols) if one-hot
            targets: Expected next symbols (length, n_symbols)
            pattern_type: Type of pattern generated
        """
        if pattern_type is None:
            pattern_array = np.array(self.config.pattern_types, dtype=object)
            pattern_type = np.random.choice(pattern_array)

        # Generate base pattern
        symbol_sequence = self.pattern_generators[pattern_type]()

        # Maybe add violation
        if include_violation or (np.random.random() < self.config.violation_probability):
            symbol_sequence = self._add_violation(symbol_sequence)

        # Encode sequence
        sequence = self._encode_sequence(symbol_sequence)

        # Generate targets (next symbol prediction)
        targets = self._generate_targets(symbol_sequence)

        return sequence, targets, pattern_type

    def _generate_abc(self) -> List[int]:
        """Generate A→B→C linear sequence pattern."""
        sequence = []
        n_triplets = self.config.sequence_length // 3

        for _ in range(n_triplets):
            a, b, c = np.random.choice(self.config.n_symbols, size=3, replace=False)
            sequence.extend([a, b, c])

        # Pad to exact length
        while len(sequence) < self.config.sequence_length:
            sequence.append(np.random.randint(0, self.config.n_symbols))

        return sequence[: self.config.sequence_length]

    def _generate_aba(self) -> List[int]:
        """Generate A→B→A repetition pattern."""
        sequence = []
        n_triplets = self.config.sequence_length // 3

        for _ in range(n_triplets):
            a, b = np.random.choice(self.config.n_symbols, size=2, replace=False)
            sequence.extend([a, b, a])

        while len(sequence) < self.config.sequence_length:
            sequence.append(np.random.randint(0, self.config.n_symbols))

        return sequence[: self.config.sequence_length]

    def _generate_aab(self) -> List[int]:
        """Generate A→A→B immediate repetition pattern."""
        sequence = []
        n_triplets = self.config.sequence_length // 3

        for _ in range(n_triplets):
            a, b = np.random.choice(self.config.n_symbols, size=2, replace=False)
            sequence.extend([a, a, b])

        while len(sequence) < self.config.sequence_length:
            sequence.append(np.random.randint(0, self.config.n_symbols))

        return sequence[: self.config.sequence_length]

    def _generate_abac(self) -> List[int]:
        """Generate A→B→A→C hierarchical pattern."""
        sequence = []
        n_quads = self.config.sequence_length // 4

        for _ in range(n_quads):
            a, b, c = np.random.choice(self.config.n_symbols, size=3, replace=False)
            sequence.extend([a, b, a, c])

        while len(sequence) < self.config.sequence_length:
            sequence.append(np.random.randint(0, self.config.n_symbols))

        return sequence[: self.config.sequence_length]

    def _generate_random(self) -> List[int]:
        """Generate random sequence (no structure)."""
        return [
            np.random.randint(0, self.config.n_symbols) for _ in range(self.config.sequence_length)
        ]

    def _add_violation(self, sequence: List[int]) -> List[int]:
        """Add pattern violation at random position."""
        violation_pos = np.random.randint(1, len(sequence) - 1)
        sequence = sequence.copy()

        # Replace with unexpected symbol
        possible_symbols = list(range(self.config.n_symbols))
        possible_symbols.remove(sequence[violation_pos])
        sequence[violation_pos] = np.random.choice(possible_symbols)

        return sequence

    def _encode_sequence(self, symbol_sequence: List[int]) -> torch.Tensor:
        """
        Encode symbol sequence as tensor.

        Args:
            symbol_sequence: List of symbol indices

        Returns:
            encoded: (length, n_symbols) for one-hot
                    or (length, embedding_dim) for distributed
        """
        if self.config.encoding == "one_hot":
            encoded = torch.zeros(
                len(symbol_sequence),
                self.config.n_symbols,
                device=self.config.device,
            )
            for t, symbol in enumerate(symbol_sequence):
                encoded[t, symbol] = 1.0
        elif self.config.encoding == "distributed":
            # Random distributed representations
            encoded = torch.randn(
                len(symbol_sequence),
                self.config.n_symbols * 2,  # 2x dimensions for distributed
                device=self.config.device,
            )
            # Make each symbol have consistent representation
            for t, symbol in enumerate(symbol_sequence):
                torch.manual_seed(symbol)  # Consistent per symbol
                encoded[t] = torch.randn(self.config.n_symbols * 2, device=self.config.device)
        else:
            raise ValueError(f"Unknown encoding: {self.config.encoding}")

        return encoded

    def _generate_targets(self, symbol_sequence: List[int]) -> torch.Tensor:
        """
        Generate target sequence (next symbol prediction).

        Args:
            symbol_sequence: List of symbol indices

        Returns:
            targets: (length, n_symbols) one-hot encoded
        """
        targets = torch.zeros(
            len(symbol_sequence),
            self.config.n_symbols,
            device=self.config.device,
        )

        for t in range(len(symbol_sequence) - 1):
            next_symbol = symbol_sequence[t + 1]
            targets[t, next_symbol] = 1.0

        # Last target is uniform (no next symbol)
        targets[-1] = 1.0 / self.config.n_symbols

        return targets

    def compute_prediction_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Compute prediction error (cross-entropy).

        Args:
            predictions: (length, n_symbols) predicted probabilities
            targets: (length, n_symbols) target one-hot

        Returns:
            error: Mean cross-entropy
        """
        # Add small epsilon to avoid log(0)
        predictions = torch.clamp(predictions, min=1e-7, max=1.0)

        # Cross-entropy: -sum(target * log(pred))
        ce = -torch.sum(targets * torch.log(predictions), dim=-1)

        return float(ce.mean().item())

    def analyze_pattern_learning(
        self,
        brain,
        n_test_sequences: int = 100,
    ) -> Dict[str, float]:
        """
        Analyze how well brain learned each pattern type.

        Args:
            brain: Brain instance to test
            n_test_sequences: Number of sequences per pattern

        Returns:
            results: Dict mapping pattern_type → prediction_accuracy
        """
        results = {}

        for pattern_type in self.config.pattern_types:
            correct = 0
            total = 0

            for _ in range(n_test_sequences):
                seq, targets, _ = self.generate_sequence(
                    pattern_type=pattern_type,
                    include_violation=False,
                )

                # Run brain on sequence
                for t in range(len(seq) - 1):
                    brain_output = brain.forward(seq[t : t + 1])  # Single timestep
                    predicted_symbol = torch.argmax(brain_output)
                    target_symbol = torch.argmax(targets[t])

                    if predicted_symbol == target_symbol:
                        correct += 1
                    total += 1

            results[pattern_type.value] = correct / total if total > 0 else 0.0

        return results


def create_stage0_temporal_dataset(
    device: torch.device = torch.device("cpu"),
) -> TemporalSequenceDataset:
    """
    Create temporal sequence dataset for Stage 0.

    Args:
        device: Device to place tensors on

    Returns:
        dataset: TemporalSequenceDataset instance
    """
    config = SequenceConfig(
        n_symbols=5,
        sequence_length=10,
        pattern_types=[PatternType.ABC, PatternType.ABA, PatternType.AAB],
        violation_probability=0.1,
        encoding="one_hot",
        device=device,
    )
    return TemporalSequenceDataset(config)
