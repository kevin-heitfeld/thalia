"""
Text Data Pipeline for Training.

Provides utilities for loading, tokenizing, and batching text data
for training the spiking language model.

Features:
- Simple character-level or word-level tokenization
- Sequence windowing for context
- Batch generation
- Integration with standard text datasets

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import torch


@dataclass
class DataConfig:
    """Configuration for data pipeline.

    Attributes:
        tokenizer_type: "char" for character-level, "word" for word-level
        vocab_size: Maximum vocabulary size (for word-level)
        context_length: Length of context window
        batch_size: Number of sequences per batch

        min_frequency: Minimum word frequency to include in vocab
        lowercase: Whether to lowercase text

        shuffle: Whether to shuffle sequences
        seed: Random seed for reproducibility
    """

    tokenizer_type: str = "char"
    vocab_size: int = 10000
    context_length: int = 128
    batch_size: int = 32

    min_frequency: int = 2
    lowercase: bool = True

    shuffle: bool = True
    seed: int = 42


class SimpleTokenizer:
    """
    Simple tokenizer for character or word-level tokenization.

    Not as sophisticated as BPE but sufficient for demonstrations.
    """

    def __init__(
        self,
        tokenizer_type: str = "char",
        vocab_size: int = 10000,
        lowercase: bool = True,
        min_frequency: int = 2,
    ):
        self.tokenizer_type = tokenizer_type
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.min_frequency = min_frequency

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # Vocabulary (built during fit)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._is_fitted = False

    def fit(self, texts: List[str]) -> SimpleTokenizer:
        """
        Fit tokenizer on texts to build vocabulary.

        Args:
            texts: List of text strings

        Returns:
            Self for chaining
        """
        # Count token frequencies
        token_counts: Dict[str, int] = {}

        for text in texts:
            if self.lowercase:
                text = text.lower()

            if self.tokenizer_type == "char":
                tokens = list(text)
            else:
                tokens = text.split()

            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        # Filter by frequency and sort by count
        filtered_tokens = [
            (token, count) for token, count in token_counts.items() if count >= self.min_frequency
        ]
        filtered_tokens.sort(key=lambda x: x[1], reverse=True)

        # Build vocabulary
        self.token_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        for token, _ in filtered_tokens[: self.vocab_size - 4]:
            self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._is_fitted = True

        return self

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self._is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")

        if self.lowercase:
            text = text.lower()

        if self.tokenizer_type == "char":
            tokens = list(text)
        else:
            tokens = text.split()

        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self._is_fitted:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")

        tokens = [self.id_to_token.get(tid, self.unk_token) for tid in token_ids]

        if self.tokenizer_type == "char":
            return "".join(tokens)
        else:
            return " ".join(tokens)

    @property
    def vocab_size_actual(self) -> int:
        """Actual vocabulary size after fitting."""
        return len(self.token_to_id)

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer vocabulary to file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{token}\n")

    def load(self, path: Union[str, Path]) -> SimpleTokenizer:
        """Load tokenizer vocabulary from file."""
        path = Path(path)
        self.token_to_id = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    idx, token = int(parts[0]), parts[1]
                    self.token_to_id[token] = idx

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._is_fitted = True
        return self


class TextDataPipeline:
    """
    Data pipeline for loading and processing text data.

    Handles:
    - Text loading from files or strings
    - Tokenization
    - Sequence windowing
    - Batch generation

    Example:
        >>> config = DataConfig(context_length=64, batch_size=16)
        >>> pipeline = TextDataPipeline(config)
        >>>
        >>> # Load text
        >>> pipeline.load_text("Hello world this is a test...")
        >>>
        >>> # Generate batches
        >>> for batch in pipeline.get_batches():
        ...     # batch["input"]: (batch_size, context_length)
        ...     # batch["target"]: (batch_size, context_length)
        ...     train_step(batch)
    """

    def __init__(self, config: DataConfig):
        self.config = config

        self.tokenizer = SimpleTokenizer(
            tokenizer_type=config.tokenizer_type,
            vocab_size=config.vocab_size,
            lowercase=config.lowercase,
            min_frequency=config.min_frequency,
        )

        self.token_ids: List[int] = []
        self.sequences: List[Tuple[List[int], List[int]]] = []

        random.seed(config.seed)

    def load_text(self, text: str) -> TextDataPipeline:
        """
        Load text and prepare sequences.

        Args:
            text: Raw text string

        Returns:
            Self for chaining
        """
        # Fit tokenizer if not already
        if not self.tokenizer._is_fitted:
            self.tokenizer.fit([text])

        # Encode
        self.token_ids = self.tokenizer.encode(text)

        # Create sequences (input, target) pairs
        # Target is input shifted by 1 (next-token prediction)
        self._create_sequences()

        return self

    def load_file(self, path: Union[str, Path]) -> TextDataPipeline:
        """Load text from file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.load_text(text)

    def _create_sequences(self) -> None:
        """Create (input, target) sequence pairs."""
        self.sequences = []

        ctx_len = self.config.context_length

        # Slide window over tokens
        for i in range(0, len(self.token_ids) - ctx_len, ctx_len // 2):
            input_seq = self.token_ids[i : i + ctx_len]
            target_seq = self.token_ids[i + 1 : i + ctx_len + 1]

            if len(input_seq) == ctx_len and len(target_seq) == ctx_len:
                self.sequences.append((input_seq, target_seq))

    def get_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Generate batches of sequences.

        Yields:
            Dict with "input" and "target" tensors
        """
        if self.config.shuffle:
            random.shuffle(self.sequences)

        batch_size = self.config.batch_size

        for i in range(0, len(self.sequences), batch_size):
            batch_seqs = self.sequences[i : i + batch_size]

            if len(batch_seqs) < batch_size:
                continue  # Skip incomplete batch

            inputs = torch.tensor([s[0] for s in batch_seqs])
            targets = torch.tensor([s[1] for s in batch_seqs])

            yield {
                "input": inputs,
                "target": targets,
            }

    def get_all_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all sequences as tensors (for small datasets)."""
        if not self._sequences:
            from thalia.core.errors import ComponentError

            raise ComponentError(
                "SequenceGenerator", "No sequences loaded. Call load_text() first."
            )

        inputs = torch.tensor([s[0] for s in self.sequences])
        targets = torch.tensor([s[1] for s in self.sequences])

        return inputs, targets

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size_actual

    @property
    def n_sequences(self) -> int:
        """Number of sequences."""
        return len(self.sequences)

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        return torch.tensor(self.tokenizer.encode(text))
