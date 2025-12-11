"""
Context Buffer - Working Memory for Recent Context.

This module provides a working memory buffer that maintains recent
context for language processing. It's inspired by the prefrontal
cortex's working memory function.

Architecture:
=============

    Incoming Tokens
           │
           ▼
    ┌─────────────────┐
    │  Context Buffer │  Fixed-size sliding window
    │  [T₁ T₂ T₃ T₄]  │  Oldest tokens evicted
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Attention Mask │  Which tokens to attend to
    │  [1.0 0.8 0.6 1.0]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Compressed Rep │  Summary for memory
    └─────────────────┘

Features:
- Fixed-size sliding window (like attention context)
- Recency weighting (recent tokens more prominent)
- Compression for long sequences
- Integration with hippocampal memory for overflow

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections import deque

import torch
import torch.nn as nn


@dataclass
class ContextBufferConfig:
    """Configuration for context buffer.

    Attributes:
        max_length: Maximum number of tokens in buffer
        n_neurons: Size of neural representations

        recency_decay: How much to decay older items (0-1)
        compression_ratio: How much to compress when buffer full

        device: Computation device
    """
    max_length: int = 512
    n_neurons: int = 256

    recency_decay: float = 0.95  # Multiply by this each position back
    compression_ratio: float = 0.5  # Compress to this fraction when full

    device: str = "cpu"


class ContextBuffer(nn.Module):
    """
    Working memory buffer for maintaining recent context.

    This is a simple but effective mechanism for keeping track of
    recent tokens in a sequence, with:
    1. Fixed-size sliding window
    2. Recency-weighted attention
    3. Compression for memory efficiency

    Example:
        >>> buffer = ContextBuffer(ContextBufferConfig(max_length=128))
        >>>
        >>> # Add tokens one by one
        >>> for token in tokens:
        ...     buffer.push(token)
        >>>
        >>> # Get current context
        >>> context = buffer.get_context()
        >>>
        >>> # Get weighted representation
        >>> weighted = buffer.get_weighted_representation()
    """

    def __init__(self, config: ContextBufferConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Token buffer (stores token IDs)
        self.token_buffer: deque[int] = deque(maxlen=config.max_length)

        # Activation buffer (stores neural representations)
        self.activation_buffer: deque[torch.Tensor] = deque(maxlen=config.max_length)

        # Compression layer for when buffer overflows
        self.compressor = nn.Linear(
            config.n_neurons * 2,  # Compress pairs
            config.n_neurons,
        )

        # Attention weights for recency (precomputed)
        self.register_buffer(
            "recency_weights",
            self._compute_recency_weights(),
        )

        self.to(self.device)

    def _compute_recency_weights(self) -> torch.Tensor:
        """Compute recency weights for attention."""
        weights = torch.zeros(self.config.max_length)
        for i in range(self.config.max_length):
            # More recent = higher weight
            weights[-(i+1)] = self.config.recency_decay ** i
        return weights

    def push(
        self,
        token_id: int,
        activation: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Push a token onto the buffer.

        Args:
            token_id: Token ID to push
            activation: Optional neural activation for this token
        """
        self.token_buffer.append(token_id)

        if activation is not None:
            self.activation_buffer.append(activation.to(self.device))
        else:
            # Create placeholder
            self.activation_buffer.append(
                torch.zeros(self.config.n_neurons, device=self.device)
            )

    def push_sequence(
        self,
        token_ids: torch.Tensor,
        activations: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Push a sequence of tokens.

        Args:
            token_ids: Token IDs, shape (seq_len,) or (1, seq_len)
            activations: Optional activations, shape (seq_len, n_neurons)
        """
        if token_ids.dim() == 2:
            token_ids = token_ids.squeeze(0)

        for i, token_id in enumerate(token_ids.tolist()):
            act = activations[i] if activations is not None else None
            self.push(token_id, act)

    def get_context(self) -> torch.Tensor:
        """Get current context as tensor of token IDs."""
        return torch.tensor(list(self.token_buffer), device=self.device)

    def get_activations(self) -> torch.Tensor:
        """Get stacked activations for all tokens in buffer."""
        if not self.activation_buffer:
            return torch.zeros(0, self.config.n_neurons, device=self.device)

        return torch.stack(list(self.activation_buffer))

    def get_weighted_representation(self) -> torch.Tensor:
        """
        Get recency-weighted average of activations.

        More recent tokens contribute more to the representation.
        """
        activations = self.get_activations()

        if len(activations) == 0:
            return torch.zeros(self.config.n_neurons, device=self.device)

        # Get relevant weights (may be shorter than max_length)
        n_items = len(activations)
        weights = self.recency_weights[-n_items:]
        weights = weights / weights.sum()  # Normalize

        # Weighted sum
        weighted = (activations * weights.unsqueeze(-1)).sum(dim=0)

        return weighted

    def get_attention_mask(self) -> torch.Tensor:
        """
        Get attention mask based on recency.

        Returns mask with higher values for more recent tokens.
        """
        n_items = len(self.token_buffer)
        if n_items == 0:
            return torch.zeros(0, device=self.device)

        return self.recency_weights[-n_items:].clone()

    def compress(self) -> torch.Tensor:
        """
        Compress buffer contents to smaller representation.

        This is called when buffer is full and we need to make room.
        Returns compressed representation of older items.
        """
        activations = self.get_activations()

        if len(activations) < 2:
            return activations.mean(dim=0) if len(activations) > 0 else torch.zeros(
                self.config.n_neurons, device=self.device
            )

        # Compress pairs of activations
        compressed_parts = []
        for i in range(0, len(activations) - 1, 2):
            pair = torch.cat([activations[i], activations[i + 1]])
            compressed = self.compressor(pair)
            compressed_parts.append(compressed)

        # Handle odd last element
        if len(activations) % 2 == 1:
            compressed_parts.append(activations[-1])

        # Average all compressed parts
        return torch.stack(compressed_parts).mean(dim=0)

    def clear(self) -> None:
        """Clear the buffer."""
        self.token_buffer.clear()
        self.activation_buffer.clear()

    def __len__(self) -> int:
        return len(self.token_buffer)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.token_buffer) >= self.config.max_length

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        activations = self.get_activations()

        return {
            "length": len(self.token_buffer),
            "max_length": self.config.max_length,
            "is_full": self.is_full(),
            "activation_stats": {
                "mean": activations.mean().item() if len(activations) > 0 else 0,
                "std": activations.std().item() if len(activations) > 0 else 0,
            } if len(activations) > 0 else {},
        }
