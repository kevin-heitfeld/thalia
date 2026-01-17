"""Sequence Memory - Hippocampal Learning of Temporal Patterns and Language.

This module provides a hippocampus-based sequence memory system that can:
1. **Store sequences** of tokens/patterns with temporal order
2. **Recall previous context** to predict next elements
3. **Learn associations** between sequence elements via CA3 recurrence

**Architecture**:
=================

.. code-block:: none

    Token Sequence: [A, B, C, D, ?]
                     │  │  │  │
                     ▼  ▼  ▼  ▼
              ┌─────────────────────┐
              │  Theta Phase Encoder │  Temporal order via phase
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Hippocampus      │
              │   DG: Separation    │  Each token gets unique code
              │   CA3: Association  │  Store A→B, B→C, C→D links
              │   CA1: Output       │  Predict next token
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Pattern Recall    │  Given [A,B,C,D] → predict [E]?
              └─────────────────────┘

**How It Works**:
=================
1. Each token is encoded with its theta phase (position in sequence)
2. CA3 recurrent connections learn: token_n → token_n+1
3. During recall, partial cue activates full sequence via pattern completion
4. Output is the predicted next token(s)

**Biological Basis**:
=====================
- **Theta sequences**: Place cells fire in sequence within each theta cycle
- **Hippocampal replay**: Sequences are replayed during rest/sleep for consolidation
- **Temporal context**: Neurons encode not just WHAT but WHEN in sequence
- **Phase precession**: Position in sequence encoded by theta phase

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from thalia.components.coding.spike_coding import CodingStrategy
from thalia.config import SequenceMemoryConfig
from thalia.language.encoder import SpikeEncoder, SpikeEncoderConfig
from thalia.language.position import OscillatoryPositionEncoder, PositionEncoderConfig
from thalia.mixins import ConfigurableMixin, DiagnosticCollectorMixin
from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from thalia.utils.core_utils import cosine_similarity_safe


@dataclass
class SequenceContext:
    """A stored sequence context with associated metadata.

    Represents a remembered sequence or context that can be
    used for prediction or associative recall.
    """

    tokens: torch.Tensor  # Token IDs in sequence
    activations: torch.Tensor  # Hippocampal CA3 pattern for this context
    position: int  # Position in larger context
    timestamp: int = 0  # When this was stored
    retrieval_count: int = 0  # How many times retrieved
    strength: float = 1.0  # Memory strength (decays or strengthens)

    def to(self, device: torch.device) -> SequenceContext:
        """Move tensors to device."""
        return SequenceContext(
            tokens=self.tokens.to(device),
            activations=self.activations.to(device),
            position=self.position,
            timestamp=self.timestamp,
            retrieval_count=self.retrieval_count,
            strength=self.strength,
        )


class SequenceMemory(ConfigurableMixin, nn.Module, DiagnosticCollectorMixin):
    """
    Hippocampus-based sequence memory for language processing.

    Uses the trisynaptic hippocampus circuit to:
    1. Encode token sequences with temporal order (via theta/gamma)
    2. Store associations between consecutive tokens (CA3 recurrent)
    3. Recall next tokens given context (pattern completion)

    Example:
        >>> from thalia.config import ThaliaConfig
        >>> config = ThaliaConfig(...)
        >>> memory = SequenceMemory.from_thalia_config(config)
        >>>
        >>> # Encode a sequence
        >>> tokens = torch.tensor([[1, 5, 3, 7, 9]])  # [batch, seq]
        >>> memory.encode_sequence(tokens)
        >>>
        >>> # Query: given [1, 5, 3], what comes next?
        >>> query = torch.tensor([[1, 5, 3]])
        >>> predicted = memory.predict_next(query)  # Should activate pattern for 7
    """

    # For ConfigurableMixin - specifies how to extract config from ThaliaConfig
    CONFIG_CONVERTER_METHOD = "to_sequence_memory_config"

    def __init__(self, config: SequenceMemoryConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Spike encoder for tokens
        encoder_config = SpikeEncoderConfig(
            vocab_size=config.vocab_size,
            n_neurons=config.n_neurons,
            n_timesteps=10,  # Timesteps per token
            coding_strategy=CodingStrategy.SDR,
            sparsity=0.05,
            device=config.device,
        )
        self.encoder = SpikeEncoder(encoder_config)

        # Position encoder for temporal order
        position_config = PositionEncoderConfig(
            n_neurons=config.n_neurons,
            max_positions=config.context_length,
            theta_frequency_hz=config.theta_frequency,
            gamma_frequency_hz=config.gamma_frequency,
            device=config.device,
        )
        self.position_encoder = OscillatoryPositionEncoder(position_config)

        # Hippocampus for sequence storage
        hippo_config = HippocampusConfig(
            n_input=config.n_neurons,
            n_output=config.n_neurons,
            learning_rate=config.learning_rate,
            ca3_learning_rate=config.learning_rate,
            device=config.device,
        )
        self.hippocampus = Hippocampus(hippo_config)

        # Get CA3 size for association weights
        self.ca3_size = self.hippocampus.ca3_size

        # Stored contexts (for explicit storage, beyond what CA3 learns)
        self.stored_contexts: deque[SequenceContext] = deque(maxlen=config.max_stored_contexts)

        # Association weights: next-token prediction
        # These learn which CA3 patterns predict which next CA3 patterns
        from thalia.components.synapses.weight_init import WeightInitializer
        from thalia.constants.learning import WEIGHT_INIT_SCALE_RECURRENT

        self.association_weights = nn.Parameter(
            WeightInitializer.gaussian(
                self.ca3_size,
                self.ca3_size,
                mean=0.0,
                std=WEIGHT_INIT_SCALE_RECURRENT,
                device=self.device,
            )
        )

        # Statistics
        self.stats = {
            "sequences_encoded": 0,
            "predictions_made": 0,
            "contexts_stored": 0,
        }

        self.to(self.device)

    def reset_state(self) -> None:
        """Reset memory state for new sequence."""
        self.hippocampus.reset_state()
        # Position encoder doesn't need reset (stateless)

    def encode_sequence(
        self,
        token_ids: torch.Tensor,
        learn: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode a token sequence into hippocampal memory.

        This processes tokens one by one, learning the associations
        between consecutive tokens via CA3 recurrent connections.

        Args:
            token_ids: Token IDs, shape (batch, seq_len)
            learn: Whether to update weights (enable for training)

        Returns:
            Dict with:
                - "patterns": List of CA3 patterns for each position
                - "associations_learned": Number of associations formed
                - "final_state": Final hippocampal state
        """
        batch_size, seq_len = token_ids.shape

        self.reset_state()

        patterns: List[torch.Tensor] = []
        prev_pattern: Optional[torch.Tensor] = None
        ca3_pattern: torch.Tensor = torch.zeros(batch_size, self.ca3_size, device=self.device)
        associations_learned = 0

        for pos in range(seq_len):
            # Get current token
            current_token = token_ids[:, pos : pos + 1]  # [batch, 1]

            # Encode token to spikes
            spikes, _ = self.encoder(current_token)  # [batch, 1, timesteps, neurons]
            token_spikes = spikes.squeeze(1)  # [batch, timesteps, neurons]

            # Add position encoding
            pos_ids = torch.tensor([[pos]], device=self.device)  # [1, 1]
            position_enc = self.position_encoder(
                pos_ids, as_spikes=True
            )  # [1, 1, timesteps, neurons]
            position_enc = position_enc.squeeze(0).squeeze(0)  # [timesteps, neurons]

            # Combine: phase modulates spike probability
            n_timesteps = min(token_spikes.size(1), position_enc.size(0))
            combined_spikes = token_spikes[:, :n_timesteps, :] * (
                1.0 + 0.5 * position_enc[:n_timesteps, :]
            )

            # Process through hippocampus (use sum over timesteps as input)
            # Theta modulation computed internally by hippocampus
            hippo_input = combined_spikes.sum(dim=1)  # [batch, neurons]
            _ = self.hippocampus.forward(hippo_input)

            # Get CA3 pattern (the "memory" for this position)
            ca3_spikes = self.hippocampus.state.ca3_spikes
            if ca3_spikes is not None:
                ca3_pattern = ca3_spikes.clone()
            else:
                ca3_pattern = torch.zeros(batch_size, self.ca3_size, device=self.device)
            patterns.append(ca3_pattern)

            # Learn association: prev_pattern → current_pattern
            if learn and prev_pattern is not None:
                self._learn_association(prev_pattern, ca3_pattern)
                associations_learned += 1

            prev_pattern = ca3_pattern

        self.stats["sequences_encoded"] += 1

        # Get final state
        final_spikes = self.hippocampus.state.ca3_spikes
        final_state = final_spikes.clone() if final_spikes is not None else ca3_pattern

        return {
            "patterns": patterns,
            "associations_learned": associations_learned,
            "final_state": final_state,
        }

    def _learn_association(
        self,
        pre_pattern: torch.Tensor,
        post_pattern: torch.Tensor,
    ) -> None:
        """
        Learn association between consecutive patterns using Hebbian learning.

        This strengthens connections from neurons active in pre_pattern
        to neurons active in post_pattern.

        This is the core mechanism for next-token prediction:
        When we see pattern A, we want to activate pattern B (the next token).
        """
        # Simple Hebbian: Δw = η * pre * post
        lr = self.config.learning_rate * self.config.association_strength

        # Outer product: (n_neurons,) x (n_neurons,) → (n_neurons, n_neurons)
        pre = pre_pattern.float().squeeze()
        post = post_pattern.float().squeeze()

        # Hebbian update
        dw = lr * torch.outer(post, pre)
        self.association_weights.data += dw

        # Weight normalization to prevent runaway
        norm = self.association_weights.data.norm(dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1.0)
        self.association_weights.data /= norm

    def predict_next(
        self,
        context_tokens: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict next token(s) given context.

        Encodes the context, then uses CA3 pattern completion and
        learned associations to predict what comes next.

        Args:
            context_tokens: Context token IDs, shape (batch, context_len)
            top_k: Number of top predictions to return

        Returns:
            Dict with:
                - "predicted_pattern": Predicted next-token pattern
                - "similarity_scores": Similarity to stored patterns
                - "top_k_indices": Indices of top-k similar patterns
        """
        # Encode context to get final CA3 pattern
        result = self.encode_sequence(context_tokens, learn=False)
        context_pattern = result["final_state"]  # [neurons]

        # Use association weights to predict next pattern
        # next_pattern ≈ W @ context_pattern
        predicted_pattern = torch.matmul(
            self.association_weights, context_pattern.float().squeeze()
        )

        # Apply threshold to make it sparse (like real CA3 output)
        threshold = predicted_pattern.mean() + predicted_pattern.std()
        predicted_pattern = (predicted_pattern > threshold).float()

        # Compare to stored patterns (if any)
        similarities = []
        for ctx in self.stored_contexts:
            sim = cosine_similarity_safe(
                predicted_pattern.unsqueeze(0),
                ctx.activations.float().unsqueeze(0),
            )
            similarities.append(sim.item())

        self.stats["predictions_made"] += 1

        return {
            "predicted_pattern": predicted_pattern,
            "similarity_scores": similarities,
            "context_pattern": context_pattern,
        }

    def store_context(
        self,
        token_ids: torch.Tensor,
        position: int = 0,
    ) -> None:
        """
        Explicitly store a context for later retrieval.

        This creates a SequenceContext that can be looked up
        beyond what CA3 learns implicitly.
        """
        result = self.encode_sequence(token_ids, learn=False)

        context = SequenceContext(
            tokens=token_ids.clone(),
            activations=result["final_state"],
            position=position,
            timestamp=self.stats["sequences_encoded"],
        )

        self.stored_contexts.append(context)
        self.stats["contexts_stored"] += 1

    def retrieve_similar(
        self,
        query_pattern: torch.Tensor,
        top_k: int = 3,
    ) -> List[SequenceContext]:
        """
        Retrieve stored contexts similar to query pattern.

        Uses cosine similarity to find matching contexts.
        """
        if not self.stored_contexts:
            return []

        # Compute similarities
        similarities = []
        for ctx in self.stored_contexts:
            sim = cosine_similarity_safe(
                query_pattern.float().unsqueeze(0),
                ctx.activations.float().unsqueeze(0),
            )
            similarities.append((sim.item(), ctx))

        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Return top-k
        return [ctx for _, ctx in similarities[:top_k]]

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information using auto-collection."""
        # Use mixin auto_collect_diagnostics for standardized collection
        return self.auto_collect_diagnostics(
            weights={"association_weights": self.association_weights},
            scalars={
                "n_stored_contexts": len(self.stored_contexts),
                "vocab_size": self.config.vocab_size,
                "context_length": self.config.context_length,
                **self.stats,  # Include all stats
            },
        )
