"""
Spike Encoder - Convert tokens to sparse distributed spike representations.

This module implements the encoding of discrete tokens (words, subwords, characters)
into spike patterns that can be processed by spiking neural networks.

Key Concepts:
=============

1. SPARSE DISTRIBUTED REPRESENTATIONS (SDR)
   Instead of one-hot encoding (1 neuron per token), we use SDR:
   - Each token activates a sparse subset of neurons (~2-5% active)
   - Similar tokens share some active neurons (semantic similarity)
   - Robust to noise (a few wrong neurons don't matter)
   - High capacity (combinatorial explosion of patterns)

2. TEMPORAL SPIKE CODING
   The "when" of spikes matters, not just "which":
   - Rate coding: More spikes = stronger activation
   - Temporal coding: Early spikes = more important
   - Phase coding: Spike timing relative to oscillation

3. POPULATION CODING
   Multiple neurons represent each token:
   - Redundancy for noise robustness
   - Graded confidence through firing rates
   - Natural for neural hardware

Biological Basis:
- Hippocampus uses sparse coding (~2% active cells)
- Cortex uses distributed population codes
- Timing is crucial for memory encoding (theta phase)

References:
- Olshausen & Field (1996): Emergence of simple-cell properties
- Quiroga et al. (2008): Sparse but not "grandmother cell" coding
- Ahmad & Hawkins (2016): Properties of sparse distributed representations

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import math

import torch
import torch.nn as nn

from thalia.components.coding.spike_coding import (
    CodingStrategy,
    SpikeCodingConfig,
    SpikeEncoder as BaseSpikeEncoder,
)


@dataclass
class SpikeEncoderConfig(SpikeCodingConfig):
    """Configuration for spike encoder.

    Extends SpikeCodingConfig with token-specific parameters.

    Attributes:
        vocab_size: Size of token vocabulary
        embedding_dim: Dimension of token embeddings (if using pretrained)
        sdr_on_bits: Number of active bits in SDR (if using fixed SDR)
        sdr_overlap: Semantic overlap between similar tokens
        learnable_embedding: Whether embedding is learnable
        pretrained_embedding: Optional pretrained embedding matrix
    """
    vocab_size: int = 50257  # GPT-2 vocabulary size
    embedding_dim: int = 256

    # SDR parameters (specific to tokens)
    sdr_on_bits: int = 50  # ~5% of 1024
    sdr_overlap: float = 0.3  # 30% overlap for similar tokens

    # Learning
    learnable_embedding: bool = True
    pretrained_embedding: Optional[torch.Tensor] = None

    @property
    def theta_period_ms(self) -> float:
        """Period of theta oscillation in ms (compatibility)."""
        return self.oscillation_period_ms

    def __post_init__(self):
        # Ensure sdr_on_bits matches sparsity target
        if self.sdr_on_bits is None:
            self.sdr_on_bits = int(self.n_neurons * self.sparsity)


class SparseDistributedRepresentation(nn.Module):
    """
    Generate Sparse Distributed Representations for tokens.

    SDR is a key concept from HTM (Hierarchical Temporal Memory):
    - Each token maps to a fixed sparse pattern
    - Patterns are generated via hash-like functions
    - Similar semantics can share some bits (if learned)

    Benefits:
    - Very high capacity: C(n, k) combinations
    - Noise robust: Hamming distance for similarity
    - Union-able: Combine patterns via OR
    - Overlap = similarity
    """

    def __init__(self, config: SpikeEncoderConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Option 1: Fixed random SDR (hash-based)
        # Generate deterministic patterns for each token
        self._generate_fixed_patterns()

        # Option 2: Learnable SDR generator
        if config.learnable_embedding:
            self.token_embedding = nn.Embedding(
                config.vocab_size,
                config.embedding_dim,
            )
            if config.pretrained_embedding is not None:
                self.token_embedding.weight.data = config.pretrained_embedding

            # Project embedding to SDR space
            self.sdr_projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.n_neurons * 2),
                nn.GELU(),
                nn.Linear(config.n_neurons * 2, config.n_neurons),
            )
        else:
            self.token_embedding = None
            self.sdr_projection = None

    def _generate_fixed_patterns(self) -> None:
        """Generate fixed SDR patterns for each token."""
        # Use a seeded random generator for reproducibility
        rng = torch.Generator()
        rng.manual_seed(42)

        # For each token, select random neurons to be active
        patterns = torch.zeros(
            self.config.vocab_size,
            self.config.n_neurons,
            device=self.device,
        )

        for token_id in range(self.config.vocab_size):
            # Generate random indices for active neurons
            active_indices = torch.randperm(
                self.config.n_neurons,
                generator=rng,
            )[:self.config.sdr_on_bits]
            patterns[token_id, active_indices] = 1.0

        self.register_buffer("fixed_patterns", patterns)

    def forward(
        self,
        token_ids: torch.Tensor,
        use_learned: bool = True,
    ) -> torch.Tensor:
        """
        Convert token IDs to SDR patterns.

        Args:
            token_ids: Token indices [batch, seq_len]
            use_learned: Whether to use learned embeddings (if available)

        Returns:
            sdr: Sparse patterns [batch, seq_len, n_neurons]
        """
        if use_learned and self.token_embedding is not None:
            # Learned SDR
            embeddings = self.token_embedding(token_ids)  # [batch, seq, emb_dim]
            logits = self.sdr_projection(embeddings)  # [batch, seq, n_neurons]

            # Apply top-k sparsity
            k = self.config.sdr_on_bits
            topk_values, topk_indices = torch.topk(logits, k, dim=-1)

            # Create sparse SDR
            sdr = torch.zeros_like(logits)
            sdr.scatter_(-1, topk_indices, torch.ones_like(topk_values))

            return sdr
        else:
            # Fixed SDR from precomputed patterns
            return self.fixed_patterns[token_ids]

    def get_similarity(
        self,
        sdr1: torch.Tensor,
        sdr2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between SDR patterns.

        Uses overlap score (Jaccard-like): intersection / union

        Args:
            sdr1: First SDR pattern [*, n_neurons]
            sdr2: Second SDR pattern [*, n_neurons]

        Returns:
            similarity: Overlap score [*]
        """
        intersection = (sdr1 * sdr2).sum(dim=-1)
        union = torch.clamp(sdr1 + sdr2, 0, 1).sum(dim=-1)
        return intersection / (union + 1e-6)


class SpikeEncoder(BaseSpikeEncoder):
    """
    Encode tokens into spike trains.

    This is the main interface for converting text to spikes:
    1. Token IDs → SDR patterns
    2. SDR patterns → Temporal spike trains
    3. Add position information via phase coding

    The output is a sequence of spike patterns over time that can
    be fed into spiking neural networks.

    Usage:
        encoder = SpikeEncoder(SpikeEncoderConfig(
            vocab_size=50000,
            n_neurons=1024,
            n_timesteps=20,
        ))

        # Encode tokens
        token_ids = torch.tensor([[1, 42, 100, 5]])  # [batch, seq_len]
        spikes = encoder(token_ids)  # [batch, seq_len, n_timesteps, n_neurons]
    """

    def __init__(self, config: SpikeEncoderConfig):
        super().__init__(config)
        self.config: SpikeEncoderConfig = config  # Type annotation for proper inference

        # SDR generator (token-specific)
        self.sdr = SparseDistributedRepresentation(config)

        # Phase tracking for temporal coding (theta oscillation)
        self.register_buffer('theta_phase', torch.tensor(0.0))
        # Phase increment per timestep (radians)
        theta_freq_hz = 8.0  # 8 Hz theta oscillation
        dt_ms = 1.0  # 1ms timestep default
        self.phase_increment = 2 * math.pi * theta_freq_hz * (dt_ms / 1000.0)

    def reset_phase(self) -> None:
        """Reset theta phase for new sequence."""
        if hasattr(self, 'theta_phase'):
            self.theta_phase.zero_()

    def encode(
        self,
        token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode tokens into spike trains.

        Args:
            token_ids: Token indices [batch, seq_len]
            position_ids: Optional position indices [batch, seq_len]

        Returns:
            spikes: Spike trains [batch, seq_len, n_timesteps, n_neurons]
        """
        # Step 1: Get SDR patterns (continuous features)
        sdr = self.sdr(token_ids)  # [batch, seq_len, n_neurons]

        # Step 2: Convert to temporal spike trains using base class
        spikes = self._apply_coding_strategy(sdr)  # [batch, seq, timesteps, neurons]

        # Step 3: Add position information via phase modulation
        if position_ids is not None:
            spikes = self._add_position_phase(spikes, position_ids)

        return spikes

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokens into spike trains (returns both spikes and SDR).

        Args:
            token_ids: Token indices [batch, seq_len]
            position_ids: Optional position indices [batch, seq_len]

        Returns:
            spikes: Spike trains [batch, seq_len, n_timesteps, n_neurons]
            sdr: Underlying SDR patterns [batch, seq_len, n_neurons]
        """
        sdr = self.sdr(token_ids)
        spikes = self.encode(token_ids, position_ids)
        return spikes, sdr

    def _generate_spike_train(
        self,
        sdr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert SDR patterns to temporal spike trains.

        Different encoding strategies create different temporal patterns.

        Args:
            sdr: SDR patterns [batch, seq_len, n_neurons]

        Returns:
            spikes: Spike trains [batch, seq_len, n_timesteps, n_neurons]
        """
        batch, seq_len, n_neurons = sdr.shape
        n_timesteps = self.config.n_timesteps

        if self.config.encoding_type == CodingStrategy.SDR:
            # Simple SDR: Same pattern repeated, sparsely spiking
            # Active neurons spike once at a random time
            spikes = torch.zeros(
                batch, seq_len, n_timesteps, n_neurons,
                device=self.device,
            )

            # Each active neuron spikes at a random timestep
            active_mask = sdr > 0.5  # [batch, seq, neurons]

            for t in range(n_timesteps):
                # Probability of spiking at each timestep
                spike_prob = sdr / n_timesteps
                random_spikes = torch.rand_like(sdr) < spike_prob
                spikes[:, :, t, :] = random_spikes.float()

            # Ensure at least one spike for active neurons
            # Find neurons that haven't spiked
            total_spikes = spikes.sum(dim=2)  # [batch, seq, neurons]
            no_spike = (total_spikes == 0) & active_mask

            # Force a spike at a random time
            if no_spike.any():
                random_times = torch.randint(0, n_timesteps, no_spike.shape, device=self.device)
                for b in range(batch):
                    for s in range(seq_len):
                        for n in range(n_neurons):
                            if no_spike[b, s, n]:
                                spikes[b, s, random_times[b, s, n], n] = 1.0

        elif self.config.encoding_type == CodingStrategy.RATE:
            # Rate coding: Spike probability proportional to activation
            spike_prob = torch.sigmoid(sdr * self.rate_scale)  # [batch, seq, neurons]
            spike_prob = spike_prob.unsqueeze(2).expand(-1, -1, n_timesteps, -1)
            spikes = (torch.rand_like(spike_prob) < spike_prob).float()

        elif self.config.encoding_type == CodingStrategy.TEMPORAL:
            # Temporal coding: Higher activation → earlier spike
            # Compute spike times based on activation rank
            spikes = torch.zeros(
                batch, seq_len, n_timesteps, n_neurons,
                device=self.device,
            )

            # Spike time is inversely proportional to activation
            # Strongest neurons spike first
            spike_times = ((1 - sdr) * self.temporal_scale).long()
            spike_times = torch.clamp(spike_times, 0, n_timesteps - 1)

            # Only active neurons (SDR > 0.5) actually spike
            active_mask = sdr > 0.5

            # Set spikes at computed times
            for b in range(batch):
                for s in range(seq_len):
                    for n in range(n_neurons):
                        if active_mask[b, s, n]:
                            t = spike_times[b, s, n].item()
                            spikes[b, s, t, n] = 1.0

        elif self.config.encoding_type == CodingStrategy.PHASE:
            # Phase coding: Spike at specific phase relative to theta
            spikes = torch.zeros(
                batch, seq_len, n_timesteps, n_neurons,
                device=self.device,
            )

            for t in range(n_timesteps):
                # Current theta phase
                phase = (self.theta_phase + t * self.phase_increment) % (2 * math.pi)

                # Neurons spike when phase matches their preferred phase
                preferred_phase = self.phase_offset + sdr * math.pi  # Active = peak phase
                phase_diff = torch.abs(phase - preferred_phase)
                phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)

                # Spike if phase is close enough
                phase_window = 0.5  # Radians
                spike_mask = (phase_diff < phase_window) & (sdr > 0.5)
                spikes[:, :, t, :] = spike_mask.float()

        elif self.config.encoding_type == CodingStrategy.BURST:
            # Burst coding: Number of spikes in burst encodes strength
            spikes = torch.zeros(
                batch, seq_len, n_timesteps, n_neurons,
                device=self.device,
            )

            # Number of spikes proportional to activation
            n_spikes = (sdr * n_timesteps * self.burst_threshold).long()
            n_spikes = torch.clamp(n_spikes, 0, n_timesteps)

            # Generate bursts at the beginning of the window
            for b in range(batch):
                for s in range(seq_len):
                    for n in range(n_neurons):
                        n_s = n_spikes[b, s, n].item()
                        if n_s > 0:
                            spikes[b, s, :n_s, n] = 1.0

        else:
            raise ValueError(f"Unknown encoding type: {self.config.encoding_type}")

        return spikes

    def _add_position_phase(
        self,
        spikes: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modulate spike patterns based on sequence position.

        Uses phase coding: each position has a characteristic phase signature.
        This is inspired by theta phase precession in hippocampus.

        Args:
            spikes: Spike trains [batch, seq_len, n_timesteps, n_neurons]
            position_ids: Position indices [batch, seq_len]

        Returns:
            phase_modulated_spikes: Modified spike patterns
        """
        batch, seq_len, n_timesteps, n_neurons = spikes.shape

        # Position determines starting phase
        position_phase = position_ids.float() * 0.2  # ~0.2 radians per position

        # Create phase-dependent modulation
        for t in range(n_timesteps):
            current_phase = position_phase + t * self.phase_increment

            # Modulation: shift some spikes based on position phase
            modulation = torch.sin(current_phase).unsqueeze(-1)  # [batch, seq, 1]

            # Slight timing shifts based on position
            # (In a full implementation, this would actually shift spike times)
            # For now, we just modulate amplitude slightly
            spikes[:, :, t, :] = spikes[:, :, t, :] * (0.9 + 0.1 * modulation)

        return spikes

    def encode_text(
        self,
        text: str,
        tokenizer: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Convenience method to encode text directly.

        Args:
            text: Input text string
            tokenizer: Tokenizer with encode() method

        Returns:
            spikes: Spike trains
            sdr: SDR patterns
            token_ids: Token ID list
        """
        token_ids = tokenizer.encode(text)
        token_tensor = torch.tensor([token_ids], device=self.device)
        position_ids = torch.arange(len(token_ids), device=self.device).unsqueeze(0)

        spikes, sdr = self(token_tensor, position_ids)

        return spikes, sdr, token_ids

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get encoding diagnostics."""
        return {
            "encoding_type": self.config.encoding_type.value,
            "n_neurons": self.config.n_neurons,
            "sparsity": self.config.sparsity,
            "n_timesteps": self.config.n_timesteps,
            "sdr_on_bits": self.config.sdr_on_bits,
            "vocab_size": self.config.vocab_size,
        }


class HierarchicalSpikeEncoder(nn.Module):
    """
    Multi-resolution spike encoding for language.

    Encodes at multiple levels:
    - Character level: Fine-grained, fast
    - Subword level: BPE tokens
    - Word level: Semantic units

    Different levels use different timescales and neuron populations.
    """

    def __init__(
        self,
        char_config: Optional[SpikeEncoderConfig] = None,
        subword_config: Optional[SpikeEncoderConfig] = None,
        word_config: Optional[SpikeEncoderConfig] = None,
    ):
        super().__init__()

        # Character encoder (fast, fine-grained)
        if char_config is not None:
            char_config.n_timesteps = 10  # Fast
            self.char_encoder = SpikeEncoder(char_config)
        else:
            self.char_encoder = None

        # Subword encoder (medium)
        if subword_config is not None:
            self.subword_encoder = SpikeEncoder(subword_config)
        else:
            self.subword_encoder = None

        # Word encoder (slow, semantic)
        if word_config is not None:
            word_config.n_timesteps = 50  # Slow
            self.word_encoder = SpikeEncoder(word_config)
        else:
            self.word_encoder = None

    def forward(
        self,
        char_ids: Optional[torch.Tensor] = None,
        subword_ids: Optional[torch.Tensor] = None,
        word_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode at available levels."""
        outputs = {}

        if char_ids is not None and self.char_encoder is not None:
            spikes, sdr = self.char_encoder(char_ids)
            outputs["char_spikes"] = spikes
            outputs["char_sdr"] = sdr

        if subword_ids is not None and self.subword_encoder is not None:
            spikes, sdr = self.subword_encoder(subword_ids)
            outputs["subword_spikes"] = spikes
            outputs["subword_sdr"] = sdr

        if word_ids is not None and self.word_encoder is not None:
            spikes, sdr = self.word_encoder(word_ids)
            outputs["word_spikes"] = spikes
            outputs["word_sdr"] = sdr

        return outputs
