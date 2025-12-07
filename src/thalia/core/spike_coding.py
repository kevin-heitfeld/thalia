"""
Spike Coding - Unified base classes for encoding/decoding patterns.

This module consolidates common patterns across different encoder/decoder implementations:
- SpikeEncoder: Base class for converting data → spikes
- SpikeDecoder: Base class for converting spikes → data
- Shared coding strategies (rate, temporal, population, phase)

The key insight: Once any modality is converted to spikes, processing is unified.
Different modalities need different front-end encoders, but they all produce spike
patterns that can be processed by the same downstream circuits.

Consolidation Benefits:
=======================
1. Reduces code duplication (~200 lines saved)
2. Ensures consistent spike coding across modalities
3. Makes it easy to add new modalities
4. Centralized improvement of encoding strategies

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class CodingStrategy(Enum):
    """Spike coding strategies (shared across encoders/decoders)."""
    RATE = "rate"           # Spike count encodes value
    TEMPORAL = "temporal"   # Spike timing encodes value
    POPULATION = "population"  # Population codes
    PHASE = "phase"         # Phase relative to oscillation
    BURST = "burst"         # Burst patterns
    SDR = "sdr"             # Sparse distributed representation
    WTA = "wta"             # Winner-take-all


@dataclass
class SpikeCodingConfig:
    """Base configuration for spike coding operations.

    All encoders/decoders should inherit from this to ensure
    compatible parameters.
    """
    # Core dimensions
    n_neurons: int = 1024
    n_timesteps: int = 20

    # Temporal parameters
    dt_ms: float = 1.0
    tau_ms: float = 20.0  # Integration time constant

    # Coding strategy
    coding_strategy: CodingStrategy = CodingStrategy.RATE

    # Sparsity (for SDR/population codes)
    sparsity: float = 0.05  # Target fraction of active neurons

    # Phase coding
    oscillation_frequency_hz: float = 8.0  # Theta rhythm

    # Temperature for decoding
    temperature: float = 1.0

    # Device
    device: str = "cpu"

    @property
    def decay_factor(self) -> float:
        """Exponential decay factor for leaky integration."""
        return 1.0 - self.dt_ms / self.tau_ms

    @property
    def oscillation_period_ms(self) -> float:
        """Period of oscillation in ms."""
        return 1000.0 / self.oscillation_frequency_hz


class SpikeEncoder(nn.Module, ABC):
    """
    Abstract base class for spike encoders.

    Subclasses implement modality-specific encoding:
    - TokenEncoder: text tokens → spikes
    - ImageEncoder: images → spikes
    - AudioEncoder: audio → spikes
    - etc.

    All produce compatible spike patterns: [batch, seq_len, n_timesteps, n_neurons]
    """

    def __init__(self, config: SpikeCodingConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Initialize strategy-specific components
        self._init_encoding_strategy()

    def _init_encoding_strategy(self) -> None:
        """Initialize components based on coding strategy."""
        strategy = self.config.coding_strategy

        if strategy == CodingStrategy.RATE:
            # Rate coding: Convert activations to spike probabilities
            self.rate_scale = nn.Parameter(torch.ones(self.config.n_neurons))

        elif strategy == CodingStrategy.TEMPORAL:
            # Temporal coding: Latency encodes value
            self.latency_scale = nn.Parameter(torch.ones(self.config.n_neurons))

        elif strategy in (CodingStrategy.PHASE, CodingStrategy.BURST):
            # Phase/burst coding: Track oscillation phase
            self.register_buffer("phase", torch.tensor(0.0))
            self.phase_increment = (
                2 * 3.14159 * self.config.dt_ms / self.config.oscillation_period_ms
            )

    @abstractmethod
    def encode(self, input_data: Any) -> torch.Tensor:
        """
        Encode input data to spike patterns.

        Args:
            input_data: Modality-specific input

        Returns:
            spikes: [batch, seq_len, n_timesteps, n_neurons]
        """
        pass

    def forward(self, input_data: Any) -> torch.Tensor:
        """Forward pass delegates to encode()."""
        return self.encode(input_data)

    def _apply_coding_strategy(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spike coding strategy to features.

        Args:
            features: Continuous features [batch, seq_len, n_neurons]

        Returns:
            spikes: Binary spike trains [batch, seq_len, n_timesteps, n_neurons]
        """
        batch, seq_len, n_neurons = features.shape
        strategy = self.config.coding_strategy

        if strategy == CodingStrategy.RATE:
            # Rate coding: Spike probability proportional to feature strength
            # features should be in [0, 1] range
            features_expanded = features.unsqueeze(2).expand(
                batch, seq_len, self.config.n_timesteps, n_neurons
            )
            spike_prob = torch.sigmoid(features_expanded * self.rate_scale)
            spikes = torch.bernoulli(spike_prob)

        elif strategy == CodingStrategy.TEMPORAL:
            # Temporal coding: Stronger features spike earlier
            spikes = torch.zeros(
                batch, seq_len, self.config.n_timesteps, n_neurons,
                device=self.device,
            )

            # Convert features to latencies: higher value = earlier spike
            latencies = (1.0 - features.clamp(0, 1)) * self.config.n_timesteps
            latencies = (latencies * self.latency_scale).long()

            # Generate spikes at computed latencies
            for b in range(batch):
                for s in range(seq_len):
                    for n in range(n_neurons):
                        t = latencies[b, s, n].item()
                        if 0 <= t < self.config.n_timesteps:
                            spikes[b, s, t, n] = 1.0

        elif strategy == CodingStrategy.POPULATION:
            # Population coding: Distribute across neurons
            # Similar to rate but with population-level normalization
            features_expanded = features.unsqueeze(2).expand(
                batch, seq_len, self.config.n_timesteps, n_neurons
            )
            # Add noise for population variability
            noise = torch.randn_like(features_expanded) * 0.1
            spike_prob = torch.sigmoid(features_expanded + noise)
            spikes = torch.bernoulli(spike_prob)

        elif strategy == CodingStrategy.SDR:
            # Sparse Distributed Representation: Top-k activation
            spikes = torch.zeros(
                batch, seq_len, self.config.n_timesteps, n_neurons,
                device=self.device,
            )

            # Select top-k neurons based on feature strength
            k = int(self.config.sparsity * n_neurons)
            _, top_indices = torch.topk(features, k, dim=-1)

            # Create sparse pattern repeated over time
            for t in range(self.config.n_timesteps):
                # Add temporal jitter
                if torch.rand(1).item() < 0.5:  # 50% chance of spike per timestep
                    for b in range(batch):
                        for s in range(seq_len):
                            spikes[b, s, t, top_indices[b, s]] = 1.0

        else:
            raise NotImplementedError(f"Coding strategy {strategy} not implemented")

        return spikes

    def reset_state(self) -> None:
        """Reset any temporal state (e.g., adaptation, phase)."""
        if hasattr(self, "phase"):
            self.phase.zero_()


class SpikeDecoder(nn.Module, ABC):
    """
    Abstract base class for spike decoders.

    Subclasses implement modality-specific decoding:
    - TokenDecoder: spikes → token probabilities
    - MotorDecoder: spikes → motor commands
    - etc.

    All accept spike patterns: [batch, seq_len, n_timesteps, n_neurons]
    """

    def __init__(self, config: SpikeCodingConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Integration state for temporal processing
        self.register_buffer(
            "integration_state",
            torch.zeros(1, config.n_neurons),
        )

        # Initialize strategy-specific components
        self._init_decoding_strategy()

    def _init_decoding_strategy(self) -> None:
        """Initialize components based on coding strategy."""
        strategy = self.config.coding_strategy

        if strategy == CodingStrategy.TEMPORAL:
            # Track first spike times for latency decoding
            self.register_buffer(
                "first_spike_time",
                torch.full((1, self.config.n_neurons), float('inf')),
            )

        elif strategy == CodingStrategy.WTA:
            # Lateral inhibition for winner-take-all
            # (Subclasses define size)
            pass

    @abstractmethod
    def decode(self, spikes: torch.Tensor) -> Any:
        """
        Decode spike patterns to output.

        Args:
            spikes: [batch, seq_len, n_timesteps, n_neurons]

        Returns:
            output: Modality-specific output
        """
        pass

    def forward(self, spikes: torch.Tensor) -> Any:
        """Forward pass delegates to decode()."""
        return self.decode(spikes)

    def _integrate_spikes(
        self,
        spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate spikes over time using coding strategy.

        Args:
            spikes: [batch, seq_len, n_timesteps, n_neurons]

        Returns:
            features: [batch, seq_len, n_neurons]
        """
        batch, seq_len, n_timesteps, n_neurons = spikes.shape
        strategy = self.config.coding_strategy
        decay = self.config.decay_factor

        if strategy == CodingStrategy.RATE:
            # Rate decoding: Count spikes, normalize by time
            features = spikes.sum(dim=2) / n_timesteps

        elif strategy == CodingStrategy.TEMPORAL:
            # Temporal decoding: Earlier spikes weighted more
            features = torch.zeros(batch, seq_len, n_neurons, device=self.device)
            for t in range(n_timesteps):
                weight = (n_timesteps - t) / n_timesteps
                features += spikes[:, :, t, :] * weight

        elif strategy == CodingStrategy.POPULATION:
            # Population decoding: Leaky integration
            features = torch.zeros(batch, seq_len, n_neurons, device=self.device)
            state = torch.zeros(batch, n_neurons, device=self.device)

            for s in range(seq_len):
                for t in range(n_timesteps):
                    state = state * decay + spikes[:, s, t, :]
                features[:, s, :] = state

        elif strategy in (CodingStrategy.SDR, CodingStrategy.WTA):
            # SDR/WTA: Sum spikes over time (binary patterns)
            features = spikes.sum(dim=2)
            # Threshold to maintain sparsity
            if strategy == CodingStrategy.SDR:
                k = int(self.config.sparsity * n_neurons)
                threshold = torch.kthvalue(features, n_neurons - k, dim=-1, keepdim=True)[0]
                features = (features >= threshold).float() * features

        else:
            # Default: Simple spike count
            features = spikes.sum(dim=2)

        return features

    def reset_state(self) -> None:
        """Reset temporal integration state."""
        self.integration_state.zero_()
        if hasattr(self, "first_spike_time"):
            self.first_spike_time.fill_(float('inf'))


class RateEncoder(SpikeEncoder):
    """
    Simple rate-based encoder for testing/prototyping.

    Converts continuous values to spike rates via Poisson process.
    """

    def __init__(self, config: SpikeCodingConfig):
        super().__init__(config)
        self.config.coding_strategy = CodingStrategy.RATE

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous values as spike rates.

        Args:
            values: [batch, seq_len, n_neurons] in range [0, 1]

        Returns:
            spikes: [batch, seq_len, n_timesteps, n_neurons]
        """
        # Use base class coding strategy
        return self._apply_coding_strategy(values)


class RateDecoder(SpikeDecoder):
    """
    Simple rate-based decoder for testing/prototyping.

    Decodes spike counts back to continuous values.
    """

    def __init__(self, config: SpikeCodingConfig):
        super().__init__(config)
        self.config.coding_strategy = CodingStrategy.RATE

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Decode spikes to continuous values via spike counts.

        Args:
            spikes: [batch, seq_len, n_timesteps, n_neurons]

        Returns:
            values: [batch, seq_len, n_neurons]
        """
        # Use base class integration
        return self._integrate_spikes(spikes)


def compute_spike_similarity(
    spikes1: torch.Tensor,
    spikes2: torch.Tensor,
    method: str = "cosine",
) -> torch.Tensor:
    """
    Compute similarity between spike patterns.

    Args:
        spikes1, spikes2: Spike patterns [batch, seq_len, n_timesteps, n_neurons]
        method: "cosine", "correlation", "overlap"

    Returns:
        similarity: [batch, seq_len]
    """
    from thalia.core.utils import cosine_similarity_safe
    
    # Flatten temporal dimension
    flat1 = spikes1.reshape(*spikes1.shape[:2], -1)
    flat2 = spikes2.reshape(*spikes2.shape[:2], -1)

    if method == "cosine":
        # Cosine similarity - use canonical implementation
        similarity = cosine_similarity_safe(flat1, flat2, eps=1e-6, dim=-1)

    elif method == "correlation":
        # Pearson correlation
        mean1 = flat1.mean(dim=-1, keepdim=True)
        mean2 = flat2.mean(dim=-1, keepdim=True)
        centered1 = flat1 - mean1
        centered2 = flat2 - mean2
        similarity = (centered1 * centered2).sum(dim=-1) / (
            centered1.norm(dim=-1) * centered2.norm(dim=-1) + 1e-6
        )

    elif method == "overlap":
        # Jaccard similarity (for binary spikes)
        intersection = (flat1 * flat2).sum(dim=-1)
        union = torch.clamp(flat1 + flat2, 0, 1).sum(dim=-1)
        similarity = intersection / (union + 1e-6)

    else:
        raise ValueError(f"Unknown similarity method: {method}")

    return similarity
