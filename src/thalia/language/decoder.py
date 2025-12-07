"""
Spike Decoder - Convert spike patterns back to token probabilities.

This module implements the decoding of spike patterns from spiking neural
networks back to discrete tokens for language generation.

Key Concepts:
=============

1. POPULATION READOUT
   Multiple neurons contribute to each output token:
   - Weighted sum of spike counts over time window
   - Winner-take-all competition
   - Softmax over token logits

2. TEMPORAL INTEGRATION
   Spike timing information is integrated:
   - Leaky integration (exponential decay)
   - Coincidence detection
   - Burst detection

3. HEBBIAN LEARNING (Delta Rule)
   The decoder learns via local rules without backpropagation:
   - Linear readout: logits = W @ activity
   - Delta rule: W += lr * outer(error, activity)
   - Error = target - prediction (one-hot vs softmax)
   - Biologically plausible: each synapse uses only local information

4. CONFIDENCE ESTIMATION
   Spike patterns can indicate confidence:
   - Higher firing rates = more confident
   - Synchronous firing = strong evidence
   - Sparse firing = uncertain

Biological Basis:
- Motor cortex uses population vectors for movement direction
- Hippocampus uses pattern completion for recall
- Decision making involves accumulation to threshold
- Perceptron-like learning in cerebellar Purkinje cells

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.spike_coding import (
    CodingStrategy,
    SpikeCodingConfig,
    SpikeDecoder as BaseSpikeDecoder,
)


# Re-export CodingStrategy members for backward compatibility
DecodingType = CodingStrategy


@dataclass
class SpikeDecoderConfig(SpikeCodingConfig):
    """Configuration for spike decoder.

    Extends SpikeCodingConfig with vocabulary-specific parameters.

    Attributes:
        vocab_size: Size of output vocabulary
        n_populations: Number of neural populations per token
        inhibition_strength: Lateral inhibition for WTA
        learning_rate: Learning rate for delta rule
        weight_decay: L2 regularization on weights
    """
    vocab_size: int = 50257

    # Population
    n_populations: int = 1

    # WTA
    inhibition_strength: float = 0.1

    # Hebbian learning
    learning_rate: float = 0.01
    weight_decay: float = 1e-5

    @property
    def decoding_type(self) -> CodingStrategy:
        """Alias for coding_strategy (backward compatibility)."""
        return self.coding_strategy

    @property
    def integration_tau_ms(self) -> float:
        """Time constant for spike integration (compatibility)."""
        return self.tau_ms

    @property
    def decay_factor(self) -> float:
        """Exponential decay factor for leaky integration."""
        return 1.0 - self.dt_ms / self.tau_ms


class SpikeDecoder(BaseSpikeDecoder):
    """
    Decode spike patterns to token probabilities using Hebbian learning.

    This decoder uses a simple linear readout that learns via the delta rule
    (a biologically plausible local learning rule). No backpropagation is needed.

    Learning Rule (Delta Rule):
        logits = W @ activity
        prediction = softmax(logits)
        error = target_one_hot - prediction
        W += learning_rate * outer(error, activity)

    This is local because each synapse w_ij only needs:
    - Presynaptic activity: activity_j (brain neuron j's firing rate)
    - Postsynaptic error: error_i (difference at output token i)

    The decoder can work in several modes for spike integration:
    - Rate decoding: Count spikes in time window
    - Temporal decoding: Use first-spike latency
    - Population decoding: Weighted readout from populations
    - WTA decoding: Winner-take-all competition

    Usage:
        decoder = SpikeDecoder(SpikeDecoderConfig(
            n_neurons=1024,
            vocab_size=50000,
        ))

        # Decode spikes to probabilities
        spikes = torch.rand(batch, seq_len, n_timesteps, n_neurons) > 0.95
        logits = decoder(spikes.float())  # [batch, seq_len, vocab_size]

        # Learn from target tokens (no backprop!)
        decoder.learn(features, target_token_ids)
    """

    def __init__(self, config: SpikeDecoderConfig):
        super().__init__(config)
        self.config: SpikeDecoderConfig = config  # Type annotation for proper inference

        # Linear readout weights: [vocab_size, n_neurons]
        # Initialized with small random values
        readout_weights = torch.randn(config.vocab_size, config.n_neurons)
        readout_weights = readout_weights * (0.01 / (config.n_neurons ** 0.5))
        self.readout_weights = nn.Parameter(readout_weights)

        # Bias for each token
        self.readout_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Track learning statistics
        self.register_buffer("n_updates", torch.tensor(0))
        self.register_buffer("total_error", torch.tensor(0.0))

        # Store last features for learning (needed when learn() called separately)
        self._last_features: Optional[torch.Tensor] = None

        # Initialize WTA inhibition if needed
        if config.coding_strategy == CodingStrategy.WTA:
            inhibition = torch.ones(config.vocab_size, config.vocab_size)
            inhibition = inhibition - torch.eye(config.vocab_size)
            inhibition = inhibition * config.inhibition_strength
            self.register_buffer("inhibition", inhibition)

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Decode spikes to token logits.

        Args:
            spikes: Spike patterns [batch, seq_len, n_timesteps, n_neurons]

        Returns:
            logits: Token logits [batch, seq_len, vocab_size]
        """
        # Step 1: Integrate spikes over time using base class
        features = self._integrate_spikes(spikes)  # [batch, seq_len, n_neurons]

        # Step 2: Decode to token space
        logits = self._decode_features(features)  # [batch, seq_len, vocab_size]

        # Step 3: Apply temperature
        logits = logits / self.config.temperature

        return logits

    def forward(
        self,
        spikes: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode spikes to token probabilities.

        Args:
            spikes: Spike patterns [batch, seq_len, n_timesteps, n_neurons]
            return_features: Whether to return intermediate features

        Returns:
            logits: Token logits [batch, seq_len, vocab_size]
            features: (optional) Intermediate features [batch, seq_len, n_neurons]
        """
        logits = self.decode(spikes)

        if return_features:
            features = self._integrate_spikes(spikes)
            return logits, features
        return logits

    def _integrate_spikes(
        self,
        spikes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate spikes over timesteps.

        Args:
            spikes: Spike patterns [batch, seq_len, n_timesteps, n_neurons]

        Returns:
            features: Integrated features [batch, seq_len, n_neurons]
        """
        batch, seq_len, n_timesteps, n_neurons = spikes.shape
        decay = self.config.decay_factor

        if self.config.decoding_type == DecodingType.RATE:
            # Rate decoding: Simply count spikes
            features = spikes.sum(dim=2)  # [batch, seq_len, n_neurons]
            # Normalize by number of timesteps
            features = features / n_timesteps

        elif self.config.decoding_type == DecodingType.TEMPORAL:
            # Temporal decoding: First spikes matter most
            features = torch.zeros(
                batch, seq_len, n_neurons,
                device=self.device,
            )

            for t in range(n_timesteps):
                # Weight by inverse latency (early spikes matter more)
                weight = (n_timesteps - t) / n_timesteps
                features += spikes[:, :, t, :] * weight

        elif self.config.decoding_type == DecodingType.POPULATION:
            # Population decoding: Leaky integration
            features = torch.zeros(
                batch, seq_len, n_neurons,
                device=self.device,
            )

            # Leaky integration with exponential decay
            state = torch.zeros(batch, n_neurons, device=self.device)

            for s in range(seq_len):
                for t in range(n_timesteps):
                    state = state * decay + spikes[:, s, t, :]
                features[:, s, :] = state

        elif self.config.decoding_type == DecodingType.WTA:
            # WTA: Competition during integration
            features = torch.zeros(
                batch, seq_len, n_neurons,
                device=self.device,
            )

            for s in range(seq_len):
                # Accumulate evidence
                evidence = spikes[:, s, :, :].sum(dim=1)  # [batch, n_neurons]

                # Apply lateral inhibition in neuron space
                # (actual WTA happens in token space during decoding)
                features[:, s, :] = evidence

        return features

    def _decode_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode integrated features to token logits using linear readout.

        Args:
            features: Integrated features [batch, seq_len, n_neurons]

        Returns:
            logits: Token logits [batch, seq_len, vocab_size]
        """
        # Store features for later learning
        self._last_features = features.detach()

        # Linear readout: logits = features @ W^T + bias
        logits = F.linear(features, self.readout_weights, self.readout_bias)

        # For WTA, apply competition in token space
        if self.config.decoding_type == DecodingType.WTA:
            logits = self._apply_wta(logits)

        return logits

    @torch.no_grad()
    def learn(
        self,
        target_ids: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update readout weights using the delta rule (Hebbian learning).

        This is a local learning rule that doesn't require backpropagation.
        Each synapse weight is updated based on:
        - Presynaptic activity (brain neuron firing rate)
        - Postsynaptic error (target - prediction)

        Delta Rule:
            prediction = softmax(W @ activity + bias)
            error = one_hot(target) - prediction
            W += learning_rate * outer(error, activity)
            bias += learning_rate * error

        Args:
            target_ids: Target token IDs [batch, seq_len] or [seq_len]
            features: Brain activity features [batch, seq_len, n_neurons]
                     If None, uses stored features from last forward pass

        Returns:
            Dictionary with learning metrics (error, weight_update_norm, etc.)
        """
        # Use stored features if not provided
        if features is None:
            features = self._last_features

        if features is None:
            return {"error": 0.0, "skipped": True}

        # Ensure target_ids has the right shape
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)  # [1, seq_len]

        batch, seq_len, n_neurons = features.shape
        vocab_size = self.config.vocab_size
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay

        # Flatten for easier processing: [batch * seq_len, n_neurons]
        features_flat = features.view(-1, n_neurons)
        target_flat = target_ids.view(-1)  # [batch * seq_len]

        # Only learn from valid positions (non-padding, etc.)
        n_samples = features_flat.shape[0]

        # Compute current predictions
        logits = F.linear(features_flat, self.readout_weights, self.readout_bias)
        predictions = F.softmax(logits, dim=-1)  # [n_samples, vocab_size]

        # Create one-hot targets
        targets_one_hot = torch.zeros_like(predictions)
        targets_one_hot.scatter_(1, target_flat.unsqueeze(1), 1.0)

        # Compute error: target - prediction
        error = targets_one_hot - predictions  # [n_samples, vocab_size]

        # Delta rule weight update: W += lr * error^T @ features
        # error: [n_samples, vocab_size], features: [n_samples, n_neurons]
        # weight_update: [vocab_size, n_neurons]
        weight_update = torch.matmul(error.T, features_flat) / n_samples

        # Apply weight decay (L2 regularization)
        weight_update = weight_update - weight_decay * self.readout_weights

        # Update weights
        self.readout_weights.data += lr * weight_update

        # Update bias: bias += lr * mean(error)
        bias_update = error.mean(dim=0)
        self.readout_bias.data += lr * bias_update

        # Track statistics
        mean_error = (1.0 - predictions.gather(1, target_flat.unsqueeze(1))).mean().item()
        update_norm = weight_update.norm().item()

        self.n_updates += 1
        self.total_error += mean_error

        return {
            "error": mean_error,
            "weight_update_norm": update_norm,
            "bias_update_norm": bias_update.norm().item(),
            "n_updates": self.n_updates.item(),
            "avg_error": (self.total_error / self.n_updates).item(),
        }

    def _apply_wta(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply winner-take-all competition.

        Args:
            logits: Token logits [batch, seq_len, vocab_size]

        Returns:
            wta_logits: Post-competition logits
        """
        # Iterative inhibition
        n_iterations = 3
        inhibition = getattr(self, 'inhibition', None)
        if inhibition is None:
            return logits

        for _ in range(n_iterations):
            # Get current winners
            probs = F.softmax(logits, dim=-1)

            # Apply lateral inhibition
            # inhibition: [vocab, vocab], probs: [batch, seq, vocab]
            inhibition_signal = torch.matmul(probs, inhibition.T)

            # Subtract inhibition from logits
            logits = logits - inhibition_signal

        return logits

    def sample(
        self,
        spikes: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tokens from spike patterns.

        Args:
            spikes: Spike patterns [batch, seq_len, n_timesteps, n_neurons]
            temperature: Sampling temperature
            top_k: Number of top tokens to consider (if set)
            top_p: Nucleus sampling probability (if set)

        Returns:
            tokens: Sampled token IDs [batch, seq_len]
            log_probs: Log probabilities of sampled tokens [batch, seq_len]
        """
        # Get logits
        logits = self(spikes)  # [batch, seq_len, vocab_size]
        logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits[indices_to_remove] = float('-inf')

        # Apply nucleus (top-p) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter back
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)

        # For each position, sample a token
        batch, seq_len, vocab_size = probs.shape
        probs_flat = probs.view(-1, vocab_size)
        tokens_flat = torch.multinomial(probs_flat, num_samples=1)
        tokens = tokens_flat.view(batch, seq_len)

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        sampled_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        return tokens, sampled_log_probs

    def greedy_decode(
        self,
        spikes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding: select most probable token.

        Args:
            spikes: Spike patterns [batch, seq_len, n_timesteps, n_neurons]

        Returns:
            tokens: Token IDs [batch, seq_len]
            probs: Probabilities of selected tokens [batch, seq_len]
        """
        logits = self(spikes)
        probs = F.softmax(logits, dim=-1)

        max_probs, tokens = probs.max(dim=-1)

        return tokens, max_probs

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get decoder diagnostics."""
        return {
            "decoding_type": self.config.decoding_type.value,
            "n_neurons": self.config.n_neurons,
            "vocab_size": self.config.vocab_size,
            "temperature": self.config.temperature,
            "learning_rate": self.config.learning_rate,
            "n_updates": int(self.n_updates),
            "avg_error": float(self.total_error / max(self.n_updates, 1)),
        }


class ConfidenceEstimator(nn.Module):
    """
    Estimate confidence of decoded tokens from spike patterns.

    Uses multiple signals:
    - Firing rate (high = confident)
    - Synchrony (correlated = confident)
    - Entropy of population activity
    """

    def __init__(self, n_neurons: int, device: str = "cpu"):
        super().__init__()
        self.n_neurons = n_neurons
        self.device_str = device

        # Learnable confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(n_neurons, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        spikes: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate confidence from spike patterns.

        Args:
            spikes: Spike patterns [batch, seq_len, n_timesteps, n_neurons]
            features: Integrated features [batch, seq_len, n_neurons]

        Returns:
            confidence: Confidence scores [batch, seq_len]
        """
        batch, seq_len, n_timesteps, n_neurons = spikes.shape

        # Feature 1: Mean firing rate
        mean_rate = spikes.mean(dim=(2, 3))  # [batch, seq_len]

        # Feature 2: Population synchrony (variance of spike counts)
        spike_counts = spikes.sum(dim=2)  # [batch, seq_len, n_neurons]
        synchrony = spike_counts.std(dim=-1) / (spike_counts.mean(dim=-1) + 1e-6)

        # Feature 3: Learned confidence from features
        learned_conf = self.confidence_net(features).squeeze(-1)  # [batch, seq_len]

        # Combine features
        confidence = 0.3 * mean_rate + 0.3 * (1 - synchrony) + 0.4 * learned_conf
        confidence = torch.clamp(confidence, 0, 1)

        return confidence


class StreamingDecoder(nn.Module):
    """
    Streaming spike decoder for real-time applications.

    Processes spikes as they arrive, maintaining state across timesteps.
    Outputs tokens when confidence threshold is reached.
    """

    def __init__(
        self,
        config: SpikeDecoderConfig,
        confidence_threshold: float = 0.8,
    ):
        super().__init__()
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(config.device)

        # Base decoder
        self.decoder = SpikeDecoder(config)

        # Confidence estimator
        self.confidence = ConfidenceEstimator(config.n_neurons, config.device)

        # Streaming state
        self.register_buffer(
            "accumulated_spikes",
            torch.zeros(1, 1, 0, config.n_neurons),
        )
        self.register_buffer(
            "evidence",
            torch.zeros(1, config.vocab_size),
        )

    def reset_state(self) -> None:
        """Reset streaming state."""
        self.accumulated_spikes = torch.zeros(
            1, 1, 0, self.config.n_neurons,
            device=self.device,
        )
        self.evidence.zero_()
        self.decoder.reset_state()

    def step(
        self,
        spikes: torch.Tensor,
    ) -> Tuple[Optional[int], float, torch.Tensor]:
        """
        Process one timestep of spikes.

        Args:
            spikes: Spike pattern for one timestep [n_neurons]

        Returns:
            token: Decoded token (if confident enough), else None
            confidence: Current confidence level
            probs: Current token probabilities
        """
        # Reshape spikes: [n_neurons] -> [1, 1, 1, n_neurons]
        spikes = spikes.view(1, 1, 1, -1)

        # Accumulate spikes
        self.accumulated_spikes = torch.cat([
            self.accumulated_spikes,
            spikes,
        ], dim=2)

        # Decode accumulated spikes
        logits, features = self.decoder(self.accumulated_spikes, return_features=True)
        probs = F.softmax(logits, dim=-1).squeeze(0).squeeze(0)  # [vocab_size]

        # Estimate confidence
        conf = self.confidence(
            self.accumulated_spikes,
            features,
        ).item()

        # Check if confident enough to emit token
        if conf >= self.confidence_threshold:
            token = probs.argmax().item()
            self.reset_state()  # Reset for next token
            return token, conf, probs

        return None, conf, probs
