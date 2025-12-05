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

3. CONFIDENCE ESTIMATION
   Spike patterns can indicate confidence:
   - Higher firing rates = more confident
   - Synchronous firing = strong evidence
   - Sparse firing = uncertain

Biological Basis:
- Motor cortex uses population vectors for movement direction
- Hippocampus uses pattern completion for recall
- Decision making involves accumulation to threshold

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecodingType(Enum):
    """Types of spike decoding strategies."""
    RATE = "rate"           # Count spikes, high count = high probability
    TEMPORAL = "temporal"   # First spike wins
    POPULATION = "population"  # Weighted population vote
    WTA = "wta"             # Winner-take-all competition


@dataclass 
class SpikeDecoderConfig:
    """Configuration for spike decoder.
    
    Attributes:
        n_neurons: Number of input neurons (from SNN output)
        vocab_size: Size of output vocabulary
        decoding_type: How to decode spikes to tokens
        
        # Temporal parameters
        integration_tau_ms: Time constant for spike integration
        dt_ms: Simulation timestep
        n_timesteps: Number of timesteps to decode from
        
        # Population decoding
        n_populations: Number of neural populations per token
        temperature: Softmax temperature for output probabilities
        
        # WTA parameters
        inhibition_strength: Lateral inhibition for WTA
        
        # Learning
        learnable_readout: Whether readout weights are learnable
        
        device: Computation device
    """
    n_neurons: int = 1024
    vocab_size: int = 50257
    decoding_type: DecodingType = DecodingType.POPULATION
    
    # Temporal
    integration_tau_ms: float = 20.0
    dt_ms: float = 1.0
    n_timesteps: int = 20
    
    # Population
    n_populations: int = 1
    temperature: float = 1.0
    
    # WTA
    inhibition_strength: float = 0.1
    
    # Learning
    learnable_readout: bool = True
    
    device: str = "cpu"
    
    @property
    def decay_factor(self) -> float:
        """Exponential decay factor for leaky integration."""
        return 1.0 - self.dt_ms / self.integration_tau_ms


class SpikeDecoder(nn.Module):
    """
    Decode spike patterns to token probabilities.
    
    This is the inverse of SpikeEncoder - it takes spike patterns and
    produces probability distributions over tokens.
    
    The decoder can work in several modes:
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
    """
    
    def __init__(self, config: SpikeDecoderConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Main readout layer
        if config.learnable_readout:
            # Learnable weight matrix: neurons -> tokens
            self.readout = nn.Linear(config.n_neurons, config.vocab_size)
            
            # Optional: Multi-layer readout for better expressivity
            self.readout_mlp = nn.Sequential(
                nn.Linear(config.n_neurons, config.n_neurons * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.n_neurons * 2, config.vocab_size),
            )
        else:
            # Fixed random readout (for testing)
            readout_weights = torch.randn(config.vocab_size, config.n_neurons)
            readout_weights = readout_weights / (config.n_neurons ** 0.5)
            self.register_buffer("fixed_readout", readout_weights)
            self.readout = None
            self.readout_mlp = None
        
        # Temporal integration state
        self.register_buffer(
            "integration_state",
            torch.zeros(1, config.n_neurons),
        )
        
        # Initialize based on decoding type
        self._init_decoder()
    
    def _init_decoder(self) -> None:
        """Initialize decoder-specific components."""
        config = self.config
        
        if config.decoding_type == DecodingType.TEMPORAL:
            # Track first spike times
            self.register_buffer(
                "first_spike_time",
                torch.full((1, config.n_neurons), float('inf')),
            )
            
        elif config.decoding_type == DecodingType.WTA:
            # Lateral inhibition weights
            inhibition = torch.ones(config.vocab_size, config.vocab_size)
            inhibition = inhibition - torch.eye(config.vocab_size)
            inhibition = inhibition * config.inhibition_strength
            self.register_buffer("inhibition", inhibition)
    
    def reset_state(self) -> None:
        """Reset temporal integration state."""
        self.integration_state.zero_()
        if hasattr(self, "first_spike_time"):
            self.first_spike_time.fill_(float('inf'))
    
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
        batch, seq_len, n_timesteps, n_neurons = spikes.shape
        
        # Step 1: Integrate spikes over time
        features = self._integrate_spikes(spikes)  # [batch, seq_len, n_neurons]
        
        # Step 2: Decode to token space
        logits = self._decode_features(features)  # [batch, seq_len, vocab_size]
        
        # Step 3: Apply temperature
        logits = logits / self.config.temperature
        
        if return_features:
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
        Decode integrated features to token logits.
        
        Args:
            features: Integrated features [batch, seq_len, n_neurons]
            
        Returns:
            logits: Token logits [batch, seq_len, vocab_size]
        """
        if self.config.learnable_readout:
            # Use learned readout
            logits = self.readout_mlp(features)
        else:
            # Fixed readout
            logits = F.linear(features, self.fixed_readout)
        
        # For WTA, apply competition in token space
        if self.config.decoding_type == DecodingType.WTA:
            logits = self._apply_wta(logits)
        
        return logits
    
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
        
        for _ in range(n_iterations):
            # Get current winners
            probs = F.softmax(logits, dim=-1)
            
            # Apply lateral inhibition
            # inhibition: [vocab, vocab], probs: [batch, seq, vocab]
            inhibition_signal = torch.matmul(probs, self.inhibition.T)
            
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
            "learnable_readout": self.config.learnable_readout,
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
    
    def reset(self) -> None:
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
            self.reset()  # Reset for next token
            return token, conf, probs
        
        return None, conf, probs
