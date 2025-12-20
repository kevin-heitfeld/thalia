"""
Language Model Integration - Connecting language to the unified brain.

This module provides utilities for integrating language processing
with DynamicBrain. Instead of a standalone language model,
language is just another sensory modality feeding into the unified brain.

Architecture:
============

    Text Input
         │
         ▼
    ┌─────────────────┐
    │ LanguagePathway │  (from thalia.pathways)
    │   Token → SDR   │
    │   → Spikes      │
    └───────┬─────────┘
            │
            ▼
    ┌──────────────────┐
    │ DynamicBrain     │  (our unified brain)
    │  Cortex → Hippo  │
    │  → PFC → Action  │
    └───────┬──────────┘
            │
            ▼
    ┌──────────────┐
    │ SpikeDecoder │  (for language output)
    │ Spikes → Text│
    └──────────────┘

Usage:
======

    from thalia.core.dynamic_brain import DynamicBrain, BrainBuilder
    from thalia.language import LanguageBrainInterface

    # Create brain with language interface
    brain = BrainBuilder.preset("default", global_config)
    lang_interface = LanguageBrainInterface(brain)

    # Process text through the brain
    result = lang_interface.process_text("Hello world")

    # Generate text from brain state
    generated = lang_interface.generate(prompt_ids, max_tokens=50)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.mixins import ConfigurableMixin

# Import encoding/decoding components
from thalia.language.encoder import (
    SpikeEncoder,
    SpikeEncoderConfig,
)
from thalia.language.decoder import (
    SpikeDecoder,
    SpikeDecoderConfig,
)
from thalia.language.position import (
    OscillatoryPositionEncoder,
    PositionEncoderConfig,
)

# Type checking imports
if TYPE_CHECKING:
    from thalia.core.dynamic_brain import DynamicBrain
    from thalia.config import ThaliaConfig


@dataclass
class LanguageInterfaceConfig:
    """Internal configuration for LanguageBrainInterface.

    This is NOT a user-facing API. Users should use ThaliaConfig for all configuration.
    This class is an internal, flattened representation of language-related parameters
    extracted from ThaliaConfig for convenience within LanguageBrainInterface.

    Typically created automatically via:
    ``LanguageBrainInterface.from_thalia_config(brain, config)``

    Attributes:
        vocab_size: Size of token vocabulary
        n_timesteps: Timesteps per token for spike encoding
        sparsity: SDR sparsity
        max_seq_len: Maximum sequence length
        brain_input_size: Size expected by brain's cortex input
        device: Computation device
    """
    vocab_size: int = 50257
    n_timesteps: int = 20
    sparsity: float = 0.05
    max_seq_len: int = 1024
    brain_input_size: int = 256
    device: str = "cpu"


class LanguageBrainInterface(ConfigurableMixin, nn.Module):
    """
    Interface between language and DynamicBrain.

    This class handles:
    1. Encoding text tokens to spikes (via SpikeEncoder)
    2. Feeding spikes through the brain's processing pipeline
    3. Decoding brain output to token probabilities (via SpikeDecoder)

    The key insight is that language is just another sensory input -
    once converted to spikes, the brain processes it like any other modality.

    Usage:
        brain = DynamicBrain.from_thalia_config(config)
        lang_interface = LanguageBrainInterface(brain, LanguageInterfaceConfig())

        # Process text
        token_ids = tokenizer.encode("Hello world")
        result = lang_interface.process_tokens(token_ids)

        # Generate continuation
        generated = lang_interface.generate(token_ids, max_new_tokens=50)
    """

    # For ConfigurableMixin - specifies how to extract config from ThaliaConfig
    CONFIG_CONVERTER_METHOD = "to_language_interface_config"

    @classmethod
    def from_thalia_config(
        cls,
        brain: "DynamicBrain",
        config: "ThaliaConfig",
    ) -> "LanguageBrainInterface":
        """Create interface from ThaliaConfig.

        This overrides ConfigurableMixin.from_thalia_config to accept brain
        as a required positional argument.

        Args:
            brain: Instantiated DynamicBrain
            config: ThaliaConfig with all settings

        Returns:
            LanguageBrainInterface instance
        """
        # Extract language interface config from ThaliaConfig
        interface_config = LanguageInterfaceConfig(
            vocab_size=config.global_.vocab_size,
            n_timesteps=config.language.encoding.n_timesteps,
            sparsity=config.language.encoding.get_sparsity(config.global_),
            max_seq_len=config.language.position.max_positions,
            brain_input_size=config.brain.sizes.input_size,
            device=config.global_.device,
        )

        return cls(brain, interface_config)

    def __init__(
        self,
        brain: "DynamicBrain",
        config: Optional[LanguageInterfaceConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = LanguageInterfaceConfig(
                brain_input_size=brain.config.input_size,
                device=brain.config.device,
            )

        self.config = config
        self.device = torch.device(config.device)

        # Token → Spike encoder
        encoder_config = SpikeEncoderConfig(
            vocab_size=config.vocab_size,
            n_neurons=config.brain_input_size,
            n_timesteps=config.n_timesteps,
            sparsity=config.sparsity,
            device=config.device,
        )
        self.encoder = SpikeEncoder(encoder_config)

        # Position encoder
        pos_config = PositionEncoderConfig(
            n_neurons=config.brain_input_size // 4,
            max_positions=config.max_seq_len,
            n_timesteps=config.n_timesteps,
            device=config.device,
        )
        self.position_encoder = OscillatoryPositionEncoder(pos_config)

        # Combine content + position
        self.position_mixer = nn.Linear(
            config.brain_input_size + config.brain_input_size // 4,
            config.brain_input_size,
            bias=False,
        )

        # Spike → Token decoder
        # Decodes from PFC output (which holds the processed representation)
        decoder_config = SpikeDecoderConfig(
            n_neurons=brain.config.pfc_size,
            vocab_size=config.vocab_size,
            n_timesteps=config.n_timesteps,
            device=config.device,
        )
        self.decoder = SpikeDecoder(decoder_config)

        # Buffer for collecting brain output spikes
        self.output_buffer: List[torch.Tensor] = []

        # Move all submodules to device BEFORE storing brain
        # (brain is not a submodule and causes hashability issues)
        self.to(self.device)

        # Store brain reference AFTER .to() to avoid hashability issues
        self._brain = brain

    def forward(
        self,
        token_ids: torch.Tensor,
        n_timesteps_per_token: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Forward pass - delegates to process_tokens for nn.Module compatibility."""
        return self.process_tokens(token_ids, n_timesteps_per_token)

    def process_tokens(
        self,
        token_ids: torch.Tensor,
        n_timesteps_per_token: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a sequence of tokens through the brain.

        Each token is encoded to spikes and fed through brain.process_sample().

        Args:
            token_ids: Token indices [batch, seq_len] or [seq_len]
            n_timesteps_per_token: Timesteps to process each token

        Returns:
            result: Dictionary with processing results
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        batch, seq_len = token_ids.shape
        n_timesteps = n_timesteps_per_token or self.config.n_timesteps

        # Encode all tokens to spikes
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch, -1)
        content_spikes, sdr = self.encoder(token_ids, position_ids)
        # content_spikes: [batch, seq_len, n_timesteps, n_neurons]

        # Add position encoding
        pos_spikes = self.position_encoder(position_ids, as_spikes=True)
        combined = self._combine_content_position(content_spikes, pos_spikes)

        # Clear output buffer
        self.output_buffer = []

        # Process each token through the brain
        all_results = []
        for t in range(seq_len):
            # Get spikes for this token
            token_spikes = combined[:, t, :, :]  # [batch, n_timesteps, n_neurons]

            # Sum over timesteps for single-vector input to brain
            # (brain expects [input_size] per call)
            # Scale by gain to reach neuron threshold (SDR spikes ~0.05, need ~1.0+ input)
            token_input = token_spikes.sum(dim=1)  # [batch, input_size]
            # ADR-005: Remove batch dimension if batch_size=1
            if batch == 1:
                token_input = token_input.squeeze(0)  # [input_size]
            token_input = token_input * 2.0  # Scale AFTER squeeze

            # Process through brain
            # Gamma slot auto-advances in hippocampus - no explicit position needed
            result = self._brain.process_sample(token_input, n_timesteps=n_timesteps)
            all_results.append(result)            # Collect PFC output for decoding
            if hasattr(self._brain, '_last_pfc_output') and self._brain._last_pfc_output is not None:
                self.output_buffer.append(self._brain._last_pfc_output.clone())

        return {
            "n_tokens": seq_len,
            "results": all_results,
            "sdr_sparsity": sdr.mean().item(),
        }

    def _combine_content_position(
        self,
        content_spikes: torch.Tensor,
        position_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Combine content and position encodings."""
        # Unpack shape (only need for concatenation dimension check)
        _ = content_spikes.shape  # batch, seq_len, n_timesteps, n_content

        # Concatenate
        combined = torch.cat([content_spikes, position_spikes], dim=-1)

        # Mix
        shape = combined.shape
        combined_flat = combined.view(-1, shape[-1])
        mixed = self.position_mixer(combined_flat)
        mixed = mixed.view(shape[0], shape[1], shape[2], -1)

        # Re-binarize
        return (mixed > 0.5).float()

    def decode_output(
        self,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Decode collected brain outputs to token probabilities.

        Returns:
            logits: Token logits [n_tokens, vocab_size]
        """
        if len(self.output_buffer) == 0:
            raise ValueError("No output collected. Call process_tokens first.")

        # Stack outputs: [n_tokens, pfc_size]
        pfc_outputs = torch.stack(self.output_buffer, dim=0)

        # Expand to fake timestep dimension for decoder
        pfc_outputs = pfc_outputs.unsqueeze(0)  # [1, n_tokens, pfc_size]

        # Create fake spike patterns (treat pfc_output as firing rates)
        n_timesteps = self.config.n_timesteps
        fake_spikes = pfc_outputs.unsqueeze(2).expand(-1, -1, n_timesteps, -1)
        fake_spikes = (torch.rand_like(fake_spikes) < torch.sigmoid(fake_spikes)).float()

        # Decode
        logits = self.decoder(fake_spikes)  # [1, n_tokens, vocab_size]
        logits = logits.squeeze(0) / temperature

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively using the brain.

        Args:
            prompt_ids: Starting token IDs [seq_len] or [1, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            generated_ids: Full sequence [1, prompt_len + generated]
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        current_ids = prompt_ids.clone()

        # Process prompt
        self.process_tokens(current_ids)

        for _ in range(max_new_tokens):
            # Get logits from brain state
            logits = self.decode_output(temperature)
            last_logits = logits[-1, :]  # [vocab_size]

            # Apply filtering
            if top_k is not None:
                indices_to_remove = last_logits < torch.topk(last_logits, top_k).values[-1]
                last_logits[indices_to_remove] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                last_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append and process
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

            # Process just the new token
            self.process_tokens(next_token.unsqueeze(0))

            # Check for EOS
            if next_token.item() == 0:  # Assuming 0 is EOS
                break

        return current_ids

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get interface diagnostics."""
        return {
            "vocab_size": self.config.vocab_size,
            "brain_input_size": self.config.brain_input_size,
            "n_timesteps": self.config.n_timesteps,
            "buffer_size": len(self.output_buffer),
            "encoder": self.encoder.get_diagnostics(),
            "decoder": self.decoder.get_diagnostics(),
        }
