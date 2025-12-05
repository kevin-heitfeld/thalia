"""
Spiking Language Model - Full language model using spiking neural networks.

This module integrates the encoder, decoder, position encoder, and core
SNN components into a complete language model.

Architecture:
=============

1. INPUT PROCESSING
   Text → Tokens → SDR → Spike Trains (with position encoding)

2. CORE PROCESSING
   Spike trains flow through:
   - Predictive Cortex (hierarchical prediction)
   - Scalable Spiking Attention (context aggregation)
   - Hippocampus (memory/retrieval)

3. OUTPUT GENERATION
   Processed spikes → Token probabilities → Text

Key Design Principles:
- Event-driven: Only process when spikes occur
- Local learning: No global backpropagation
- Biologically plausible: Based on real neural circuits
- Scalable: O(n) or O(n·k) operations where possible

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from language module
from thalia.language.encoder import (
    SpikeEncoder,
    SpikeEncoderConfig,
    EncodingType,
)
from thalia.language.decoder import (
    SpikeDecoder,
    SpikeDecoderConfig,
    DecodingType,
)
from thalia.language.position import (
    OscillatoryPositionEncoder,
    PositionEncoderConfig,
    PositionEncodingType,
)

# Import core components (conditional to avoid circular imports)
try:
    from thalia.core.predictive_coding import (
        HierarchicalPredictiveCoding,
        PredictiveCodingConfig,
    )
    from thalia.core.scalable_attention import (
        ScalableSpikingAttention,
        AttentionConfig,
        AttentionType,
    )
    HAS_PREDICTIVE = True
except ImportError:
    HAS_PREDICTIVE = False
    HierarchicalPredictiveCoding = None
    PredictiveCodingConfig = None
    ScalableSpikingAttention = None
    AttentionConfig = None
    AttentionType = None


@dataclass
class SpikingLanguageModelConfig:
    """Configuration for spiking language model.
    
    Attributes:
        vocab_size: Size of token vocabulary
        n_neurons: Number of neurons per layer
        n_layers: Number of processing layers
        n_heads: Number of attention heads (for attention layers)
        max_seq_len: Maximum sequence length
        n_timesteps: Number of timesteps per token
        
        # Encoding
        encoding_type: Type of spike encoding
        sparsity: Target sparsity for SDR
        
        # Position encoding
        position_type: Type of position encoding
        
        # Attention
        use_attention: Whether to use spiking attention
        attention_type: Type of spiking attention
        
        # Predictive coding
        use_predictive_coding: Whether to use predictive coding
        
        # Learning
        learning_rate: Learning rate for local rules
        use_eligibility_traces: Whether to use eligibility traces
        
        device: Computation device
    """
    vocab_size: int = 50257
    n_neurons: int = 1024
    n_layers: int = 4
    n_heads: int = 8
    max_seq_len: int = 1024
    n_timesteps: int = 20
    
    # Encoding
    encoding_type: EncodingType = EncodingType.SDR
    sparsity: float = 0.05
    
    # Position
    position_type: PositionEncodingType = PositionEncodingType.NESTED_GAMMA
    
    # Attention
    use_attention: bool = True
    attention_type: str = "gamma_phase"  # Will be converted to AttentionType
    
    # Predictive coding
    use_predictive_coding: bool = True
    
    # Learning
    learning_rate: float = 0.01
    use_eligibility_traces: bool = True
    
    device: str = "cpu"


class SpikingLanguageModel(nn.Module):
    """
    Complete spiking neural network language model.
    
    This model can process text and generate text using only
    spiking neural networks with local learning rules.
    
    Architecture Overview:
    ---------------------
    
    Input: Token IDs [batch, seq_len]
        ↓ (SpikeEncoder)
    Spike Patterns [batch, seq_len, n_timesteps, n_neurons]
        ↓ (+ OscillatoryPositionEncoder)
    Position-Encoded Spikes
        ↓ (ProcessingStack: Predictive Coding + Attention)
    Processed Spikes
        ↓ (SpikeDecoder)
    Token Logits [batch, seq_len, vocab_size]
        ↓ (Sampling)
    Output: Token IDs [batch, seq_len]
    
    Usage:
        model = SpikingLanguageModel(SpikingLanguageModelConfig(
            vocab_size=50000,
            n_neurons=1024,
            n_layers=4,
        ))
        
        # Forward pass
        token_ids = torch.tensor([[1, 42, 100, 5]])
        logits = model(token_ids)
        
        # Generate text
        generated = model.generate(prompt_ids, max_new_tokens=100)
    """
    
    def __init__(self, config: SpikingLanguageModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # 1. Token Encoder
        encoder_config = SpikeEncoderConfig(
            vocab_size=config.vocab_size,
            n_neurons=config.n_neurons,
            sparsity=config.sparsity,
            n_timesteps=config.n_timesteps,
            encoding_type=config.encoding_type,
            device=config.device,
        )
        self.encoder = SpikeEncoder(encoder_config)
        
        # 2. Position Encoder
        position_config = PositionEncoderConfig(
            n_neurons=config.n_neurons // 4,  # Smaller position encoding
            max_positions=config.max_seq_len,
            encoding_type=config.position_type,
            n_timesteps=config.n_timesteps,
            device=config.device,
        )
        self.position_encoder = OscillatoryPositionEncoder(position_config)
        
        # 3. Position mixing layer
        self.position_mixer = nn.Linear(
            config.n_neurons + config.n_neurons // 4,
            config.n_neurons,
            bias=False,
        )
        
        # 4. Processing Stack
        self._build_processing_stack()
        
        # 5. Token Decoder
        decoder_config = SpikeDecoderConfig(
            n_neurons=config.n_neurons,
            vocab_size=config.vocab_size,
            decoding_type=DecodingType.POPULATION,
            n_timesteps=config.n_timesteps,
            device=config.device,
        )
        self.decoder = SpikeDecoder(decoder_config)
        
        # 6. Eligibility traces for learning
        if config.use_eligibility_traces:
            self._init_eligibility_traces()
        
        # Move to device
        self.to(self.device)
    
    def _build_processing_stack(self) -> None:
        """Build the main processing layers."""
        config = self.config
        
        self.processing_layers = nn.ModuleList()
        
        for layer_idx in range(config.n_layers):
            layer_modules = nn.ModuleDict()
            
            # Predictive coding layer
            if config.use_predictive_coding and HAS_PREDICTIVE:
                pc_config = PredictiveCodingConfig(
                    n_input=config.n_neurons,
                    n_prediction=config.n_neurons,
                    n_error=config.n_neurons // 4,
                    learning_rate=config.learning_rate,
                    device=config.device,
                )
                layer_modules["predictive"] = HierarchicalPredictiveCoding(
                    layer_configs=[pc_config],
                )
            
            # Attention layer
            if config.use_attention and HAS_PREDICTIVE:
                # Map string to AttentionType if needed
                if AttentionType is not None:
                    att_type = AttentionType(config.attention_type)
                else:
                    att_type = config.attention_type
                    
                att_config = AttentionConfig(
                    n_neurons=config.n_neurons,
                    n_heads=config.n_heads,
                    attention_type=att_type,
                    device=config.device,
                )
                layer_modules["attention"] = ScalableSpikingAttention(att_config)
            
            # Simple feedforward if no special layers
            if len(layer_modules) == 0:
                layer_modules["feedforward"] = nn.Sequential(
                    nn.Linear(config.n_neurons, config.n_neurons * 2),
                    nn.GELU(),
                    nn.Linear(config.n_neurons * 2, config.n_neurons),
                )
            
            self.processing_layers.append(layer_modules)
    
    def _init_eligibility_traces(self) -> None:
        """Initialize eligibility traces for three-factor learning."""
        # Traces decay over time and are modulated by reward signal
        self.eligibility_traces: Dict[str, torch.Tensor] = {}
        self.trace_decay = 0.95
    
    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_spikes: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            token_ids: Token indices [batch, seq_len]
            position_ids: Optional position indices [batch, seq_len]
            return_spikes: Whether to return intermediate spike patterns
            
        Returns:
            logits: Token logits [batch, seq_len, vocab_size]
            spikes: (optional) Dictionary of spike patterns
        """
        batch, seq_len = token_ids.shape
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch, -1)
        
        # 1. Encode tokens to spikes
        content_spikes, sdr = self.encoder(token_ids, position_ids)
        # content_spikes: [batch, seq_len, n_timesteps, n_neurons]
        
        # 2. Get position encoding
        position_spikes = self.position_encoder(position_ids, as_spikes=True)
        # position_spikes: [batch, seq_len, n_timesteps, n_pos_neurons]
        
        # 3. Combine content and position
        combined_spikes = self._combine_content_position(content_spikes, position_spikes)
        # combined_spikes: [batch, seq_len, n_timesteps, n_neurons]
        
        # 4. Process through layers
        spikes_dict = {"input": combined_spikes}
        
        processed = combined_spikes
        for layer_idx, layer_modules in enumerate(self.processing_layers):
            processed = self._process_layer(processed, layer_modules, layer_idx)
            spikes_dict[f"layer_{layer_idx}"] = processed
        
        # 5. Decode to logits
        logits = self.decoder(processed)
        
        if return_spikes:
            return logits, spikes_dict
        return logits
    
    def _combine_content_position(
        self,
        content_spikes: torch.Tensor,
        position_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Combine content and position spike patterns."""
        batch, seq_len, n_timesteps, n_content = content_spikes.shape
        _, _, _, n_position = position_spikes.shape
        
        # Concatenate along neuron dimension
        combined = torch.cat([content_spikes, position_spikes], dim=-1)
        # combined: [batch, seq_len, n_timesteps, n_content + n_position]
        
        # Reshape for linear layer
        combined_flat = combined.view(batch * seq_len * n_timesteps, -1)
        mixed = self.position_mixer(combined_flat)
        mixed = mixed.view(batch, seq_len, n_timesteps, -1)
        
        # Convert back to spikes via threshold
        spikes = (mixed > 0.5).float()
        
        return spikes
    
    def _process_layer(
        self,
        spikes: torch.Tensor,
        layer_modules: nn.ModuleDict,
        layer_idx: int,
    ) -> torch.Tensor:
        """Process spikes through one layer."""
        batch, seq_len, n_timesteps, n_neurons = spikes.shape
        
        residual = spikes
        
        # Predictive coding
        if "predictive" in layer_modules:
            # Process through predictive coding
            # Need to reshape: [batch, seq, time, neurons] -> process each position
            output_spikes = torch.zeros_like(spikes)
            
            for s in range(seq_len):
                # Get spikes for this position
                pos_spikes = spikes[:, s, :, :]  # [batch, time, neurons]
                
                # Process through predictive coding (time as sequence)
                pc_output = layer_modules["predictive"](pos_spikes)
                
                # Get output spikes from the result
                if isinstance(pc_output, dict) and "output" in pc_output:
                    output_spikes[:, s, :, :] = pc_output["output"]
                else:
                    output_spikes[:, s, :, :] = pos_spikes  # Passthrough if failed
            
            spikes = output_spikes + residual * 0.1  # Residual connection
        
        # Attention
        if "attention" in layer_modules:
            # Reshape for attention: [batch, seq, time, neurons] -> [batch, time, seq, neurons]
            spikes_t = spikes.permute(0, 2, 1, 3)
            
            # Process each timestep
            attended = torch.zeros_like(spikes_t)
            for t in range(n_timesteps):
                step_spikes = spikes_t[:, t, :, :]  # [batch, seq, neurons]
                
                # Spiking attention
                att_output = layer_modules["attention"](step_spikes, step_spikes, step_spikes)
                if isinstance(att_output, tuple):
                    attended[:, t, :, :] = att_output[0]
                else:
                    attended[:, t, :, :] = att_output
            
            # Reshape back
            spikes = attended.permute(0, 2, 1, 3) + residual * 0.1
        
        # Feedforward
        if "feedforward" in layer_modules:
            # Simple feedforward on integrated spikes
            integrated = spikes.mean(dim=2)  # [batch, seq, neurons]
            ff_out = layer_modules["feedforward"](integrated)
            
            # Broadcast back to all timesteps and add
            ff_out = ff_out.unsqueeze(2)  # [batch, seq, 1, neurons]
            spikes = spikes + ff_out * 0.1
        
        return spikes
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            prompt_ids: Starting token IDs [batch, prompt_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            generated_ids: Full sequence including prompt [batch, prompt_len + generated]
        """
        self.eval()
        
        current_ids = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self(current_ids)  # [batch, seq_len, vocab_size]
            
            # Only need the last position
            last_logits = logits[:, -1, :] / temperature  # [batch, vocab_size]
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = last_logits < torch.topk(last_logits, top_k, dim=-1).values[:, -1:]
                last_logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                last_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for EOS (assuming 0 is EOS)
            if (next_token == 0).all():
                break
        
        return current_ids
    
    def learn_from_reward(
        self,
        reward: float,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Apply three-factor learning based on reward signal.
        
        This is how the model learns without backpropagation:
        - Eligibility traces mark recent synaptic activity
        - Reward modulates trace → weight change
        - Positive reward strengthens active synapses
        - Negative reward weakens them
        
        Args:
            reward: Reward signal (positive = good, negative = bad)
            token_ids: Optional token sequence that generated the reward
            
        Returns:
            stats: Learning statistics
        """
        if not hasattr(self, "eligibility_traces") or len(self.eligibility_traces) == 0:
            return {"error": "No eligibility traces available"}
        
        stats = {
            "reward": reward,
            "weight_updates": 0,
            "mean_trace": 0.0,
        }
        
        # Apply eligibility-modulated learning to all parameters
        lr = self.config.learning_rate
        
        for name, param in self.named_parameters():
            if name in self.eligibility_traces and param.requires_grad:
                trace = self.eligibility_traces[name]
                
                # Three-factor rule: Δw = η * reward * trace
                weight_update = lr * reward * trace
                param.data.add_(weight_update)
                
                stats["weight_updates"] += weight_update.abs().sum().item()
                stats["mean_trace"] += trace.abs().mean().item()
        
        # Decay traces
        for name in self.eligibility_traces:
            self.eligibility_traces[name] *= self.trace_decay
        
        return stats
    
    def update_eligibility(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        layer_name: str,
    ) -> None:
        """
        Update eligibility traces based on spike correlations.
        
        Called after each forward pass to update traces based on
        pre/post synaptic spike correlations.
        
        Args:
            pre_spikes: Presynaptic spikes [batch, n_pre]
            post_spikes: Postsynaptic spikes [batch, n_post]
            layer_name: Name of the layer (for trace storage)
        """
        # STDP-like eligibility: trace increases when pre fires before post
        # Simplified: just use outer product of spike rates
        pre_rate = pre_spikes.mean(dim=0)
        post_rate = post_spikes.mean(dim=0)
        
        # Create trace from outer product
        trace = torch.outer(pre_rate, post_rate)
        
        # Store or update trace
        if layer_name in self.eligibility_traces:
            self.eligibility_traces[layer_name] = (
                self.eligibility_traces[layer_name] * self.trace_decay + trace
            )
        else:
            self.eligibility_traces[layer_name] = trace
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics."""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "config": {
                "vocab_size": self.config.vocab_size,
                "n_neurons": self.config.n_neurons,
                "n_layers": self.config.n_layers,
                "n_heads": self.config.n_heads,
                "max_seq_len": self.config.max_seq_len,
                "n_timesteps": self.config.n_timesteps,
            },
            "parameters": {
                "total": n_params,
                "trainable": n_trainable,
            },
            "components": {
                "encoder": self.encoder.get_diagnostics(),
                "decoder": self.decoder.get_diagnostics(),
                "position": self.position_encoder.get_diagnostics(),
            },
            "has_predictive_coding": self.config.use_predictive_coding and HAS_PREDICTIVE,
            "has_attention": self.config.use_attention and HAS_PREDICTIVE,
        }


class MinimalSpikingLM(nn.Module):
    """
    Minimal spiking language model for testing.
    
    This is a simplified version without the full predictive coding
    and attention stack - useful for testing the encoding/decoding
    pipeline.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        n_neurons: int = 256,
        n_timesteps: int = 10,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        
        # Simple encoder
        self.encoder = SpikeEncoder(SpikeEncoderConfig(
            vocab_size=vocab_size,
            n_neurons=n_neurons,
            n_timesteps=n_timesteps,
            encoding_type=EncodingType.SDR,
            device=device,
        ))
        
        # Simple processing layer
        self.process = nn.Sequential(
            nn.Linear(n_neurons, n_neurons * 2),
            nn.ReLU(),
            nn.Linear(n_neurons * 2, n_neurons),
        )
        
        # Simple decoder
        self.decoder = SpikeDecoder(SpikeDecoderConfig(
            n_neurons=n_neurons,
            vocab_size=vocab_size,
            n_timesteps=n_timesteps,
            device=device,
        ))
        
        self.to(self.device)
    
    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        # Encode
        spikes, _ = self.encoder(token_ids)
        
        # Process (integrate spikes first)
        integrated = spikes.mean(dim=2)  # [batch, seq, neurons]
        processed = self.process(integrated)
        
        # Convert back to spikes for decoder
        n_timesteps = spikes.shape[2]
        processed_spikes = processed.unsqueeze(2).expand(-1, -1, n_timesteps, -1)
        processed_spikes = (torch.rand_like(processed_spikes) < torch.sigmoid(processed_spikes)).float()
        
        # Decode
        logits = self.decoder(processed_spikes)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """Simple greedy generation."""
        self.eval()
        
        current_ids = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            logits = self(current_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return current_ids
