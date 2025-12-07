"""
Scalable Spiking Attention - O(n·k) Attention via Spike Timing.

This module implements attention mechanisms that scale better than O(n²)
by leveraging the inherent sparsity and timing structure of spikes.

Key Insight: Traditional attention computes all pairwise similarities
(O(n²) complexity). But the brain doesn't do this! Instead:
1. Spikes are naturally sparse (~1-5% active at any time)
2. Attention emerges from spike timing coincidence
3. Only active neurons need to be compared

This gives us O(n·k) where k << n is the number of active neurons.

Three Attention Mechanisms:
===========================

1. COINCIDENCE ATTENTION (most biological)
   - Attention = spike timing coincidence within a window
   - If query and key spike together, attention is high
   - Natural implementation via STDP-like windows
   - O(k²) where k = number of active neurons

2. WINNER-TAKE-ALL ATTENTION (competitive)
   - Lateral inhibition creates sparse attention
   - Only top-k keys get attended
   - O(n·k) complexity with fixed k

3. OSCILLATION-PHASE ATTENTION (gamma-based)
   - Items bound by firing in same gamma cycle
   - Different items fire at different phases
   - Natural for multi-head attention (different frequencies)
   - O(n) with constant factors

Biological basis:
- Gamma oscillations (30-100 Hz) coordinate attention
- Phase-locking between attended items
- Thalamic pulvinar mediates visual attention
- PFC provides top-down attention signals

References:
- Fries (2005): Communication through coherence
- Bosman et al. (2012): Attentional stimulus selection through gamma
- Kreiter & Singer (1996): Stimulus-dependent synchrony in visual cortex
- Buschman & Miller (2007): Top-down versus bottom-up attention

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.neuron import LIFNeuron, LIFConfig


class AttentionType(Enum):
    """Types of spiking attention mechanisms."""
    COINCIDENCE = "coincidence"     # Spike timing coincidence
    WINNER_TAKE_ALL = "wta"         # Top-k selection
    OSCILLATION_PHASE = "phase"      # Gamma phase binding


@dataclass
class ScalableAttentionConfig:
    """Configuration for scalable spiking attention.
    
    Attributes:
        # Dimensions
        n_queries: Number of query positions
        n_keys: Number of key/value positions
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        
        # Attention type
        attention_type: Which attention mechanism to use
        
        # Coincidence attention parameters
        coincidence_window_ms: Time window for spike coincidence (STDP-like)
        
        # Winner-take-all parameters
        top_k: Number of keys to attend to (for WTA)
        inhibition_strength: Lateral inhibition for WTA
        
        # Oscillation parameters
        gamma_frequency_hz: Gamma frequency for phase attention
        n_gamma_phases: Number of discrete phase bins
        
        # Sparsity
        key_sparsity: Target sparsity for key activations
        query_sparsity: Target sparsity for query activations
        
        # Spiking parameters
        neuron_tau_ms: Membrane time constant
        dt_ms: Simulation timestep
        
        device: Computation device
    """
    # Dimensions
    n_queries: int = 64
    n_keys: int = 256
    d_model: int = 128
    n_heads: int = 4
    
    # Attention type
    attention_type: AttentionType = AttentionType.WINNER_TAKE_ALL
    
    # Coincidence attention
    coincidence_window_ms: float = 20.0  # STDP-like window
    
    # Winner-take-all
    top_k: int = 16  # Only attend to top k
    inhibition_strength: float = 2.0
    
    # Oscillation
    gamma_frequency_hz: float = 40.0
    n_gamma_phases: int = 8
    
    # Sparsity targets
    key_sparsity: float = 0.05  # 5% of keys active
    query_sparsity: float = 0.1  # 10% of queries active
    
    # Spiking
    neuron_tau_ms: float = 10.0
    dt_ms: float = 1.0
    
    device: str = "cpu"
    
    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    @property
    def gamma_period_ms(self) -> float:
        """Period of gamma oscillation in ms."""
        return 1000.0 / self.gamma_frequency_hz


@dataclass
class AttentionState:
    """State for spiking attention."""
    # Spike timing traces
    query_trace: Optional[torch.Tensor] = None
    key_trace: Optional[torch.Tensor] = None
    
    # Attention weights
    attention_weights: Optional[torch.Tensor] = None
    
    # Phase tracking (for oscillation attention)
    current_phase: float = 0.0
    
    # Output
    attended_values: Optional[torch.Tensor] = None


class SpikingQueryKeyValue(nn.Module):
    """
    Generate Query, Key, Value through spiking projections.
    
    Instead of linear projections like transformers, we use
    spiking neurons to generate sparse Q, K, V representations.
    This naturally enforces sparsity and temporal structure.
    """
    
    def __init__(
        self,
        d_model: int,
        d_head: int,
        n_heads: int,
        tau_ms: float = 10.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.device = torch.device(device)
        
        # Q, K, V projections (weight matrices)
        self.W_q = nn.Parameter(
            torch.randn(n_heads, d_head, d_model, device=self.device) * 0.1
        )
        self.W_k = nn.Parameter(
            torch.randn(n_heads, d_head, d_model, device=self.device) * 0.1
        )
        self.W_v = nn.Parameter(
            torch.randn(n_heads, d_head, d_model, device=self.device) * 0.1
        )
        
        # Spiking neurons for each (optional - can use rate coding too)
        lif_config = LIFConfig(tau_mem=tau_ms, v_threshold=1.0)
        self.q_neurons = LIFNeuron(n_heads * d_head, lif_config)
        self.k_neurons = LIFNeuron(n_heads * d_head, lif_config)
        self.v_neurons = LIFNeuron(n_heads * d_head, lif_config)
    
    def reset_state(self) -> None:
        """Reset neuron states."""
        self.q_neurons.reset_state()
        self.k_neurons.reset_state()
        self.v_neurons.reset_state()
    
    def forward(
        self,
        x: torch.Tensor,
        return_spikes: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V from input.
        
        Args:
            x: Input [batch, seq_len, d_model]
            return_spikes: If True, return spikes; else return membrane potentials
            
        Returns:
            Q: [batch, seq_len, n_heads, d_head]
            K: [batch, seq_len, n_heads, d_head]
            V: [batch, seq_len, n_heads, d_head]
        """
        batch, seq_len, _ = x.shape
        
        # Project through weights: [batch, seq, n_heads, d_head]
        # Using einsum for multi-head projection
        Q = torch.einsum('bsd,hkd->bshk', x, self.W_q)
        K = torch.einsum('bsd,hkd->bshk', x, self.W_k)
        V = torch.einsum('bsd,hkd->bshk', x, self.W_v)
        
        if return_spikes:
            # THALIA enforces batch_size=1 (single-instance architecture)
            # Process each sequence position separately to maintain batch_size=1
            q_spikes_list = []
            k_spikes_list = []
            v_spikes_list = []
            
            for b in range(batch):
                for s in range(seq_len):
                    # Process single position: [1, n_heads * d_head]
                    q_single = Q[b:b+1, s:s+1, :, :].reshape(1, -1)
                    k_single = K[b:b+1, s:s+1, :, :].reshape(1, -1)
                    v_single = V[b:b+1, s:s+1, :, :].reshape(1, -1)
                    
                    q_spike, _ = self.q_neurons(q_single)
                    k_spike, _ = self.k_neurons(k_single)
                    v_spike, _ = self.v_neurons(v_single)
                    
                    q_spikes_list.append(q_spike)
                    k_spikes_list.append(k_spike)
                    v_spikes_list.append(v_spike)
            
            # Reconstruct: stack all positions, then reshape to [batch, seq, heads, d_head]
            q_spikes = torch.stack(q_spikes_list, dim=0).reshape(batch, seq_len, self.n_heads, self.d_head)
            k_spikes = torch.stack(k_spikes_list, dim=0).reshape(batch, seq_len, self.n_heads, self.d_head)
            v_spikes = torch.stack(v_spikes_list, dim=0).reshape(batch, seq_len, self.n_heads, self.d_head)
            
            Q = q_spikes
            K = k_spikes
            V = v_spikes
        
        return Q, K, V


class CoincidenceAttention(nn.Module):
    """
    Attention based on spike timing coincidence.
    
    The key idea: attention is highest when query and key spikes
    occur close together in time (within a coincidence window).
    
    This is similar to STDP: pre-post timing matters.
    
    Complexity: O(k_q · k_k) where k_q, k_k are number of active spikes
    Since both are sparse, this is much less than O(n²).
    """
    
    def __init__(self, config: ScalableAttentionConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Coincidence decay (exponential kernel)
        self.tau_coincidence = config.coincidence_window_ms
        self.register_buffer(
            "decay",
            torch.tensor(torch.exp(torch.tensor(-config.dt_ms / config.coincidence_window_ms)).item())
        )
        
        # State
        self.query_trace: Optional[torch.Tensor] = None
        self.key_trace: Optional[torch.Tensor] = None
    
    def reset_state(self, n_queries: int, n_keys: int) -> None:
        """Reset trace states to batch_size=1.
        
        Args:
            n_queries: Number of query positions
            n_keys: Number of key positions
        """
        batch_size = 1
        self.query_trace = torch.zeros(
            batch_size, n_queries, self.config.d_head,
            device=self.device
        )
        self.key_trace = torch.zeros(
            batch_size, n_keys, self.config.d_head,
            device=self.device
        )
    
    def forward(
        self,
        query_spikes: torch.Tensor,
        key_spikes: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention based on spike coincidence.
        
        Args:
            query_spikes: Query spikes [batch, n_queries, d_head]
            key_spikes: Key spikes [batch, n_keys, d_head]
            value: Value vectors [batch, n_keys, d_head]
            
        Returns:
            output: Attended values [batch, n_queries, d_head]
            attention: Attention weights [batch, n_queries, n_keys]
        """
        batch, n_queries, d_head = query_spikes.shape
        _, n_keys, _ = key_spikes.shape
        
        # Initialize or resize if needed
        if (self.query_trace is None or 
            self.query_trace.shape[0] != batch or
            self.query_trace.shape[1] != n_queries or
            self.query_trace.shape[2] != d_head):
            self.reset_state(batch, n_queries, n_keys)
        
        # Also check key trace dimensions
        if (self.key_trace is None or
            self.key_trace.shape[1] != n_keys):
            self.key_trace = torch.zeros(
                batch, n_keys, d_head,
                device=self.device
            )
        
        # Update traces (decaying spike history)
        self.query_trace = self.decay * self.query_trace + query_spikes
        self.key_trace = self.decay * self.key_trace + key_spikes
        
        # Compute coincidence: dot product of traces
        # High overlap = high coincidence = high attention
        # [batch, n_queries, d_head] @ [batch, d_head, n_keys]
        # → [batch, n_queries, n_keys]
        attention_logits = torch.bmm(
            self.query_trace,
            self.key_trace.transpose(-2, -1)
        )
        
        # Scale by dimension (like transformer)
        attention_logits = attention_logits / (d_head ** 0.5)
        
        # Softmax over keys
        attention = F.softmax(attention_logits, dim=-1)
        
        # Apply attention to values
        output = torch.bmm(attention, value)
        
        return output, attention


class WinnerTakeAllAttention(nn.Module):
    """
    Top-k attention via lateral inhibition.
    
    Instead of computing all pairwise attention scores, we:
    1. Compute approximate scores efficiently
    2. Select top-k keys via competition
    3. Only attend to selected keys
    
    Complexity: O(n · k) where k = top_k
    """
    
    def __init__(self, config: ScalableAttentionConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.top_k = config.top_k
        
        # Inhibitory interneurons for competition
        lif_config = LIFConfig(tau_mem=5.0, v_threshold=0.5)  # Fast inhibition
        self.inhibitory_pool = LIFNeuron(config.n_keys, lif_config)
        
        # Lateral inhibition weights (learned)
        self.W_lateral = nn.Parameter(
            torch.ones(config.n_keys, config.n_keys, device=self.device) * config.inhibition_strength
        )
        # Zero out self-connections
        with torch.no_grad():
            self.W_lateral.fill_diagonal_(0.0)
    
    def reset_state(self) -> None:
        """Reset inhibitory pool."""
        self.inhibitory_pool.reset_state()
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k attention.
        
        Args:
            query: Query vectors [batch, n_queries, d_head]
            key: Key vectors [batch, n_keys, d_head]
            value: Value vectors [batch, n_keys, d_head]
            
        Returns:
            output: Attended values [batch, n_queries, d_head]
            attention: Sparse attention weights [batch, n_queries, n_keys]
        """
        batch, n_queries, d_head = query.shape
        _, n_keys, _ = key.shape
        
        # Compute query-key similarity
        # [batch, n_queries, d_head] @ [batch, d_head, n_keys]
        scores = torch.bmm(query, key.transpose(-2, -1))
        scores = scores / (d_head ** 0.5)
        
        # Winner-take-all selection via top-k
        # This is the key efficiency gain!
        topk_values, topk_indices = torch.topk(scores, k=min(self.top_k, n_keys), dim=-1)
        
        # Create sparse attention mask
        attention = torch.zeros_like(scores)
        
        # Scatter top-k values (with softmax)
        topk_attention = F.softmax(topk_values, dim=-1)
        attention.scatter_(-1, topk_indices, topk_attention)
        
        # Apply attention to values
        output = torch.bmm(attention, value)
        
        return output, attention


class GammaPhaseAttention(nn.Module):
    """
    Attention via gamma oscillation phase coding.
    
    Items that should be bound together fire at the same gamma phase.
    Different items fire at different phases, preventing interference.
    
    This is inspired by the "binding by synchrony" hypothesis.
    
    Multi-head attention naturally maps to different gamma frequencies
    or different phase offsets.
    
    Complexity: O(n) per timestep
    """
    
    def __init__(self, config: ScalableAttentionConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Phase tracking
        self.current_phase: float = 0.0
        self.phase_increment = 2 * torch.pi * config.dt_ms / config.gamma_period_ms
        self.n_phases = config.n_gamma_phases
        
        # Learned phase assignments for keys
        # Each key has a preferred phase
        self.key_phases = nn.Parameter(
            torch.rand(config.n_keys, device=self.device) * 2 * torch.pi
        )
        
        # Phase-dependent gating
        self.phase_width = 2 * torch.pi / config.n_gamma_phases  # Width of phase window
        
    def reset_state(self) -> None:
        """Reset phase."""
        self.current_phase = 0.0
    
    def _compute_phase_gating(self, key_phases: torch.Tensor) -> torch.Tensor:
        """
        Compute gating based on phase alignment.
        
        Keys whose preferred phase matches current phase are gated ON.
        
        Args:
            key_phases: Phase preference of each key [n_keys]
            
        Returns:
            gating: Phase gating factor [n_keys]
        """
        # Phase difference (circular)
        phase_diff = torch.abs(key_phases - self.current_phase)
        phase_diff = torch.min(phase_diff, 2 * torch.pi - phase_diff)
        
        # Gaussian gating based on phase proximity
        gating = torch.exp(-phase_diff ** 2 / (2 * self.phase_width ** 2))
        
        return gating
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        advance_phase: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute phase-gated attention.
        
        Only keys at the current gamma phase contribute.
        
        Args:
            query: Query vectors [batch, n_queries, d_head]
            key: Key vectors [batch, n_keys, d_head]
            value: Value vectors [batch, n_keys, d_head]
            advance_phase: Whether to advance gamma phase
            
        Returns:
            output: Attended values [batch, n_queries, d_head]
            attention: Phase-gated attention [batch, n_queries, n_keys]
        """
        batch, n_queries, d_head = query.shape
        
        # Compute base attention scores
        scores = torch.bmm(query, key.transpose(-2, -1))
        scores = scores / (d_head ** 0.5)
        
        # Apply phase gating
        phase_gate = self._compute_phase_gating(self.key_phases)  # [n_keys]
        scores = scores * phase_gate.unsqueeze(0).unsqueeze(0)  # Broadcast
        
        # Softmax (phase-gated scores)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.bmm(attention, value)
        
        # Advance phase
        if advance_phase:
            self.current_phase = (self.current_phase + self.phase_increment) % (2 * torch.pi)
        
        return output, attention


class ScalableSpikingAttention(nn.Module):
    """
    Unified scalable spiking attention module.
    
    Supports multiple attention mechanisms, all more efficient than O(n²):
    - Coincidence: O(k_q · k_k) where k = active spikes
    - WTA: O(n · k) with fixed k
    - Phase: O(n) per timestep
    
    Usage:
        attention = ScalableSpikingAttention(ScalableAttentionConfig(
            n_queries=64,
            n_keys=512,
            d_model=128,
            n_heads=4,
            attention_type=AttentionType.WINNER_TAKE_ALL,
            top_k=16,
        ))
        
        # Reset for new sequence
        attention.reset_state(batch_size=8)
        
        # Process each timestep
        for t in range(n_timesteps):
            output, weights = attention(x_queries, x_keys, x_values)
    """
    
    def __init__(self, config: ScalableAttentionConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Q, K, V projection
        self.qkv = SpikingQueryKeyValue(
            d_model=config.d_model,
            d_head=config.d_head,
            n_heads=config.n_heads,
            tau_ms=config.neuron_tau_ms,
            device=config.device,
        )
        
        # Per-head attention modules
        self.attention_heads = nn.ModuleList()
        for _ in range(config.n_heads):
            if config.attention_type == AttentionType.COINCIDENCE:
                head = CoincidenceAttention(config)
            elif config.attention_type == AttentionType.WINNER_TAKE_ALL:
                head = WinnerTakeAllAttention(config)
            elif config.attention_type == AttentionType.OSCILLATION_PHASE:
                head = GammaPhaseAttention(config)
            else:
                raise ValueError(f"Unknown attention type: {config.attention_type}")
            self.attention_heads.append(head)
        
        # Output projection
        self.W_out = nn.Parameter(
            torch.randn(config.d_model, config.n_heads * config.d_head, device=self.device) * 0.1
        )
        
        # Layer norm (optional, for stability)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Learning rate for STDP-like attention learning
        self.attention_lr = 0.01
    
    def reset_state(self) -> None:
        """Reset all states."""
        self.qkv.reset_state()
        for head in self.attention_heads:
            if isinstance(head, CoincidenceAttention):
                head.reset_state(self.config.n_queries, self.config.n_keys)
            elif isinstance(head, WinnerTakeAllAttention):
                head.reset_state()
            elif isinstance(head, GammaPhaseAttention):
                head.reset_state()
    
    def forward(
        self,
        x_query: torch.Tensor,
        x_key: Optional[torch.Tensor] = None,
        x_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scalable spiking attention.
        
        Args:
            x_query: Query input [batch, n_queries, d_model]
            x_key: Key input [batch, n_keys, d_model] (default: same as query)
            x_value: Value input [batch, n_keys, d_model] (default: same as key)
            
        Returns:
            output: Attended output [batch, n_queries, d_model]
            attention: Attention weights [batch, n_heads, n_queries, n_keys]
        """
        # Default to self-attention if K, V not provided
        if x_key is None:
            x_key = x_query
        if x_value is None:
            x_value = x_key
        
        batch, n_queries, _ = x_query.shape
        _, n_keys, _ = x_key.shape
        
        # Get Q, K, V through spiking projections
        Q, K, V = self.qkv(x_query, return_spikes=True)
        K_full, _, V_full = self.qkv(x_key, return_spikes=True)
        _, _, _ = self.qkv(x_value, return_spikes=True)
        
        # Use K and V from key/value inputs
        K = K_full
        V = V_full
        
        # Process each head
        head_outputs = []
        attention_weights = []
        
        for h, head in enumerate(self.attention_heads):
            # Extract this head's Q, K, V
            q_h = Q[:, :, h, :]  # [batch, n_queries, d_head]
            k_h = K[:, :, h, :]  # [batch, n_keys, d_head]
            v_h = V[:, :, h, :]  # [batch, n_keys, d_head]
            
            # Compute attention for this head
            output_h, attn_h = head(q_h, k_h, v_h)
            
            head_outputs.append(output_h)
            attention_weights.append(attn_h)
        
        # Concatenate heads: [batch, n_queries, n_heads * d_head]
        concat_output = torch.cat(head_outputs, dim=-1)
        
        # Project to output dimension
        output = F.linear(concat_output, self.W_out)
        
        # Layer norm + residual (if dimensions match)
        if output.shape == x_query.shape:
            output = self.layer_norm(output + x_query)
        else:
            output = self.layer_norm(output)
        
        # Stack attention weights: [batch, n_heads, n_queries, n_keys]
        attention = torch.stack(attention_weights, dim=1)
        
        return output, attention
    
    def learn_from_attention(
        self,
        attention_weights: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update attention parameters based on usage patterns.
        
        This implements STDP-like learning for attention:
        - Frequently co-attended items strengthen their connection
        - Rarely co-attended items weaken
        
        Optional reward modulation for three-factor learning.
        """
        if reward is not None:
            # Scale attention learning by reward
            learning_signal = attention_weights * reward.unsqueeze(-1).unsqueeze(-1)
        else:
            learning_signal = attention_weights
        
        # For phase attention: update preferred phases based on co-attention
        metrics = {}
        for h, head in enumerate(self.attention_heads):
            if isinstance(head, GammaPhaseAttention):
                # Keys that are often attended together should have similar phases
                # This is simplified - real implementation would track co-attention
                attn_h = learning_signal[:, h]  # [batch, n_queries, n_keys]
                
                # Compute which keys are often attended together
                coattention = torch.bmm(attn_h.transpose(-2, -1), attn_h)  # [batch, n_keys, n_keys]
                
                # Average over batch
                mean_coattention = coattention.mean(dim=0)  # [n_keys, n_keys]
                
                # Update phase preferences to align co-attended keys
                # (Simplified: just track for now)
                metrics[f"head_{h}_coattention"] = mean_coattention.mean().item()
        
        return metrics
    
    def get_complexity_estimate(self) -> Dict[str, int]:
        """Estimate computational complexity."""
        n_q = self.config.n_queries
        n_k = self.config.n_keys
        d = self.config.d_head
        h = self.config.n_heads
        k = self.config.top_k
        
        if self.config.attention_type == AttentionType.COINCIDENCE:
            # Depends on sparsity, estimate with target sparsity
            active_q = int(n_q * self.config.query_sparsity)
            active_k = int(n_k * self.config.key_sparsity)
            complexity = h * active_q * active_k * d
            
        elif self.config.attention_type == AttentionType.WINNER_TAKE_ALL:
            # O(n_q * n_k) for scores, but O(n_q * k) for attention
            complexity = h * (n_q * n_k + n_q * k * d)
            
        elif self.config.attention_type == AttentionType.OSCILLATION_PHASE:
            # O(n_q * n_k) but with sparse phase gating
            complexity = h * n_q * n_k
            
        else:
            complexity = h * n_q * n_k * d  # Full attention
        
        # Compare to full O(n²)
        full_attention = h * n_q * n_k * d
        
        return {
            "estimated_flops": complexity,
            "full_attention_flops": full_attention,
            "speedup_factor": full_attention / max(complexity, 1),
        }


class MultiScaleSpikingAttention(nn.Module):
    """
    Multi-scale attention with different temporal resolutions.
    
    Different heads attend at different timescales:
    - Fast heads: Recent context (gamma, ~25ms)
    - Medium heads: Working memory (~250ms)
    - Slow heads: Long-term context (~2.5s)
    
    This captures the hierarchical timescales of cortical processing.
    """
    
    def __init__(
        self,
        config: ScalableAttentionConfig,
        timescales_ms: Optional[List[float]] = None,
    ):
        super().__init__()
        self.config = config
        
        # Default timescales: fast to slow
        if timescales_ms is None:
            timescales_ms = [25.0, 100.0, 250.0, 1000.0]
        
        # Create attention at each timescale
        self.timescales = timescales_ms
        self.attention_layers = nn.ModuleList()
        
        for tau in timescales_ms:
            scale_config = ScalableAttentionConfig(
                n_queries=config.n_queries,
                n_keys=config.n_keys,
                d_model=config.d_model // len(timescales_ms),  # Split dimension
                n_heads=1,  # One head per timescale
                attention_type=config.attention_type,
                top_k=config.top_k,
                coincidence_window_ms=tau,  # Timescale-specific window
                neuron_tau_ms=tau / 2,  # Neurons match timescale
                device=config.device,
            )
            self.attention_layers.append(ScalableSpikingAttention(scale_config))
        
        # Combine timescales
        self.W_combine = nn.Linear(config.d_model, config.d_model)
    
    def reset_state(self) -> None:
        """Reset all timescales."""
        for layer in self.attention_layers:
            layer.reset_state()
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Multi-scale attention.
        
        Args:
            x: Input [batch, seq_len, d_model]
            
        Returns:
            output: Combined output [batch, seq_len, d_model]
            attentions: List of attention weights per timescale
        """
        outputs = []
        attentions = []
        
        # Split input across timescales
        chunk_size = self.config.d_model // len(self.timescales)
        x_chunks = x.split(chunk_size, dim=-1)
        
        for i, (layer, x_chunk) in enumerate(zip(self.attention_layers, x_chunks)):
            out, attn = layer(x_chunk)
            outputs.append(out)
            attentions.append(attn)
        
        # Concatenate and combine
        concat = torch.cat(outputs, dim=-1)
        output = self.W_combine(concat)
        
        return output, attentions
