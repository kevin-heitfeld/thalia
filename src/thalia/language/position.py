"""
Oscillatory Position Encoder - Represent sequence position through neural oscillations.

This module implements position encoding using oscillatory patterns, inspired by
the brain's use of neural oscillations (theta, gamma) for representing sequences.

Key Concepts:
=============

1. THETA PHASE PRECESSION
   In hippocampus, a place cell's firing phase advances through theta:
   - At start of field: fires at late theta phase
   - At end of field: fires at early theta phase
   - Position encoded in phase, not just rate

2. NESTED OSCILLATIONS (THETA-GAMMA)
   Gamma cycles nested within theta represent sequence positions:
   - Theta: ~8 Hz, represents context/chunk
   - Gamma: ~40 Hz, represents items within chunk
   - ~7 gamma cycles per theta = ~7 item capacity (working memory)

3. SINUSOIDAL POSITION ENCODING
   Similar to Transformer's position encoding:
   - Different frequencies for different dimensions
   - Enables generalization to unseen positions
   - But using actual oscillations, not static vectors

Biological Basis:
- O'Keefe & Recce (1993): Phase precession discovery
- Lisman & Jensen (2013): Theta-gamma neural code
- Buzsáki & Draguhn (2004): Neuronal oscillations hierarchy

References:
- Vaswani et al. (2017): Attention Is All You Need (sinusoidal encoding)
- Souza et al. (2020): Oscillatory networks for temporal coding

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn


class PositionEncodingType(Enum):
    """Types of position encoding."""
    SINUSOIDAL = "sinusoidal"       # Classic transformer-style
    OSCILLATORY = "oscillatory"     # Neural oscillation-based
    PHASE_PRECESSION = "phase_precession"  # Hippocampal-style
    NESTED_GAMMA = "nested_gamma"   # Theta-nested gamma


@dataclass
class PositionEncoderConfig:
    """Configuration for position encoder.
    
    Attributes:
        n_neurons: Number of output neurons
        max_positions: Maximum sequence length supported
        encoding_type: Type of position encoding
        n_timesteps: Number of timesteps for temporal encoding
        
        # Oscillation parameters
        theta_frequency_hz: Theta oscillation frequency
        gamma_frequency_hz: Gamma oscillation frequency
        dt_ms: Simulation timestep
        
        # Sinusoidal parameters
        base_frequency: Base frequency for sinusoidal encoding
        
        # Phase precession parameters
        precession_rate: How fast phase precesses with position
        
        # Learning
        learnable: Whether position encoding is learnable
        
        device: Computation device
    """
    n_neurons: int = 256
    max_positions: int = 2048
    encoding_type: PositionEncodingType = PositionEncodingType.NESTED_GAMMA
    n_timesteps: int = 20
    
    # Oscillations
    theta_frequency_hz: float = 8.0
    gamma_frequency_hz: float = 40.0
    dt_ms: float = 1.0
    
    # Sinusoidal
    base_frequency: float = 10000.0
    
    # Phase precession
    precession_rate: float = 0.5  # Radians per position
    
    # Learning
    learnable: bool = False
    
    device: str = "cpu"
    
    @property
    def theta_period_ms(self) -> float:
        return 1000.0 / self.theta_frequency_hz
    
    @property
    def gamma_period_ms(self) -> float:
        return 1000.0 / self.gamma_frequency_hz
    
    @property
    def gammas_per_theta(self) -> float:
        return self.gamma_frequency_hz / self.theta_frequency_hz


class OscillatoryPositionEncoder(nn.Module):
    """
    Encode sequence positions using neural oscillations.
    
    This creates position-dependent spike patterns that can be added
    to or combined with content representations.
    
    The oscillatory encoding has several advantages:
    - Temporal dynamics: positions unfold over time
    - Biological plausibility: matches brain mechanisms
    - Relative position: phase differences encode distance
    - Extrapolation: generalizes to longer sequences
    
    Usage:
        encoder = OscillatoryPositionEncoder(PositionEncoderConfig(
            n_neurons=256,
            max_positions=1024,
        ))
        
        # Get position encoding
        positions = torch.arange(10).unsqueeze(0)  # [1, 10]
        encoding = encoder(positions)  # [1, 10, n_timesteps, n_neurons]
    """
    
    def __init__(self, config: PositionEncoderConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize based on encoding type
        if config.encoding_type == PositionEncodingType.SINUSOIDAL:
            self._init_sinusoidal()
        elif config.encoding_type == PositionEncodingType.OSCILLATORY:
            self._init_oscillatory()
        elif config.encoding_type == PositionEncodingType.PHASE_PRECESSION:
            self._init_phase_precession()
        elif config.encoding_type == PositionEncodingType.NESTED_GAMMA:
            self._init_nested_gamma()
        
        # Optional learnable component
        if config.learnable:
            self.learnable_pos = nn.Embedding(config.max_positions, config.n_neurons)
        else:
            self.learnable_pos = None
    
    def _init_sinusoidal(self) -> None:
        """Initialize sinusoidal position encoding (Transformer-style)."""
        config = self.config
        
        # Create position encoding matrix
        pe = torch.zeros(config.max_positions, config.n_neurons)
        position = torch.arange(config.max_positions).unsqueeze(1).float()
        
        # Different frequencies for each dimension
        div_term = torch.exp(
            torch.arange(0, config.n_neurons, 2).float() *
            (-math.log(config.base_frequency) / config.n_neurons)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:config.n_neurons // 2])
        
        self.register_buffer("sinusoidal_pe", pe)
    
    def _init_oscillatory(self) -> None:
        """Initialize oscillatory position encoding."""
        config = self.config
        
        # Each neuron has a preferred phase
        # Phases are distributed to cover full range
        phases = torch.linspace(0, 2 * math.pi, config.n_neurons)
        self.register_buffer("preferred_phases", phases)
        
        # Frequency gradient across neurons (for multi-scale encoding)
        # Low-numbered neurons = slow oscillation (long-range)
        # High-numbered neurons = fast oscillation (fine-grained)
        freq_range = torch.linspace(0.5, 2.0, config.n_neurons)  # Relative to theta
        self.register_buffer("frequency_scale", freq_range)
    
    def _init_phase_precession(self) -> None:
        """Initialize phase precession encoding."""
        config = self.config
        
        # Each neuron has a "place field" center position
        field_centers = torch.linspace(0, config.max_positions, config.n_neurons)
        self.register_buffer("field_centers", field_centers)
        
        # Field width (tuning curve width)
        self.field_width = config.max_positions / config.n_neurons * 3  # Overlapping
    
    def _init_nested_gamma(self) -> None:
        """Initialize theta-nested gamma encoding."""
        config = self.config
        
        # Theta-phase neurons
        n_theta = config.n_neurons // 2
        theta_phases = torch.linspace(0, 2 * math.pi, n_theta)
        self.register_buffer("theta_phases", theta_phases)
        
        # Gamma-phase neurons (remaining neurons)
        n_gamma = config.n_neurons - n_theta
        gamma_phases = torch.linspace(0, 2 * math.pi, n_gamma)
        self.register_buffer("gamma_phases", gamma_phases)
        
        # Number of gamma cycles per theta
        self.gammas_per_theta = int(config.gammas_per_theta)
    
    def forward(
        self,
        position_ids: torch.Tensor,
        as_spikes: bool = True,
    ) -> torch.Tensor:
        """
        Generate position encoding.
        
        Args:
            position_ids: Position indices [batch, seq_len]
            as_spikes: Whether to return spike patterns (vs continuous)
            
        Returns:
            encoding: Position encoding
                If as_spikes: [batch, seq_len, n_timesteps, n_neurons]
                Else: [batch, seq_len, n_neurons]
        """
        config = self.config
        
        if config.encoding_type == PositionEncodingType.SINUSOIDAL:
            encoding = self._encode_sinusoidal(position_ids, as_spikes)
        elif config.encoding_type == PositionEncodingType.OSCILLATORY:
            encoding = self._encode_oscillatory(position_ids, as_spikes)
        elif config.encoding_type == PositionEncodingType.PHASE_PRECESSION:
            encoding = self._encode_phase_precession(position_ids, as_spikes)
        elif config.encoding_type == PositionEncodingType.NESTED_GAMMA:
            encoding = self._encode_nested_gamma(position_ids, as_spikes)
        else:
            raise ValueError(f"Unknown encoding type: {config.encoding_type}")
        
        # Add learnable component if available
        if self.learnable_pos is not None:
            learned = self.learnable_pos(position_ids)  # [batch, seq, n_neurons]
            if as_spikes:
                # Broadcast learned to all timesteps and add
                learned = learned.unsqueeze(2)  # [batch, seq, 1, n_neurons]
                encoding = encoding + learned * 0.1  # Subtle modulation
            else:
                encoding = encoding + learned * 0.1
        
        return encoding
    
    def _encode_sinusoidal(
        self,
        position_ids: torch.Tensor,
        as_spikes: bool,
    ) -> torch.Tensor:
        """Sinusoidal encoding with optional spike conversion."""
        # Look up precomputed encoding
        encoding = self.sinusoidal_pe[position_ids]  # [batch, seq, n_neurons]
        
        if as_spikes:
            # Convert to spike patterns over time
            encoding = self._continuous_to_spikes(encoding)
        
        return encoding
    
    def _encode_oscillatory(
        self,
        position_ids: torch.Tensor,
        as_spikes: bool,
    ) -> torch.Tensor:
        """Generate oscillatory position encoding."""
        batch, seq_len = position_ids.shape
        config = self.config
        
        # Position determines the starting phase
        position_phase = position_ids.float().unsqueeze(-1) * config.precession_rate
        # [batch, seq, 1]
        
        if as_spikes:
            # Generate spike patterns over time
            spikes = torch.zeros(
                batch, seq_len, config.n_timesteps, config.n_neurons,
                device=self.device,
            )
            
            for t in range(config.n_timesteps):
                # Current oscillation phase
                time_phase = 2 * math.pi * t * config.dt_ms / config.theta_period_ms
                
                # Each neuron's phase
                neuron_phase = (
                    position_phase +  # Position-dependent
                    self.preferred_phases +  # Neuron's preferred phase
                    time_phase * self.frequency_scale  # Time evolution
                )
                
                # Spike probability: maximum at phase = 0
                phase_mod = neuron_phase % (2 * math.pi)
                spike_prob = torch.exp(-4 * (phase_mod - math.pi) ** 2)
                
                spikes[:, :, t, :] = (torch.rand_like(spike_prob) < spike_prob * 0.3).float()
            
            return spikes
        else:
            # Return phase encoding
            encoding = torch.cos(position_phase + self.preferred_phases)
            return encoding
    
    def _encode_phase_precession(
        self,
        position_ids: torch.Tensor,
        as_spikes: bool,
    ) -> torch.Tensor:
        """Generate phase precession encoding."""
        batch, seq_len = position_ids.shape
        config = self.config
        
        # Distance from each neuron's field center
        positions = position_ids.float().unsqueeze(-1)  # [batch, seq, 1]
        distance = positions - self.field_centers  # [batch, seq, n_neurons]
        
        # Normalized position within field (-1 to 1)
        normalized_pos = distance / self.field_width
        
        # Neurons active when position is within their field
        in_field = torch.abs(normalized_pos) < 1.0
        
        # Phase precesses from late to early as position advances through field
        # Late phase = positive normalized_pos, early = negative
        precession_phase = -normalized_pos * math.pi  # Full pi precession
        
        if as_spikes:
            spikes = torch.zeros(
                batch, seq_len, config.n_timesteps, config.n_neurons,
                device=self.device,
            )
            
            for t in range(config.n_timesteps):
                # Current theta phase
                theta_phase = 2 * math.pi * t * config.dt_ms / config.theta_period_ms
                
                # Spike when theta matches precessed phase
                phase_diff = (theta_phase - precession_phase) % (2 * math.pi)
                phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)
                
                # Spike probability: high when phase matches, neuron in field
                spike_prob = torch.exp(-4 * phase_diff ** 2) * in_field.float()
                
                spikes[:, :, t, :] = (torch.rand_like(spike_prob) < spike_prob * 0.5).float()
            
            return spikes
        else:
            # Return place field activation with phase
            activation = torch.exp(-2 * normalized_pos ** 2) * in_field.float()
            return activation
    
    def _encode_nested_gamma(
        self,
        position_ids: torch.Tensor,
        as_spikes: bool,
    ) -> torch.Tensor:
        """
        Generate theta-nested gamma encoding.
        
        Items are represented within theta cycles via gamma:
        - Each theta cycle = one "chunk" of ~7 items
        - Position within chunk encoded by gamma phase
        - Chunk number encoded by theta phase
        """
        batch, seq_len = position_ids.shape
        config = self.config
        
        positions = position_ids.float()  # [batch, seq]
        
        # Chunk index (which theta cycle)
        chunk_idx = (positions / self.gammas_per_theta).floor()
        
        # Position within chunk (gamma phase index)
        within_chunk_pos = positions % self.gammas_per_theta
        
        n_theta = config.n_neurons // 2
        n_gamma = config.n_neurons - n_theta
        
        if as_spikes:
            spikes = torch.zeros(
                batch, seq_len, config.n_timesteps, config.n_neurons,
                device=self.device,
            )
            
            for t in range(config.n_timesteps):
                # Current theta and gamma phases
                time_ms = t * config.dt_ms
                theta_phase = 2 * math.pi * time_ms / config.theta_period_ms
                gamma_phase = 2 * math.pi * time_ms / config.gamma_period_ms
                
                # Theta neurons: encode chunk
                theta_target = chunk_idx.unsqueeze(-1) * 0.5  # Chunk-dependent phase
                theta_diff = (theta_phase + theta_target - self.theta_phases) % (2 * math.pi)
                theta_diff = torch.min(theta_diff, 2 * math.pi - theta_diff)
                theta_prob = torch.exp(-4 * theta_diff ** 2)
                
                # Gamma neurons: encode within-chunk position
                gamma_target = within_chunk_pos.unsqueeze(-1) * (2 * math.pi / self.gammas_per_theta)
                gamma_diff = (gamma_phase + gamma_target - self.gamma_phases) % (2 * math.pi)
                gamma_diff = torch.min(gamma_diff, 2 * math.pi - gamma_diff)
                gamma_prob = torch.exp(-4 * gamma_diff ** 2)
                
                # Generate spikes
                theta_spikes = (torch.rand_like(theta_prob) < theta_prob * 0.3).float()
                gamma_spikes = (torch.rand_like(gamma_prob) < gamma_prob * 0.3).float()
                
                spikes[:, :, t, :n_theta] = theta_spikes
                spikes[:, :, t, n_theta:] = gamma_spikes
            
            return spikes
        else:
            # Return phase encoding
            encoding = torch.zeros(batch, seq_len, config.n_neurons, device=self.device)
            
            # Theta encoding
            theta_target = chunk_idx.unsqueeze(-1) * 0.5
            encoding[:, :, :n_theta] = torch.cos(theta_target - self.theta_phases)
            
            # Gamma encoding  
            gamma_target = within_chunk_pos.unsqueeze(-1) * (2 * math.pi / self.gammas_per_theta)
            encoding[:, :, n_theta:] = torch.cos(gamma_target - self.gamma_phases)
            
            return encoding
    
    def _continuous_to_spikes(
        self,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert continuous encoding to spike patterns.
        
        Args:
            encoding: Continuous encoding [batch, seq_len, n_neurons]
            
        Returns:
            spikes: Spike patterns [batch, seq_len, n_timesteps, n_neurons]
        """
        batch, seq_len, n_neurons = encoding.shape
        n_timesteps = self.config.n_timesteps
        
        # Normalize encoding to [0, 1]
        encoding_norm = (encoding + 1) / 2  # Assuming sinusoidal in [-1, 1]
        
        # Generate spikes probabilistically
        spikes = torch.zeros(
            batch, seq_len, n_timesteps, n_neurons,
            device=self.device,
        )
        
        for t in range(n_timesteps):
            # Time-varying modulation
            phase = 2 * math.pi * t / n_timesteps
            modulation = (1 + torch.sin(
                torch.arange(n_neurons, device=self.device).float() * 0.1 + phase
            )) / 2
            
            spike_prob = encoding_norm * modulation * 0.2
            spikes[:, :, t, :] = (torch.rand_like(spike_prob) < spike_prob).float()
        
        return spikes
    
    def get_relative_encoding(
        self,
        position_ids: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get relative position encoding between positions.
        
        This is useful for attention mechanisms that care about
        relative rather than absolute positions.
        
        Args:
            position_ids: Key positions [batch, seq_len_k]
            query_positions: Query positions [batch, seq_len_q]
            
        Returns:
            relative: Relative encoding [batch, seq_len_q, seq_len_k]
        """
        # Compute relative distances
        distances = query_positions.unsqueeze(-1) - position_ids.unsqueeze(-2)
        # [batch, seq_q, seq_k]
        
        # Create relative position encoding using oscillatory patterns
        relative_encoding = torch.zeros(
            distances.shape + (self.config.n_neurons,),
            device=self.device,
        )
        
        # Different frequencies encode different scales
        for i in range(self.config.n_neurons):
            freq = 1.0 / (10000 ** (2 * i / self.config.n_neurons))
            relative_encoding[..., i] = torch.cos(distances.float() * freq)
        
        # Return mean over neurons (or could return full)
        return relative_encoding.mean(dim=-1)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get encoder diagnostics."""
        return {
            "encoding_type": self.config.encoding_type.value,
            "n_neurons": self.config.n_neurons,
            "max_positions": self.config.max_positions,
            "theta_frequency_hz": self.config.theta_frequency_hz,
            "gamma_frequency_hz": self.config.gamma_frequency_hz,
            "gammas_per_theta": self.config.gammas_per_theta,
        }


class SequenceTimer(nn.Module):
    """
    Track temporal structure of sequences using oscillations.
    
    Maintains a running oscillatory state that tracks:
    - Current position in sequence
    - Elapsed time
    - Rhythm phase
    
    Useful for generating predictions about timing.
    """
    
    def __init__(
        self,
        n_neurons: int = 128,
        theta_freq_hz: float = 8.0,
        gamma_freq_hz: float = 40.0,
        dt_ms: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms
        self.device_str = device
        
        # Oscillation periods
        self.theta_period = 1000.0 / theta_freq_hz
        self.gamma_period = 1000.0 / gamma_freq_hz
        
        # Phase state
        self.register_buffer("theta_phase", torch.tensor(0.0))
        self.register_buffer("gamma_phase", torch.tensor(0.0))
        self.register_buffer("position", torch.tensor(0))
        
        # Phase to neuron mapping
        phases = torch.linspace(0, 2 * math.pi, n_neurons)
        self.register_buffer("neuron_phases", phases)
    
    def reset(self) -> None:
        """Reset timer state."""
        self.theta_phase.zero_()
        self.gamma_phase.zero_()
        self.position.zero_()
    
    def step(self, n_steps: int = 1) -> torch.Tensor:
        """
        Advance timer by n_steps.
        
        Args:
            n_steps: Number of timesteps to advance
            
        Returns:
            spikes: Current oscillatory state as spikes [n_neurons]
        """
        # Update phases
        delta_t = n_steps * self.dt_ms
        self.theta_phase.add_(2 * math.pi * delta_t / self.theta_period)
        self.gamma_phase.add_(2 * math.pi * delta_t / self.gamma_period)
        
        # Wrap phases
        self.theta_phase.remainder_(2 * math.pi)
        self.gamma_phase.remainder_(2 * math.pi)
        
        # Generate spikes based on phase proximity
        n_theta = self.n_neurons // 2
        
        # Theta neurons
        theta_diff = (self.theta_phase - self.neuron_phases[:n_theta]) % (2 * math.pi)
        theta_diff = torch.min(theta_diff, 2 * math.pi - theta_diff)
        theta_prob = torch.exp(-4 * theta_diff ** 2) * 0.3
        
        # Gamma neurons
        gamma_diff = (self.gamma_phase - self.neuron_phases[:self.n_neurons - n_theta]) % (2 * math.pi)
        gamma_diff = torch.min(gamma_diff, 2 * math.pi - gamma_diff)
        gamma_prob = torch.exp(-4 * gamma_diff ** 2) * 0.3
        
        # Generate spikes
        spikes = torch.zeros(self.n_neurons, device=self.theta_phase.device)
        spikes[:n_theta] = (torch.rand(n_theta, device=spikes.device) < theta_prob).float()
        spikes[n_theta:] = (torch.rand(self.n_neurons - n_theta, device=spikes.device) < gamma_prob).float()
        
        return spikes
    
    def advance_position(self) -> None:
        """Advance sequence position by one."""
        self.position.add_(1)
    
    def get_state(self) -> Dict[str, float]:
        """Get current timer state."""
        return {
            "theta_phase": self.theta_phase.item(),
            "gamma_phase": self.gamma_phase.item(),
            "position": self.position.item(),
        }
