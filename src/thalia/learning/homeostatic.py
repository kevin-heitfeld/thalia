"""
Homeostatic plasticity mechanisms.

These mechanisms maintain stable network activity by:
1. Intrinsic plasticity: Adjusting neuron thresholds based on firing rate
2. Synaptic scaling: Globally scaling synaptic weights to target activity

Homeostasis is crucial for:
- Preventing runaway excitation or silence
- Maintaining useful dynamic range
- Enabling stable learning over long timescales
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class IntrinsicPlasticityConfig:
    """Configuration for intrinsic plasticity.
    
    Attributes:
        target_rate: Target firing rate in Hz
        learning_rate: Learning rate for threshold adaptation
        tau_avg: Time constant for rate averaging in ms
        dt: Simulation timestep in ms
    """
    target_rate: float = 10.0  # Hz
    learning_rate: float = 0.001
    tau_avg: float = 1000.0  # Average over 1 second
    dt: float = 1.0


class IntrinsicPlasticity(nn.Module):
    """Intrinsic plasticity: adapt neuron thresholds to maintain target firing rate.
    
    When neurons fire too much, their threshold increases (harder to fire).
    When neurons fire too little, their threshold decreases (easier to fire).
    
    This implements a simplified version of intrinsic plasticity where:
        Δθ = η * (rate - target_rate)
    
    Args:
        n_neurons: Number of neurons to regulate
        config: Intrinsic plasticity configuration
        
    Example:
        >>> ip = IntrinsicPlasticity(n_neurons=100)
        >>> for t in range(1000):
        ...     spikes, voltage = neuron(input_current)
        ...     threshold_delta = ip(spikes)
        ...     neuron.v_threshold += threshold_delta
    """
    
    def __init__(
        self,
        n_neurons: int,
        config: Optional[IntrinsicPlasticityConfig] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or IntrinsicPlasticityConfig()
        
        # Compute decay factor for exponential moving average
        decay = torch.exp(torch.tensor(-self.config.dt / self.config.tau_avg))
        self.register_buffer("decay", decay)
        
        # Target rate per timestep
        target_per_step = self.config.target_rate * self.config.dt / 1000.0
        self.register_buffer("target", torch.tensor(target_per_step))
        
        # Running average of firing rate
        self.rate_avg: Optional[torch.Tensor] = None
        
    def reset(self, batch_size: int = 1) -> None:
        """Reset running rate average."""
        device = self.decay.device
        self.rate_avg = torch.zeros(
            batch_size, self.n_neurons, 
            device=device, dtype=torch.float32
        )
        
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Update rate estimate and compute threshold adjustment.
        
        Args:
            spikes: Binary spike tensor, shape (batch, n_neurons)
            
        Returns:
            Threshold adjustment (add to neuron thresholds), shape (batch, n_neurons)
        """
        if self.rate_avg is None:
            self.reset(batch_size=spikes.shape[0])
            
        # Update exponential moving average of rate
        self.rate_avg = self.decay * self.rate_avg + (1 - self.decay) * spikes
        
        # Compute threshold adjustment: increase if firing too much
        # Δθ = η * (rate_avg - target)
        threshold_delta = self.config.learning_rate * (self.rate_avg - self.target)
        
        return threshold_delta
    
    def get_rate_estimate(self) -> torch.Tensor:
        """Get current firing rate estimate in Hz."""
        if self.rate_avg is None:
            return torch.zeros(1, self.n_neurons)
        return self.rate_avg * 1000.0 / self.config.dt


@dataclass
class SynapticScalingConfig:
    """Configuration for synaptic scaling.
    
    Attributes:
        target_rate: Target firing rate in Hz
        learning_rate: Learning rate for weight scaling
        tau_avg: Time constant for rate averaging in ms
        dt: Simulation timestep in ms
        scale_min: Minimum scaling factor
        scale_max: Maximum scaling factor
    """
    target_rate: float = 10.0
    learning_rate: float = 0.0001
    tau_avg: float = 5000.0  # Average over 5 seconds
    dt: float = 1.0
    scale_min: float = 0.1
    scale_max: float = 10.0


class SynapticScaling(nn.Module):
    """Synaptic scaling: globally adjust synaptic weights to maintain activity.
    
    Unlike intrinsic plasticity (which adjusts thresholds), synaptic scaling
    multiplicatively adjusts all incoming weights to a neuron.
    
    When a neuron fires too little, all its incoming weights are scaled up.
    When a neuron fires too much, all its incoming weights are scaled down.
    
    Args:
        n_neurons: Number of post-synaptic neurons
        config: Synaptic scaling configuration
        
    Example:
        >>> ss = SynapticScaling(n_neurons=100)
        >>> for t in range(1000):
        ...     spikes, _ = layer(input_spikes)
        ...     scale = ss(spikes)
        ...     # Apply: synapse.weight.data *= scale.unsqueeze(0)
    """
    
    def __init__(
        self,
        n_neurons: int,
        config: Optional[SynapticScalingConfig] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or SynapticScalingConfig()
        
        decay = torch.exp(torch.tensor(-self.config.dt / self.config.tau_avg))
        self.register_buffer("decay", decay)
        
        target_per_step = self.config.target_rate * self.config.dt / 1000.0
        self.register_buffer("target", torch.tensor(target_per_step))
        
        # Running average and current scale factor
        self.rate_avg: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        
    def reset(self, batch_size: int = 1) -> None:
        """Reset state."""
        device = self.decay.device
        self.rate_avg = torch.zeros(
            batch_size, self.n_neurons,
            device=device, dtype=torch.float32
        )
        self.scale = torch.ones(
            batch_size, self.n_neurons,
            device=device, dtype=torch.float32
        )
        
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Update rate estimate and compute scaling factor.
        
        Args:
            spikes: Binary spike tensor, shape (batch, n_neurons)
            
        Returns:
            Scaling factor for incoming weights, shape (batch, n_neurons)
        """
        if self.rate_avg is None:
            self.reset(batch_size=spikes.shape[0])
            
        # Update exponential moving average
        self.rate_avg = self.decay * self.rate_avg + (1 - self.decay) * spikes
        
        # Compute scaling adjustment
        # If rate < target: scale up (multiply by >1)
        # If rate > target: scale down (multiply by <1)
        rate_ratio = self.target / (self.rate_avg + 1e-8)  # Avoid div by zero
        
        # Gradual adjustment toward target ratio
        self.scale = self.scale * (1 + self.config.learning_rate * (rate_ratio - 1))
        
        # Clamp to reasonable bounds
        self.scale = torch.clamp(
            self.scale, 
            self.config.scale_min, 
            self.config.scale_max
        )
        
        return self.scale
