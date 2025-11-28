"""
Leaky Integrate-and-Fire (LIF) neuron model.

The LIF neuron is the fundamental building block of THALIA networks.
It accumulates input over time, leaks toward a resting potential,
and fires a spike when threshold is reached.

Membrane dynamics:
    τ_m * dV/dt = -(V - V_rest) + R * I

When V >= V_threshold:
    - Emit spike
    - V = V_reset
    - Enter refractory period
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LIFConfig:
    """Configuration for LIF neuron parameters.
    
    Attributes:
        tau_mem: Membrane time constant in ms (default: 20.0)
        v_rest: Resting membrane potential (default: 0.0)
        v_reset: Reset potential after spike (default: 0.0)
        v_threshold: Spike threshold (default: 1.0)
        tau_ref: Refractory period in ms (default: 2.0)
        dt: Simulation timestep in ms (default: 1.0)
    """
    tau_mem: float = 20.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau_ref: float = 2.0
    dt: float = 1.0
    
    @property
    def decay(self) -> float:
        """Membrane decay factor per timestep."""
        return torch.exp(torch.tensor(-self.dt / self.tau_mem)).item()
    
    @property
    def ref_steps(self) -> int:
        """Refractory period in timesteps."""
        return int(self.tau_ref / self.dt)


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron layer.
    
    Implements a population of LIF neurons with shared parameters.
    Supports batch processing and GPU acceleration.
    
    Args:
        n_neurons: Number of neurons in the layer
        config: LIF configuration parameters
        
    Example:
        >>> config = LIFConfig(tau_mem=20.0, v_threshold=1.0)
        >>> lif = LIFNeuron(n_neurons=100, config=config)
        >>> 
        >>> # Simulate with input current
        >>> spikes = []
        >>> lif.reset_state(batch_size=1)
        >>> for t in range(1000):
        ...     input_current = torch.randn(1, 100) * 0.5
        ...     spike, voltage = lif(input_current)
        ...     spikes.append(spike)
    """
    
    def __init__(
        self, 
        n_neurons: int, 
        config: Optional[LIFConfig] = None
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config or LIFConfig()
        
        # Register decay as buffer (moves with device)
        self.register_buffer(
            "decay", 
            torch.tensor(self.config.decay, dtype=torch.float32)
        )
        self.register_buffer(
            "v_threshold",
            torch.tensor(self.config.v_threshold, dtype=torch.float32)
        )
        self.register_buffer(
            "v_rest",
            torch.tensor(self.config.v_rest, dtype=torch.float32)
        )
        self.register_buffer(
            "v_reset",
            torch.tensor(self.config.v_reset, dtype=torch.float32)
        )
        
        # State variables (initialized on first forward or reset)
        self.membrane: Optional[torch.Tensor] = None
        self.refractory: Optional[torch.Tensor] = None
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset neuron state to resting potential.
        
        Args:
            batch_size: Batch dimension for parallel processing
        """
        device = self.decay.device
        self.membrane = torch.full(
            (batch_size, self.n_neurons), 
            self.config.v_rest,
            device=device,
            dtype=torch.float32
        )
        self.refractory = torch.zeros(
            (batch_size, self.n_neurons),
            device=device,
            dtype=torch.int32
        )
        
    def forward(
        self, 
        input_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep of input.
        
        Args:
            input_current: Input current to neurons, shape (batch, n_neurons)
            
        Returns:
            spikes: Binary spike tensor, shape (batch, n_neurons)
            membrane: Membrane potentials after update, shape (batch, n_neurons)
        """
        # Initialize state if needed
        if self.membrane is None:
            self.reset_state(batch_size=input_current.shape[0])
            
        # Ensure state matches batch size
        if self.membrane.shape[0] != input_current.shape[0]:
            self.reset_state(batch_size=input_current.shape[0])
        
        # Decrement refractory counter
        self.refractory = torch.clamp(self.refractory - 1, min=0)
        
        # Check which neurons are not in refractory period
        not_refractory = (self.refractory == 0).float()
        
        # Leaky integration (only for non-refractory neurons)
        # V(t+1) = decay * (V(t) - V_rest) + V_rest + I(t) * not_refractory
        self.membrane = (
            self.decay * (self.membrane - self.v_rest) 
            + self.v_rest 
            + input_current * not_refractory
        )
        
        # Spike generation
        spikes = (self.membrane >= self.v_threshold).float()
        
        # Reset spiking neurons
        self.membrane = torch.where(
            spikes.bool(),
            self.v_reset.expand_as(self.membrane),
            self.membrane
        )
        
        # Set refractory period for spiking neurons
        self.refractory = torch.where(
            spikes.bool(),
            torch.full_like(self.refractory, self.config.ref_steps),
            self.refractory
        )
        
        return spikes, self.membrane.clone()
    
    def __repr__(self) -> str:
        return (
            f"LIFNeuron(n_neurons={self.n_neurons}, "
            f"tau_mem={self.config.tau_mem}, "
            f"v_threshold={self.config.v_threshold})"
        )
