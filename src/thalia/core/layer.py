"""
SNN layer: a collection of neurons with optional recurrent connections.
"""

from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.synapse import Synapse, SynapseConfig


class SNNLayer(nn.Module):
    """A layer of spiking neurons with optional input and recurrent connections.
    
    Args:
        n_neurons: Number of neurons in the layer
        neuron_config: Configuration for the neurons
        input_size: Size of external input (None for no external input)
        input_connectivity: Connection probability for input synapses
        recurrent: Whether to add recurrent connections
        recurrent_connectivity: Connection probability for recurrent synapses
        
    Example:
        >>> layer = SNNLayer(
        ...     n_neurons=100,
        ...     input_size=784,  # e.g., MNIST pixels
        ...     recurrent=True,
        ...     recurrent_connectivity=0.1
        ... )
        >>> 
        >>> layer.reset_state(batch_size=32)
        >>> for t in range(100):
        ...     input_spikes = encode_input(images)  # Your encoding
        ...     spikes, voltages = layer(input_spikes)
    """
    
    def __init__(
        self,
        n_neurons: int,
        neuron_config: Optional[LIFConfig] = None,
        input_size: Optional[int] = None,
        input_connectivity: float = 1.0,
        recurrent: bool = False,
        recurrent_connectivity: float = 0.1,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.recurrent = recurrent
        
        # Create neurons
        self.neurons = LIFNeuron(n_neurons, config=neuron_config)
        
        # Input synapses (if external input provided)
        if input_size is not None:
            self.input_synapses = Synapse(
                n_pre=input_size,
                n_post=n_neurons,
                connectivity=input_connectivity,
            )
        else:
            self.input_synapses = None
            
        # Recurrent synapses
        if recurrent:
            self.recurrent_synapses = Synapse(
                n_pre=n_neurons,
                n_post=n_neurons,
                connectivity=recurrent_connectivity,
            )
        else:
            self.recurrent_synapses = None
            
        # Store last spikes for recurrence
        self.last_spikes: Optional[torch.Tensor] = None
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all state (neurons and spike history).
        
        Args:
            batch_size: Batch dimension for parallel processing
        """
        self.neurons.reset_state(batch_size)
        self.last_spikes = torch.zeros(
            batch_size, 
            self.n_neurons,
            device=next(self.parameters()).device
        )
        
    def forward(
        self,
        input_spikes: Optional[torch.Tensor] = None,
        external_current: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.
        
        Args:
            input_spikes: External input spikes, shape (batch, input_size)
            external_current: Direct current injection, shape (batch, n_neurons)
            
        Returns:
            spikes: Output spikes, shape (batch, n_neurons)
            voltages: Membrane potentials, shape (batch, n_neurons)
        """
        # Initialize state if needed
        if self.last_spikes is None:
            batch_size = 1
            if input_spikes is not None:
                batch_size = input_spikes.shape[0]
            elif external_current is not None:
                batch_size = external_current.shape[0]
            self.reset_state(batch_size)
        
        # Accumulate input current
        total_current = torch.zeros_like(self.last_spikes)
        
        # Input from external sources
        if input_spikes is not None and self.input_synapses is not None:
            total_current = total_current + self.input_synapses(input_spikes)
            
        # Input from recurrent connections
        if self.recurrent_synapses is not None:
            total_current = total_current + self.recurrent_synapses(self.last_spikes)
            
        # Direct current injection
        if external_current is not None:
            total_current = total_current + external_current
            
        # Process through neurons
        spikes, voltages = self.neurons(total_current)
        
        # Store spikes for next recurrent step
        self.last_spikes = spikes
        
        return spikes, voltages
    
    def simulate(
        self,
        duration: int,
        input_spikes: Optional[torch.Tensor] = None,
        external_current: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Simulate the layer for a given duration.
        
        Args:
            duration: Number of timesteps to simulate
            input_spikes: Time series of input spikes, shape (time, batch, input_size)
            external_current: Time series of currents, shape (time, batch, n_neurons)
            batch_size: Batch size (used if no input provided)
            
        Returns:
            Dictionary with 'spikes' and 'voltages' tensors of shape (time, batch, n_neurons)
        """
        self.reset_state(batch_size)
        
        spike_history = []
        voltage_history = []
        
        for t in range(duration):
            # Get input for this timestep
            inp_t = None if input_spikes is None else input_spikes[t]
            cur_t = None if external_current is None else external_current[t]
            
            spikes, voltages = self(input_spikes=inp_t, external_current=cur_t)
            
            spike_history.append(spikes)
            voltage_history.append(voltages)
            
        return {
            "spikes": torch.stack(spike_history),
            "voltages": torch.stack(voltage_history),
        }
        
    def __repr__(self) -> str:
        parts = [f"SNNLayer(n_neurons={self.n_neurons}"]
        if self.input_size:
            parts.append(f"input_size={self.input_size}")
        if self.recurrent:
            parts.append("recurrent=True")
        return ", ".join(parts) + ")"
