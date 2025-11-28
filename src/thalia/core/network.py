"""
Network containers for multi-layer SNN architectures.
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn

from thalia.core.layer import SNNLayer
from thalia.core.neuron import LIFConfig


class SNNNetwork(nn.Module):
    """A feedforward network of SNN layers.
    
    Args:
        layer_sizes: List of layer sizes (including input)
        neuron_config: Configuration for all neurons
        connectivity: Connection probability between layers
        recurrent: Whether layers have recurrent connections
        
    Example:
        >>> network = SNNNetwork(
        ...     layer_sizes=[784, 256, 128, 10],
        ...     recurrent=True
        ... )
        >>> network.reset_state(batch_size=32)
        >>> for t in range(100):
        ...     output_spikes = network(input_spikes[t])
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        neuron_config: Optional[LIFConfig] = None,
        connectivity: float = 1.0,
        recurrent: bool = False,
        recurrent_connectivity: float = 0.1,
    ):
        super().__init__()
        
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output sizes")
            
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # Create layers
        layers = []
        for i in range(self.n_layers):
            layer = SNNLayer(
                n_neurons=layer_sizes[i + 1],
                neuron_config=neuron_config,
                input_size=layer_sizes[i],
                input_connectivity=connectivity,
                recurrent=recurrent,
                recurrent_connectivity=recurrent_connectivity,
            )
            layers.append(layer)
            
        self.layers = nn.ModuleList(layers)
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset_state(batch_size)
            
    def forward(
        self, 
        input_spikes: torch.Tensor
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Process one timestep through all layers.
        
        Args:
            input_spikes: Input spikes, shape (batch, input_size)
            
        Returns:
            output_spikes: Output layer spikes
            all_spikes: List of spike tensors for each layer
        """
        current_input = input_spikes
        all_spikes = []
        
        for layer in self.layers:
            spikes, _ = layer(input_spikes=current_input)
            all_spikes.append(spikes)
            current_input = spikes
            
        return all_spikes[-1], all_spikes
    
    def simulate(
        self,
        duration: int,
        input_spikes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Simulate network for given duration.
        
        Args:
            duration: Number of timesteps
            input_spikes: Input spike train, shape (time, batch, input_size)
            
        Returns:
            Dictionary with output spikes and per-layer spike histories
        """
        batch_size = input_spikes.shape[1]
        self.reset_state(batch_size)
        
        output_history = []
        layer_histories = [[] for _ in range(self.n_layers)]
        
        for t in range(duration):
            output, all_spikes = self(input_spikes[t])
            output_history.append(output)
            for i, spikes in enumerate(all_spikes):
                layer_histories[i].append(spikes)
                
        return {
            "output": torch.stack(output_history),
            "layers": [torch.stack(h) for h in layer_histories],
        }
        
    def __repr__(self) -> str:
        sizes = " -> ".join(map(str, self.layer_sizes))
        return f"SNNNetwork({sizes})"
