"""
Working memory with reverberating circuits.

Implements sustained neural activity for holding information
temporarily in mind. Key features:
- Reverberating loops that maintain activity without input
- Capacity limits (7±2 items, like human WM)
- Decay over time if not refreshed
- Gating mechanisms for loading/clearing

This is biologically inspired by prefrontal cortex circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.synapse import Synapse, SynapseConfig, SynapseType


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory module.
    
    Attributes:
        n_slots: Number of memory slots (capacity)
        slot_size: Number of neurons per slot
        tau_mem: Membrane time constant
        reverb_strength: Strength of recurrent connections for maintenance
        decay_rate: Rate of activity decay when not refreshed
        gate_threshold: Threshold for loading/clearing gates
        noise_std: Noise for spontaneous variability
        dt: Simulation timestep
    """
    n_slots: int = 7  # 7±2 slots like human WM
    slot_size: int = 50
    tau_mem: float = 30.0  # Longer time constant for persistence
    reverb_strength: float = 0.8
    decay_rate: float = 0.01
    gate_threshold: float = 0.5
    noise_std: float = 0.02
    dt: float = 1.0
    
    @property
    def total_neurons(self) -> int:
        """Total number of neurons."""
        return self.n_slots * self.slot_size


class MemorySlot(nn.Module):
    """A single working memory slot.
    
    Each slot can hold one item through reverberating activity.
    Has gating for controlled loading and clearing.
    """
    
    def __init__(
        self,
        size: int,
        reverb_strength: float = 0.8,
        decay_rate: float = 0.01,
        tau_mem: float = 30.0,
        noise_std: float = 0.02,
        dt: float = 1.0,
    ):
        super().__init__()
        self.size = size
        self.reverb_strength = reverb_strength
        self.decay_rate = decay_rate
        
        # Neurons for this slot
        neuron_config = LIFConfig(
            tau_mem=tau_mem,
            noise_std=noise_std,
            dt=dt,
        )
        self.neurons = LIFNeuron(n_neurons=size, config=neuron_config)
        
        # Recurrent weights for reverberating activity
        self.recurrent = nn.Parameter(
            torch.randn(size, size) * 0.1 * reverb_strength
        )
        # Mask out self-connections
        self.register_buffer("self_mask", 1 - torch.eye(size))
        
        # Gating signals
        self._load_gate: float = 0.0
        self._clear_gate: float = 0.0
        
        # Activity tracking
        self._last_spikes: Optional[torch.Tensor] = None
        self._activity_level: float = 0.0
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset neuron states."""
        self.neurons.reset_state(batch_size)
        self._last_spikes = None
        self._activity_level = 0.0
        
    def set_gates(self, load: float = 0.0, clear: float = 0.0) -> None:
        """Set gating signals.
        
        Args:
            load: Load gate (0-1), controls writing new content
            clear: Clear gate (0-1), controls erasing content
        """
        self._load_gate = max(0.0, min(1.0, load))
        self._clear_gate = max(0.0, min(1.0, clear))
        
    def forward(
        self,
        input_pattern: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Process one timestep.
        
        Args:
            input_pattern: External input to load, shape (batch, size)
            
        Returns:
            spikes: Output spikes, shape (batch, size)
            activity_level: Mean activity level (0-1)
        """
        # Get batch size
        if input_pattern is not None:
            batch_size = input_pattern.shape[0]
        elif self.neurons.membrane is not None:
            batch_size = self.neurons.membrane.shape[0]
        else:
            batch_size = 1
            
        # Initialize if needed
        if self.neurons.membrane is None:
            self.reset_state(batch_size)
            
        # Compute recurrent input from last spikes
        if self._last_spikes is not None:
            weights = self.recurrent * self.self_mask
            reverb_input = torch.matmul(self._last_spikes, weights)
            
            # Apply decay
            reverb_input = reverb_input * (1 - self.decay_rate)
        else:
            reverb_input = torch.zeros(batch_size, self.size, 
                                       device=self.neurons.membrane.device)
        
        # Clear gate reduces recurrent input
        reverb_input = reverb_input * (1 - self._clear_gate)
        
        # Load gate modulates external input
        if input_pattern is not None:
            external_input = input_pattern * self._load_gate
        else:
            external_input = torch.zeros_like(reverb_input)
            
        # Total input
        total_input = reverb_input + external_input
        
        # Step neurons
        spikes, _ = self.neurons(total_input)
        
        # Update state
        self._last_spikes = spikes
        self._activity_level = spikes.mean().item()
        
        return spikes, self._activity_level
    
    @property
    def is_active(self) -> bool:
        """Check if slot has sustained activity."""
        return self._activity_level > 0.05
    
    def get_content(self) -> Optional[torch.Tensor]:
        """Get current content pattern (if active)."""
        if not self.is_active or self._last_spikes is None:
            return None
        return self._last_spikes.clone()


class WorkingMemory(nn.Module):
    """Working memory system with multiple slots.
    
    Maintains information through reverberating neural activity.
    Has limited capacity and gated access.
    
    Example:
        >>> wm = WorkingMemory(WorkingMemoryConfig(n_slots=5))
        >>> wm.reset_state(batch_size=1)
        >>> 
        >>> # Load a pattern into slot 0
        >>> pattern = torch.randn(1, 50)
        >>> wm.load(0, pattern)
        >>> 
        >>> # Step forward
        >>> for t in range(100):
        ...     output = wm()
        >>> 
        >>> # Read content
        >>> recalled = wm.read(0)
    """
    
    def __init__(self, config: Optional[WorkingMemoryConfig] = None):
        super().__init__()
        self.config = config or WorkingMemoryConfig()
        
        # Create memory slots
        self.slots = nn.ModuleList([
            MemorySlot(
                size=self.config.slot_size,
                reverb_strength=self.config.reverb_strength,
                decay_rate=self.config.decay_rate,
                tau_mem=self.config.tau_mem,
                noise_std=self.config.noise_std,
                dt=self.config.dt,
            )
            for _ in range(self.config.n_slots)
        ])
        
        # Track what's in each slot
        self.slot_contents: List[Optional[str]] = [None] * self.config.n_slots
        
        # Pending operations
        self._pending_loads: dict[int, torch.Tensor] = {}
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all slots."""
        for slot in self.slots:
            slot.reset_state(batch_size)
        self._pending_loads = {}
        
    def load(
        self,
        slot_idx: int,
        pattern: torch.Tensor,
        label: Optional[str] = None,
    ) -> None:
        """Load a pattern into a slot.
        
        Args:
            slot_idx: Which slot to load into
            pattern: Pattern to load, shape (batch, slot_size)
            label: Optional label for the content
        """
        if slot_idx < 0 or slot_idx >= self.config.n_slots:
            raise ValueError(f"Invalid slot index: {slot_idx}")
            
        self.slots[slot_idx].set_gates(load=1.0, clear=0.0)
        self._pending_loads[slot_idx] = pattern
        self.slot_contents[slot_idx] = label
        
    def clear(self, slot_idx: int) -> None:
        """Clear a slot.
        
        Args:
            slot_idx: Which slot to clear
        """
        if slot_idx < 0 or slot_idx >= self.config.n_slots:
            raise ValueError(f"Invalid slot index: {slot_idx}")
            
        self.slots[slot_idx].set_gates(load=0.0, clear=1.0)
        self.slot_contents[slot_idx] = None
        
    def clear_all(self) -> None:
        """Clear all slots."""
        for i in range(self.config.n_slots):
            self.clear(i)
            
    def read(self, slot_idx: int) -> Optional[torch.Tensor]:
        """Read content from a slot.
        
        Args:
            slot_idx: Which slot to read from
            
        Returns:
            Content pattern or None if slot is empty
        """
        if slot_idx < 0 or slot_idx >= self.config.n_slots:
            raise ValueError(f"Invalid slot index: {slot_idx}")
            
        return self.slots[slot_idx].get_content()
    
    def forward(self) -> torch.Tensor:
        """Process one timestep for all slots.
        
        Returns:
            Combined activity from all slots, shape (batch, total_neurons)
        """
        all_outputs = []
        
        for i, slot in enumerate(self.slots):
            # Check for pending load
            input_pattern = self._pending_loads.pop(i, None)
            
            # Step the slot
            spikes, _ = slot(input_pattern)
            all_outputs.append(spikes)
            
            # Reset gates after one step
            slot.set_gates(load=0.0, clear=0.0)
        
        # Concatenate all slot outputs
        return torch.cat(all_outputs, dim=-1)
    
    def get_status(self) -> dict:
        """Get status of all slots.
        
        Returns:
            Dict with slot information
        """
        status = {
            "n_slots": self.config.n_slots,
            "slots": [],
        }
        
        for i, slot in enumerate(self.slots):
            status["slots"].append({
                "index": i,
                "active": slot.is_active,
                "activity_level": slot._activity_level,
                "label": self.slot_contents[i],
            })
            
        status["active_count"] = sum(1 for s in status["slots"] if s["active"])
        status["capacity_used"] = status["active_count"] / self.config.n_slots
        
        return status
    
    def find_empty_slot(self) -> Optional[int]:
        """Find an empty slot.
        
        Returns:
            Index of first empty slot, or None if full
        """
        for i, slot in enumerate(self.slots):
            if not slot.is_active:
                return i
        return None
    
    def refresh(self, slot_idx: int, strength: float = 0.5) -> None:
        """Refresh a slot to prevent decay.
        
        This simulates attention refreshing working memory content.
        
        Args:
            slot_idx: Which slot to refresh
            strength: Refresh strength (0-1)
        """
        if slot_idx < 0 or slot_idx >= self.config.n_slots:
            raise ValueError(f"Invalid slot index: {slot_idx}")
            
        content = self.read(slot_idx)
        if content is not None:
            # Re-inject content as weak input
            self.slots[slot_idx].set_gates(load=strength, clear=0.0)
            self._pending_loads[slot_idx] = content


class WorkingMemorySNN(nn.Module):
    """Full working memory SNN with input encoding and output decoding.
    
    Integrates working memory with encoding/decoding layers for
    complete input-to-output processing with memory.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: Optional[WorkingMemoryConfig] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or WorkingMemoryConfig()
        
        # Encoding layer: maps input to slot patterns
        self.encoder = nn.Linear(input_size, self.config.slot_size)
        
        # Working memory
        self.memory = WorkingMemory(self.config)
        
        # Decoding layer: maps memory to output
        self.decoder = nn.Linear(
            self.config.total_neurons,
            output_size,
        )
        
        # Output neurons
        neuron_config = LIFConfig(tau_mem=10.0)
        self.output_neurons = LIFNeuron(n_neurons=output_size, config=neuron_config)
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all states."""
        self.memory.reset_state(batch_size)
        self.output_neurons.reset_state(batch_size)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into memory pattern."""
        return torch.relu(self.encoder(x))
    
    def decode(self, memory_output: torch.Tensor) -> torch.Tensor:
        """Decode memory output to final output."""
        return self.decoder(memory_output)
    
    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        load_slot: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.
        
        Args:
            x: Optional input to encode and load
            load_slot: Which slot to load input into (if x provided)
            
        Returns:
            output_spikes: Output layer spikes
            memory_activity: Raw memory activity
        """
        # Optionally load new input
        if x is not None and load_slot is not None:
            encoded = self.encode(x)
            self.memory.load(load_slot, encoded)
        
        # Step memory
        memory_activity = self.memory()
        
        # Decode and output
        decoded = self.decode(memory_activity)
        output_spikes, _ = self.output_neurons(decoded)
        
        return output_spikes, memory_activity

