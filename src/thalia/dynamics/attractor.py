"""
Attractor network for pattern storage and recall.

Implements a Hopfield-like spiking attractor network where:
- Patterns are stable states of neural activity
- Partial patterns complete to full stored patterns
- Noise can drive transitions between attractors

This is the foundation for concept representation in THALIA.
Thoughts are trajectories through attractor landscapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.synapse import Synapse, SynapseConfig, SynapseType


@dataclass
class AttractorConfig:
    """Configuration for attractor network.
    
    Attributes:
        n_neurons: Number of neurons in the network
        tau_mem: Membrane time constant
        noise_std: Noise for spontaneous transitions
        inhibition_strength: Global inhibition to create competition
        excitation_strength: Base excitatory connection strength  
        sparsity: Target sparsity of patterns (fraction active)
        dt: Simulation timestep in ms
    """
    n_neurons: int = 100
    tau_mem: float = 20.0
    noise_std: float = 0.05
    inhibition_strength: float = 0.1
    excitation_strength: float = 0.5
    sparsity: float = 0.1  # 10% of neurons active per pattern
    dt: float = 1.0


class AttractorNetwork(nn.Module):
    """Spiking attractor network for pattern storage and recall.
    
    Patterns are stored as stable activity states. The network
    uses Hebbian-like learning to strengthen connections between
    co-active neurons, creating basins of attraction.
    
    Args:
        config: Network configuration
        
    Example:
        >>> net = AttractorNetwork(AttractorConfig(n_neurons=100))
        >>> 
        >>> # Store patterns
        >>> pattern1 = (torch.rand(100) < 0.1).float()  # 10% active
        >>> pattern2 = (torch.rand(100) < 0.1).float()
        >>> net.store_pattern(pattern1)
        >>> net.store_pattern(pattern2)
        >>> 
        >>> # Recall from partial cue
        >>> cue = pattern1.clone()
        >>> cue[50:] = 0  # Only first half
        >>> recalled = net.recall(cue, steps=100)
    """
    
    def __init__(self, config: Optional[AttractorConfig] = None):
        super().__init__()
        self.config = config or AttractorConfig()
        n = self.config.n_neurons
        
        # Create neurons with noise for spontaneous activity
        neuron_config = LIFConfig(
            tau_mem=self.config.tau_mem,
            noise_std=self.config.noise_std,
            dt=self.config.dt,
        )
        self.neurons = LIFNeuron(n_neurons=n, config=neuron_config)
        
        # Recurrent weight matrix (learnable)
        # Initialize with small random values
        self.weights = nn.Parameter(torch.randn(n, n) * 0.01)
        
        # Mask out self-connections
        self.register_buffer("self_mask", 1 - torch.eye(n))
        
        # Global inhibition for competition
        self.register_buffer(
            "inhibition", 
            torch.ones(n, n) * (-self.config.inhibition_strength / n)
        )
        
        # Stored patterns for reference
        self.patterns: List[torch.Tensor] = []
        
        # State tracking
        self.activity_history: List[torch.Tensor] = []
        
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset network state."""
        self.neurons.reset_state(batch_size)
        self.activity_history = []
        
    def get_effective_weights(self) -> torch.Tensor:
        """Get weight matrix with self-connections masked and inhibition added."""
        # Mask self-connections and add global inhibition
        w = self.weights * self.self_mask + self.inhibition * self.self_mask
        return w
    
    def store_pattern(self, pattern: torch.Tensor) -> None:
        """Store a pattern using Hebbian learning.
        
        Updates weights: ΔW = η * (pattern ⊗ pattern - W)
        This creates an attractor at the pattern state.
        
        Args:
            pattern: Binary pattern to store, shape (n_neurons,)
        """
        pattern = pattern.float()
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
            
        # Normalize pattern to have target sparsity
        # (optional: could enforce sparsity here)
        
        # Outer product: Hebbian association
        # Neurons that fire together wire together
        outer = torch.outer(pattern.squeeze(), pattern.squeeze())
        
        # Update weights (simple Hebbian with decay toward outer)
        learning_rate = self.config.excitation_strength / max(1, len(self.patterns) + 1)
        with torch.no_grad():
            self.weights.data += learning_rate * (outer - self.weights.data * outer.bool().float())
        
        # Store pattern for reference
        self.patterns.append(pattern.squeeze().clone())
        
    def store_patterns(self, patterns: torch.Tensor) -> None:
        """Store multiple patterns.
        
        Args:
            patterns: Patterns to store, shape (n_patterns, n_neurons)
        """
        for i in range(patterns.shape[0]):
            self.store_pattern(patterns[i])
    
    def forward(
        self, 
        external_input: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.
        
        Args:
            external_input: Optional external current, shape (batch, n_neurons)
            
        Returns:
            spikes: Spike output, shape (batch, n_neurons)
            membrane: Membrane potentials, shape (batch, n_neurons)
        """
        # Get last activity (or initialize)
        if self.neurons.membrane is None:
            batch_size = external_input.shape[0] if external_input is not None else 1
            self.reset_state(batch_size)
        
        # Compute recurrent input from last spikes
        if hasattr(self, '_last_spikes') and self._last_spikes is not None:
            weights = self.get_effective_weights()
            recurrent_input = torch.matmul(self._last_spikes, weights)
        else:
            recurrent_input = torch.zeros_like(self.neurons.membrane)
        
        # Total input
        total_input = recurrent_input
        if external_input is not None:
            total_input = total_input + external_input
            
        # Step neurons
        spikes, membrane = self.neurons(total_input)
        
        # Store for next step
        self._last_spikes = spikes
        
        # Track activity
        self.activity_history.append(spikes.detach().clone())
        
        return spikes, membrane
    
    def recall(
        self, 
        cue: torch.Tensor, 
        steps: int = 100,
        cue_strength: float = 1.0,
        cue_duration: int = 10,
    ) -> torch.Tensor:
        """Recall a pattern from a partial cue.
        
        The network is initialized with the cue and allowed to settle
        into an attractor state.
        
        Args:
            cue: Partial pattern cue, shape (n_neurons,) or (batch, n_neurons)
            steps: Number of timesteps to simulate
            cue_strength: Strength of cue input current
            cue_duration: How long to apply cue (steps)
            
        Returns:
            Recalled pattern (mean activity), shape (batch, n_neurons)
        """
        if cue.dim() == 1:
            cue = cue.unsqueeze(0)
        batch_size = cue.shape[0]
        
        self.reset_state(batch_size)
        
        # Apply cue as input current for first cue_duration steps
        activity_sum = torch.zeros_like(cue)
        
        for t in range(steps):
            if t < cue_duration:
                external = cue * cue_strength
            else:
                external = None
            
            spikes, _ = self.forward(external)
            
            # Accumulate activity (after initial transient)
            if t >= steps // 2:
                activity_sum += spikes
        
        # Return mean activity as recalled pattern
        mean_activity = activity_sum / (steps // 2)
        
        return mean_activity
    
    def similarity_to_patterns(self, activity: torch.Tensor) -> torch.Tensor:
        """Compute similarity of activity to stored patterns.
        
        Args:
            activity: Current activity, shape (batch, n_neurons) or (n_neurons,)
            
        Returns:
            Similarity scores, shape (batch, n_patterns) or (n_patterns,)
        """
        if len(self.patterns) == 0:
            return torch.tensor([])
            
        if activity.dim() == 1:
            activity = activity.unsqueeze(0)
            
        patterns = torch.stack(self.patterns)  # (n_patterns, n_neurons)
        
        # Cosine similarity
        activity_norm = activity / (activity.norm(dim=-1, keepdim=True) + 1e-8)
        patterns_norm = patterns / (patterns.norm(dim=-1, keepdim=True) + 1e-8)
        
        similarity = torch.matmul(activity_norm, patterns_norm.T)
        
        return similarity.squeeze()
    
    def get_attractor_state(self) -> int:
        """Identify which attractor the network is closest to.
        
        Returns:
            Index of closest stored pattern, or -1 if no patterns stored
        """
        if len(self.patterns) == 0 or len(self.activity_history) == 0:
            return -1
            
        # Use recent average activity
        recent = torch.stack(self.activity_history[-20:]).mean(dim=0)
        sim = self.similarity_to_patterns(recent)
        
        return sim.argmax().item()
    
    def energy(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute network energy (lower = more stable).
        
        E = -0.5 * sum_ij W_ij * s_i * s_j
        
        Args:
            state: Activity state, or uses current if None
            
        Returns:
            Energy value (scalar)
        """
        if state is None:
            if len(self.activity_history) == 0:
                return torch.tensor(0.0)
            state = self.activity_history[-1]
            
        if state.dim() > 1:
            state = state.squeeze()
            
        weights = self.get_effective_weights()
        energy = -0.5 * torch.sum(weights * torch.outer(state, state))
        
        return energy


class ConceptNetwork(AttractorNetwork):
    """Extended attractor network for concept representation.
    
    Adds:
    - Named concepts with semantic labels
    - Inter-concept associations
    - Concept hierarchies
    """
    
    def __init__(self, config: Optional[AttractorConfig] = None):
        super().__init__(config)
        self.concept_names: List[str] = []
        self.associations: dict[tuple[int, int], float] = {}
        
    def store_concept(self, pattern: torch.Tensor, name: str) -> int:
        """Store a named concept.
        
        Args:
            pattern: Activity pattern for the concept
            name: Human-readable concept name
            
        Returns:
            Index of the stored concept
        """
        self.store_pattern(pattern)
        self.concept_names.append(name)
        return len(self.patterns) - 1
    
    def associate(self, concept_a: int, concept_b: int, strength: float = 1.0) -> None:
        """Create association between concepts.
        
        When concept_a is active, it will tend to activate concept_b.
        
        Args:
            concept_a: Index of first concept
            concept_b: Index of second concept
            strength: Association strength
        """
        if concept_a >= len(self.patterns) or concept_b >= len(self.patterns):
            raise ValueError("Concept index out of range")
            
        # Strengthen connections between the two patterns
        pattern_a = self.patterns[concept_a]
        pattern_b = self.patterns[concept_b]
        
        # Cross-pattern Hebbian: neurons in A connect to neurons in B
        outer = torch.outer(pattern_a, pattern_b)
        
        with torch.no_grad():
            self.weights.data += strength * 0.1 * outer
            
        self.associations[(concept_a, concept_b)] = strength
        self.associations[(concept_b, concept_a)] = strength  # Bidirectional
    
    def get_concept_name(self, index: int) -> str:
        """Get name of concept by index."""
        if 0 <= index < len(self.concept_names):
            return self.concept_names[index]
        return f"concept_{index}"
    
    def active_concept(self) -> tuple[int, str]:
        """Get currently active concept.
        
        Returns:
            (index, name) of most active concept
        """
        idx = self.get_attractor_state()
        name = self.get_concept_name(idx) if idx >= 0 else "none"
        return idx, name
