"""
Neural Pathway Protocol - Unified interface for all pathway types.

This module defines the NeuralPathway protocol that standardizes the interface
across sensory pathways, inter-region pathways, and specialized pathways.

Biological Motivation:
=====================

Neural pathways in the brain share common computational patterns:
1. **Transform Information**: Map input activity to output activity
2. **Adapt Through Experience**: Learn continuously during forward passes (like regions)
3. **Maintain State**: Track temporal dynamics and history
4. **Provide Diagnostics**: Report health and activity metrics

Types of Pathways:
==================

1. **Sensory Pathways** (SensoryPathway):
   - Transform raw sensory input → spike patterns
   - Examples: Visual (retina→V1), Auditory (cochlea→A1), Language (tokens→spikes)
   - Primary method: encode()

2. **Inter-Region Pathways** (SpikingPathway):
   - Transform spikes between brain regions
   - Examples: Cortex→Hippocampus, Hippocampus→Cortex
   - Primary method: forward()
   - Learning: Automatic STDP during every forward pass (always learning)

3. **Specialized Pathways**:
   - SpikingAttentionPathway: PFC→Cortex attention modulation
   - SpikingReplayPathway: Hippocampus→Cortex memory consolidation
   - Inherit from SpikingPathway with specialized forward() behavior

Protocol Design:
================

The protocol allows for flexibility while ensuring consistency:
- All pathways must implement forward() OR encode()
- Learning is optional (some pathways learn, others don't)
- State management and diagnostics are required
- Type hints enable static checking

Usage Example:
==============
    def process_pathway(pathway: NeuralPathway, input_data: Any) -> torch.Tensor:
        # Works with any pathway type
        if hasattr(pathway, 'encode'):
            output, metadata = pathway.encode(input_data)
        else:
            output = pathway.forward(input_data)
        
        return output
    
    # Reset all pathways uniformly
    def reset_all(pathways: List[NeuralPathway]):
        for pathway in pathways:
            pathway.reset_state()

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class NeuralPathway(Protocol):
    """
    Protocol defining the unified interface for all neural pathways.
    
    All pathways (sensory, inter-region, specialized) implement this interface
    to ensure consistent usage patterns across the codebase.
    
    Core Methods:
    -------------
    - forward() or encode(): Transform input to output (learning happens automatically)
    - reset_state(): Reset temporal state
    - get_diagnostics(): Report pathway metrics
    
    Design Rationale:
    -----------------
    1. **Sensory pathways** use encode() (raw input → spikes)
    2. **Inter-region pathways** use forward() (spikes → spikes)
    3. Both can coexist since they serve different purposes
    4. **Pathways always learn** during forward/encode (like regions)
    5. Learning is via STDP, BCM, or other plasticity rules applied automatically
    """
    
    def forward(
        self,
        input_data: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Transform input to output (primary method for inter-region pathways).
        
        Args:
            input_data: Input tensor (typically spikes)
            **kwargs: Additional arguments (dt, time_ms, etc.)
            
        Returns:
            output: Output tensor, or (output, metadata) tuple
            
        Note:
            Sensory pathways may implement encode() instead of forward().
            Inter-region pathways always implement forward().
        """
        ...
    
    def reset_state(self) -> None:
        """
        Reset pathway temporal state.
        
        Clears:
        - Synaptic traces
        - Membrane potentials
        - Delay buffers
        - Adaptation state
        - Any other temporal dynamics
        
        Call this between trials/sequences to ensure clean state.
        """
        ...
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get pathway diagnostics and metrics.
        
        Returns:
            Dictionary containing:
            - Activity metrics (spike rates, membrane stats)
            - Learning metrics (weight changes, STDP traces)
            - Health indicators (weights in bounds, no NaNs)
            - Pathway-specific metrics
        """
        ...


@runtime_checkable
class SensoryPathwayProtocol(NeuralPathway, Protocol):
    """
    Extended protocol for sensory pathways.
    
    Sensory pathways transform raw sensory input (images, audio, tokens)
    into spike patterns. They use encode() as their primary method,
    but may also implement forward() for convenience.
    """
    
    def encode(
        self,
        raw_input: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode raw sensory input to spike patterns.
        
        Args:
            raw_input: Modality-specific input
                - Visual: [batch, channels, height, width]
                - Auditory: [batch, time_samples]
                - Language: [batch, seq_len] (token IDs)
            **kwargs: Modality-specific parameters
            
        Returns:
            spikes: Spike patterns [batch, n_timesteps, output_size]
                   (or [batch, seq_len, n_timesteps, output_size] for sequences)
            metadata: Dictionary with encoding statistics
        """
        ...
    
    def get_modality(self) -> Any:  # Returns Modality enum
        """
        Return the sensory modality type.
        
        Returns:
            Modality enum (VISUAL, AUDITORY, LANGUAGE, etc.)
        """
        ...


# Note: LearnablePathway protocol removed - pathways ALWAYS learn during forward/encode.
# Learning happens automatically via STDP, BCM, or other plasticity rules,
# just like regions (Prefrontal, Hippocampus, etc.) always learn.
# No separate learn() method needed.


# =============================================================================
# Base Class for Pathways
# =============================================================================

class BaseNeuralPathway(nn.Module, ABC):
    """
    Abstract base class for all neural pathways.
    
    This provides a concrete base class that pathways should inherit from,
    implementing the NeuralPathway protocol interface. Unlike the protocol
    (which uses duck typing), inheriting from this base class makes the
    relationship explicit and provides better IDE support.
    
    All subclasses must implement:
    - forward() or encode(): Primary transformation
    - reset_state(): Clear temporal state
    - get_diagnostics(): Report metrics
    
    Learning happens automatically during forward/encode passes.
    """
    
    @abstractmethod
    def reset_state(self) -> None:
        """Reset pathway temporal state. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get pathway diagnostics. Must be implemented by subclasses."""
        pass
    
    # Note: forward() is already abstract in nn.Module, so no need to redeclare


# Type aliases for convenience
Pathway = Union[NeuralPathway, SensoryPathwayProtocol]


__all__ = [
    "NeuralPathway",
    "SensoryPathwayProtocol",
    "BaseNeuralPathway",
    "Pathway",
]
