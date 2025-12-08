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
   - Primary method: forward() (standard PyTorch convention, ADR-007)

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
- All pathways must implement forward() (standard PyTorch convention, ADR-007)
- Learning is optional (some pathways learn, others don't)
- State management and diagnostics are required
- Type hints enable static checking

Usage Example:
==============
    def process_pathway(pathway: NeuralPathway, input_data: Any) -> torch.Tensor:
        # Works with any pathway type - all use forward()
        # Callable syntax (ADR-007):
        output = pathway(input_data)
        # Or explicit:
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
    - forward(): Transform input to output (standard PyTorch, ADR-007)
    - reset_state(): Reset temporal state
    - get_diagnostics(): Report pathway metrics
    
    Design Rationale:
    -----------------
    1. **All pathways** use forward() (standard PyTorch convention, ADR-007)
    2. **Sensory pathways**: forward(raw_input) → (spikes, metadata)
    3. **Inter-region pathways**: forward(spikes) → spikes
    4. **Pathways always learn** during forward passes (like regions)
    5. Learning is via STDP, BCM, or other plasticity rules applied automatically
    """
    
    def forward(
        self,
        input_data: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Transform input to output (standard PyTorch convention, ADR-007).
        
        All pathways use forward() to enable callable syntax:
        >>> output = pathway(input_data)  # Calls forward() automatically
        
        Args:
            input_data: Input tensor
                - Inter-region pathways: spikes [n_neurons]
                - Sensory pathways: raw input (image, audio, tokens)
            **kwargs: Additional arguments (dt, time_ms, etc.)
            
        Returns:
            output: Output tensor, or (output, metadata) tuple
                - Inter-region: spikes [n_neurons]
                - Sensory: (spikes [n_timesteps, n_neurons], metadata)
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


# Note: SensoryPathwayProtocol removed (ADR-007)
# All pathways (sensory and inter-region) use forward() for consistency.
# Sensory pathways return (spikes, metadata) tuple from forward().
# Inter-region pathways return spikes tensor from forward().

# Note: LearnablePathway protocol removed - pathways ALWAYS learn during forward passes.
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
    
    COMPONENT PROTOCOL
    ==================
    BaseNeuralPathway implements the BrainComponent protocol, which defines
    the unified interface shared with BrainRegion. This ensures feature parity
    between pathways and regions.
    
    **CRITICAL**: Pathways are just as important as regions!
    Both are active learning components that need:
    - Growth support (add_neurons, get_capacity_metrics)
    - Diagnostics (get_diagnostics, check_health)
    - Checkpointing (get_full_state, load_full_state)
    - Continuous learning during forward passes
    
    When features are added to BrainRegion, they MUST be added to
    BaseNeuralPathway as well. The protocol enforces this.
    
    All components use forward() for processing (standard PyTorch, ADR-007).
    
    See: src/thalia/core/component_protocol.py
         docs/patterns/component-parity.md
         docs/decisions/adr-007-pytorch-consistency.md
    
    All subclasses must implement:
    - forward(): Primary transformation (standard PyTorch convention)
    - reset_state(): Clear temporal state
    - get_diagnostics(): Report metrics
    
    Learning happens automatically during forward passes.
    """
    
    @abstractmethod
    def reset_state(self) -> None:
        """Reset pathway temporal state. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get pathway diagnostics. Must be implemented by subclasses."""
        pass
    
    def check_health(self) -> 'HealthReport':
        """Check for pathological states in pathway.
        
        Detects:
        - Zero output: No spikes passing through
        - Weight saturation: Too many weights at limits
        - Dead connections: Unused pathways
        
        Returns:
            HealthReport with detected issues
        """
        from thalia.diagnostics.health_monitor import HealthReport, IssueReport, IssueSeverity
        
        issues = []
        diagnostics = self.get_diagnostics()
        
        # Check for zero output
        if 'output_spike_rate' in diagnostics:
            rate = diagnostics['output_spike_rate']
            if rate < 0.001:  # Less than 0.1%
                issues.append(IssueReport(
                    severity=IssueSeverity.MEDIUM,
                    issue_type='silence',
                    message=f'Pathway output very low: {rate:.3%}',
                    suggested_fix='Check input activity, weights, or pathway gating'
                ))
        
        # Check weight statistics
        if 'weight_stats' in diagnostics:
            stats = diagnostics['weight_stats']
            mean = stats.get('mean', 0)
            if abs(mean) < 0.01:
                issues.append(IssueReport(
                    severity=IssueSeverity.LOW,
                    issue_type='weak_connections',
                    message=f'Very weak pathway weights: mean={mean:.3f}',
                    suggested_fix='May need weight reinitialization or stronger learning'
                ))
        
        # Create report
        is_healthy = len(issues) == 0
        overall_severity = max([issue.severity.value for issue in issues]) if issues else 0.0
        
        if is_healthy:
            summary = f"{self.__class__.__name__}: Healthy"
        else:
            summary = f"{self.__class__.__name__}: {len(issues)} issue(s) detected"
        
        return HealthReport(
            is_healthy=is_healthy,
            overall_severity=overall_severity,
            issues=issues,
            summary=summary,
            metrics=diagnostics
        )
    
    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to pathway for growth and capacity expansion.
        
        This is a default implementation that should be overridden by
        specific pathway implementations for proper weight matrix expansion.
        
        Pathways need growth support because they connect regions that may
        themselves grow. When a region adds neurons, connected pathways must
        expand their weight matrices to accommodate the new connections,
        preserve existing learned weights, and maintain capacity for continued learning.
        
        Args:
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
            
        Raises:
            NotImplementedError: If pathway doesn't support growth
        """
        raise NotImplementedError(
            f"Pathway {self.__class__.__name__} does not implement add_neurons(). "
            "Growth support requires pathway-specific implementation."
        )
    
    def get_capacity_metrics(self) -> Dict[str, float]:
        """Get capacity utilization metrics for growth decisions.
        
        Default implementation provides basic metrics. Pathways can override
        for more sophisticated analysis.
        
        Returns:
            Dict with metrics:
            - firing_rate: Average firing rate (0-1)
            - weight_saturation: Fraction of weights near max
            - synapse_usage: Fraction of active synapses
            - synapse_count: Total synapses
            - growth_recommended: Whether growth is advised
        """
        from ..core.growth import GrowthManager
        
        # Use GrowthManager for standard metrics computation
        manager = GrowthManager(region_name=self.__class__.__name__)
        metrics = manager.get_capacity_metrics(self)
        return metrics.to_dict()
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get complete pathway state for checkpointing.
        
        Default implementation captures common pathway state.
        Pathways with specialized state should override this.
        
        Returns:
            Dictionary with keys:
            - 'weights': Dict[str, torch.Tensor] - All learnable parameters
            - 'pathway_state': Dict[str, Any] - Dynamic state (traces, etc.)
            - 'class_name': str - Pathway class for reconstruction
            - 'diagnostics': Dict[str, Any] - Current metrics
        """
        state = {
            'weights': {
                name: param.detach().clone()
                for name, param in self.named_parameters()
            },
            'pathway_state': {},  # Subclasses should populate this
            'class_name': self.__class__.__name__,
            'diagnostics': self.get_diagnostics(),
        }
        return state
    
    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete pathway state from checkpoint.
        
        Default implementation restores weights.
        Pathways with specialized state should override this.
        
        Args:
            state: Dictionary from get_full_state()
            
        Raises:
            ValueError: If state is incompatible
        """
        # Verify class matches
        if state.get('class_name') != self.__class__.__name__:
            raise ValueError(
                f"State class mismatch: expected {self.__class__.__name__}, "
                f"got {state.get('class_name')}"
            )
        
        # Restore weights
        weights = state.get('weights', {})
        for name, param in self.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])
    
    # Note: forward() is already abstract in nn.Module, so no need to redeclare


# Type alias for convenience (ADR-007: all pathways use NeuralPathway protocol)
Pathway = NeuralPathway


__all__ = [
    "NeuralPathway",
    "BaseNeuralPathway",
    "Pathway",
]
