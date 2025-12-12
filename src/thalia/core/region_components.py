"""
Standardized Component Base Classes for Brain Regions

Provides consistent base classes for extracting complexity from region implementations.
All regions should use these standardized component types when extracting logic
into separate classes.

Architecture Pattern:
====================
When a region file grows too large (>800 lines), extract concerns into components:
- Learning logic → LearningComponent subclass
- Homeostatic regulation → HomeostasisComponent subclass
- Memory management → MemoryComponent subclass

Benefits:
- Consistent naming across regions
- Clear boundaries between concerns
- Reusable patterns
- Easier testing

Example Usage:
=============
```python
# Before: region with embedded logic
class Striatum(NeuralComponent):
    def __init__(self, config):
        # 1000+ lines of learning, homeostasis, etc.
        pass

# After: region with extracted components
class Striatum(NeuralComponent):
    def __init__(self, config):
        self.learning = StriatumLearningComponent(config, context)
        self.homeostasis = StriatumHomeostasisComponent(config, context)
        # Main region stays focused (~500 lines)
```

See Also:
=========
- docs/patterns/component-parity.md - Component protocol
- thalia.core.base_manager - BaseManager implementation
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Dict, Any, TYPE_CHECKING

import torch

from thalia.managers.base_manager import BaseManager

if TYPE_CHECKING:
    from thalia.core.base.component_config import NeuralComponentConfig


class LearningComponent(BaseManager["NeuralComponentConfig"]):
    """Base class for region learning components.

    Manages plasticity rules and weight updates for a region.
    Each region implements its own learning rule (STDP, BCM, three-factor, etc.)
    but shares the common component interface.

    Responsibilities:
    - Apply plasticity rules (STDP, Hebbian, BCM, three-factor, etc.)
    - Update synaptic weights
    - Manage learning traces (eligibility, pre/post traces)
    - Provide learning diagnostics

    Examples:
    ---------
    Striatum: Three-factor rule (eligibility × dopamine)
    Hippocampus: One-shot Hebbian + STDP
    Cortex: BCM or unsupervised STDP
    Cerebellum: Supervised error-corrective (delta rule)
    """

    @abstractmethod
    def apply_learning(self, *args, **kwargs) -> Dict[str, Any]:
        """Apply plasticity rule and update weights.

        Args:
            *args: Positional arguments (region-specific)
            **kwargs: Region-specific parameters (spikes, rewards, errors, etc.)

        Returns:
            Dict with learning metrics (weight_change, trace_magnitude, etc.)
        """
        pass

    def get_learning_diagnostics(self) -> Dict[str, Any]:
        """Get learning-specific diagnostics.

        Returns:
            Dict with learning metrics (should include weight stats, trace stats)
        """
        # Default implementation - subclasses can override
        return {
            "component_type": "learning",
            "learning_enabled": True,
        }


class HomeostasisComponent(BaseManager["NeuralComponentConfig"]):
    """Base class for homeostatic regulation components.

    Manages stability and balance mechanisms for a region.
    Prevents runaway excitation/inhibition and maintains healthy dynamics.

    Responsibilities:
    - Synaptic scaling (normalize weights)
    - Intrinsic plasticity (adjust thresholds)
    - Activity-dependent modulation
    - E/I balance maintenance

    Examples:
    ---------
    Striatum: Budget-constrained D1/D2 balance
    Hippocampus: Synaptic scaling in CA3
    Cortex: E/I balance and BCM threshold adaptation
    """

    @abstractmethod
    def apply_homeostasis(self, *args, **kwargs) -> Dict[str, Any]:
        """Apply homeostatic regulation.

        Args:
            **kwargs: Region-specific state (spikes, weights, activity history)

        Returns:
            Dict with homeostasis metrics (scaling_applied, threshold_change, etc.)
        """
        pass

    def get_homeostasis_diagnostics(self) -> Dict[str, Any]:
        """Get homeostasis-specific diagnostics.

        Returns:
            Dict with homeostasis metrics
        """
        # Default implementation - subclasses can override
        return {
            "component_type": "homeostasis",
            "homeostasis_enabled": True,
        }


class MemoryComponent(BaseManager["NeuralComponentConfig"]):
    """Base class for memory management components.

    Manages episodic or working memory storage and retrieval.
    Primarily used by hippocampus and prefrontal cortex.

    Responsibilities:
    - Store episodes/experiences
    - Retrieve relevant memories
    - Manage buffer capacity
    - Prioritize memories
    - Support replay/consolidation

    Examples:
    ---------
    Hippocampus: Episodic buffer with priority sampling
    Prefrontal: Working memory buffer with gating
    """

    @abstractmethod
    def store_memory(self, *args, **kwargs) -> None:
        """Store a memory/episode.

        Args:
            **kwargs: Memory data (state, action, reward, context, etc.)
        """
        pass

    @abstractmethod
    def retrieve_memories(self, *args, **kwargs) -> Any:
        """Retrieve relevant memories.

        Args:
            **kwargs: Query parameters (n, priority_threshold, similarity, etc.)

        Returns:
            Retrieved memories (format depends on implementation)
        """
        pass

    def get_memory_diagnostics(self) -> Dict[str, Any]:
        """Get memory-specific diagnostics.

        Returns:
            Dict with memory metrics (buffer_size, utilization, etc.)
        """
        # Default implementation - subclasses can override
        return {
            "component_type": "memory",
            "buffer_size": 0,
        }


class ExplorationComponent(BaseManager["NeuralComponentConfig"]):
    """Base class for exploration strategy components.

    Manages exploration vs exploitation tradeoffs for decision-making regions.
    Primarily used by striatum for action selection.

    Responsibilities:
    - UCB (Upper Confidence Bound) tracking
    - Adaptive exploration (tonic dopamine adjustment)
    - Action count tracking
    - Performance history

    Examples:
    ---------
    Striatum: UCB + adaptive tonic dopamine exploration
    """

    @abstractmethod
    def compute_exploration_bonus(self, *args, **kwargs) -> torch.Tensor:
        """Compute exploration bonus for actions.

        Args:
            **kwargs: Exploration parameters (action_counts, uncertainty, etc.)

        Returns:
            Exploration bonus per action
        """
        pass

    def get_exploration_diagnostics(self) -> Dict[str, Any]:
        """Get exploration-specific diagnostics.

        Returns:
            Dict with exploration metrics
        """
        # Default implementation - subclasses can override
        return {
            "component_type": "exploration",
            "exploration_enabled": True,
        }
