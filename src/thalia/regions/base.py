"""
Base classes for brain region modules.

This module previously contained LearningRule and NeuralComponentState, which have
been moved to core/ for better separation of concerns:

- LearningRule → thalia.core.learning_rules
- NeuralComponentState → thalia.core.component_state

These are core types used across the entire architecture, not region-specific.

Architecture (v3.0+):
- Brain regions inherit from NeuralRegion (thalia.core.neural_region)
- Custom pathways inherit from LearnableComponent (thalia.core.protocols.component)

Usage:
- Regions: `class MyRegion(NeuralRegion)`
- Custom pathways: `class MyPathway(LearnableComponent)`

Date: January 16, 2026 - Refactored for clearer core/regions boundary
"""

from __future__ import annotations

# Re-export for backward compatibility (to be removed in future version)
from thalia.core.learning_rules import LearningRule
from thalia.core.component_state import NeuralComponentState

__all__ = ["LearningRule", "NeuralComponentState"]
