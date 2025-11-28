"""
Hierarchical multi-layer architectures.

This module provides hierarchical SNNs with multiple temporal scales:
- Fast layers for sensory processing
- Slow layers for abstract reasoning
- Bidirectional connections for top-down modulation
"""

from thalia.hierarchy.hierarchical import (
    LayerConfig,
    HierarchicalConfig,
    HierarchicalLayer,
    HierarchicalSNN,
)

__all__ = [
    "LayerConfig",
    "HierarchicalConfig",
    "HierarchicalLayer",
    "HierarchicalSNN",
]
