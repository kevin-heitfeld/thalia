"""
THALIA - Thinking Architecture via Learning Integrated Attractors

A framework for building genuinely thinking spiking neural networks.
"""

from __future__ import annotations

__version__ = "0.1.0"

# ============================================================================
# PUBLIC API
# ============================================================================

# Visualization (optional - requires manim)
# try:
#     from thalia.visualization import BrainActivityVisualization, MANIM_AVAILABLE
# except ImportError:
#     BrainActivityVisualization = None
#     MANIM_AVAILABLE = False
MANIM_AVAILABLE = False

__all__ = [
    # Version
    "__version__",
    # Visualization
    "MANIM_AVAILABLE",
]
