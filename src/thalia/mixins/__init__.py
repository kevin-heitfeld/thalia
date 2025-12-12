"""Mixin classes for brain components.

This module provides reusable mixins that add capabilities to brain regions
and pathways. Mixins follow the principle of component parity - any capability
added to regions should also be available to pathways.

Available Mixins:
- GrowthMixin: Neuron expansion utilities and template methods
"""

from thalia.mixins.growth_mixin import GrowthMixin

__all__ = ['GrowthMixin']
