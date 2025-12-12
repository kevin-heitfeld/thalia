"""Mixin classes for brain components.

This module provides reusable mixins that add capabilities to brain regions
and pathways. Mixins follow the principle of component parity - any capability
added to regions should also be available to pathways.

Available Mixins:
- DeviceMixin: Standardized device management across components
- ResettableMixin: Standard interface for resetting component state
- ConfigurableMixin: Factory method pattern for ThaliaConfig instantiation
- DiagnosticCollectorMixin: Helper methods for consistent diagnostic collection
- DiagnosticsMixin: Advanced diagnostic tracking and pathology detection
- GrowthMixin: Neuron expansion utilities and template methods
"""

from thalia.mixins.device_mixin import DeviceMixin
from thalia.mixins.resettable_mixin import ResettableMixin
from thalia.mixins.configurable_mixin import ConfigurableMixin
from thalia.mixins.diagnostic_collector_mixin import DiagnosticCollectorMixin
from thalia.mixins.diagnostics_mixin import DiagnosticsMixin
from thalia.mixins.growth_mixin import GrowthMixin

__all__ = [
    'DeviceMixin',
    'ResettableMixin',
    'ConfigurableMixin',
    'DiagnosticCollectorMixin',
    'DiagnosticsMixin',
    'GrowthMixin',
]
