"""
DEPRECATED: Use learning_component.HippocampusLearningComponent instead.

This module provides backwards compatibility only. The implementation has been
moved to learning_component.py following the Component Standardization Pattern.

Migration:
    Old: from .plasticity_manager import PlasticityManager
    New: from .learning_component import HippocampusLearningComponent

See: docs/patterns/component-standardization.md
"""

from .learning_component import HippocampusLearningComponent as PlasticityManager

__all__ = ["PlasticityManager"]

# Backwards compatibility - PlasticityManager is now HippocampusLearningComponent
