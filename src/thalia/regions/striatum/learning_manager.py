"""
DEPRECATED: Use learning_component.StriatumLearningComponent instead.

This module provides backwards compatibility only. The implementation has been
moved to learning_component.py following the Component Standardization Pattern.

Migration:
    Old: from .learning_manager import LearningManager
    New: from .learning_component import StriatumLearningComponent

See: docs/patterns/component-standardization.md
"""

from .learning_component import StriatumLearningComponent as LearningManager

__all__ = ["LearningManager"]

# Backwards compatibility - LearningManager is now StriatumLearningComponent
