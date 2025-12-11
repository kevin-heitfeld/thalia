"""
DEPRECATED: Use memory_component.HippocampusMemoryComponent instead.

This module provides backwards compatibility only. The implementation has been
moved to memory_component.py following the Component Standardization Pattern.

Migration:
    Old: from .episode_manager import EpisodeManager
    New: from .memory_component import HippocampusMemoryComponent

See: docs/patterns/component-standardization.md
"""

from .memory_component import HippocampusMemoryComponent as EpisodeManager

__all__ = ["EpisodeManager"]

# Backwards compatibility - EpisodeManager is now HippocampusMemoryComponent
