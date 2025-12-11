"""
DEPRECATED: Use homeostasis_component.StriatumHomeostasisComponent instead.

This module provides backwards compatibility only. The implementation has been
moved to homeostasis_component.py following the Component Standardization Pattern.

Migration:
    Old: from .homeostasis_manager import HomeostasisManager
    New: from .homeostasis_component import StriatumHomeostasisComponent

See: docs/patterns/component-standardization.md
"""

from .homeostasis_component import (
    StriatumHomeostasisComponent as HomeostasisManager,
    HomeostasisManagerConfig,
)

__all__ = ["HomeostasisManager", "HomeostasisManagerConfig"]

# Backwards compatibility - HomeostasisManager is now StriatumHomeostasisComponent
