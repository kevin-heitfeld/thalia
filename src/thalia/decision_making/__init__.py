"""
Decision Making Module.

Provides standalone decision-making utilities that can be used by any brain
region for action selection, planning, and goal-directed behavior.

Modules:
    action_selection: Converting neural votes to discrete actions

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations


from .action_selection import (
    ActionSelector,
    ActionSelectionConfig,
    SelectionMode,
)

__all__ = [
    'ActionSelector',
    'ActionSelectionConfig',
    'SelectionMode',
]
