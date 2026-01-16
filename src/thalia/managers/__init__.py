"""Component Managers.

Registry and base manager classes for brain components.
"""

from __future__ import annotations


from thalia.managers.component_registry import (
    ComponentRegistry,
    register_region,
    register_pathway,
    register_module,
)
from thalia.managers.base_manager import (
    BaseManager,
    ManagerContext,
)
from thalia.managers.base_checkpoint_manager import BaseCheckpointManager

__all__ = [
    # Component Registry
    "ComponentRegistry",
    "register_region",
    "register_pathway",
    "register_module",
    # Base Manager
    "BaseManager",
    "ManagerContext",
    # Checkpoint Manager
    "BaseCheckpointManager",
]
