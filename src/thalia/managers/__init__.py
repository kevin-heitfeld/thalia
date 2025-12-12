"""Component Managers.

Registry and base manager classes for brain components.
"""

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

__all__ = [
    # Component Registry
    "ComponentRegistry",
    "register_region",
    "register_pathway",
    "register_module",
    # Base Manager
    "BaseManager",
    "ManagerContext",
]
