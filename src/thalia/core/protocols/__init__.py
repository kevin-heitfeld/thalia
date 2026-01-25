"""Core Protocols.

Protocol definitions for brain components and neural elements.
"""

from __future__ import annotations

from .component import (
    BrainComponent,
    BrainComponentMixin,
    RoutingComponent,
)

__all__ = [
    # Component Protocol
    "BrainComponent",
    "BrainComponentMixin",
    "RoutingComponent",
]
