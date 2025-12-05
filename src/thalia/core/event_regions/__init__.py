"""
Event-Driven Region Adapters Package.

This package provides adapters that wrap existing brain regions to work
with the event-driven simulation framework. These adapters handle:

1. Event translation: Convert events to region-specific inputs
2. Membrane decay: Apply decay between events (no wasted computation)
3. Output routing: Create events with appropriate delays
4. State tracking: Track last update time for decay calculation

The adapters allow gradual migration from the sequential BrainSystem
to the parallel event-driven architecture.

Usage:
    from thalia.core.event_regions import (
        EventDrivenCortex,
        EventDrivenHippocampus,
        EventDrivenPFC,
        EventDrivenStriatum,
        EventDrivenCerebellum,
        EventRegionConfig,
        create_event_driven_brain,
    )

Author: Thalia Project
Date: December 2025
"""

from .base import (
    EventRegionConfig,
    EventDrivenRegionBase,
    SimpleLIFRegion,
)
from .cortex import EventDrivenCortex
from .hippocampus import EventDrivenHippocampus
from .pfc import EventDrivenPFC
from .striatum import EventDrivenStriatum
from .cerebellum import EventDrivenCerebellum
from .factory import create_event_driven_brain

__all__ = [
    # Base classes
    "EventRegionConfig",
    "EventDrivenRegionBase",
    "SimpleLIFRegion",
    # Region adapters
    "EventDrivenCortex",
    "EventDrivenHippocampus",
    "EventDrivenPFC",
    "EventDrivenStriatum",
    "EventDrivenCerebellum",
    # Factory
    "create_event_driven_brain",
]
