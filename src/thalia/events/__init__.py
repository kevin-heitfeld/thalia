"""
Event-Driven Neural Simulation Package.

This package provides the foundation for asynchronous, event-driven brain simulation
where regions operate independently and communicate via spike events with realistic
axonal delays.

Subpackages:
===========
- system: Core event scheduling and management
- parallel: Parallel execution framework for multi-core simulation
- adapters: Event-driven wrappers for brain regions

Key Concepts:
============

1. EVENT-DRIVEN PROCESSING:
   - Regions only compute when they receive events (spikes or neuromodulators)
   - Between events, membrane potentials decay analytically (no wasted computation)
   - This mirrors how real neurons work: they're event-driven, not clock-driven

2. AXONAL DELAYS:
   - Spike propagation takes time (1-20ms depending on distance)
   - Different pathways have different delays
   - This creates natural temporal dynamics and phase relationships

3. PARALLELISM:
   - Events at the same time can be processed in parallel
   - Different regions can run on different processes/machines
   - Scales naturally with number of regions

Usage:
======

    from thalia.events import (
        # Event system
        Event, EventType, EventScheduler,
        SpikePayload, Connection,
        get_axonal_delay, AXONAL_DELAYS,
        # Parallel execution
        ParallelExecutor,
        # Region adapters
        EventDrivenCortex, EventDrivenHippocampus,
        EventDrivenPFC, EventDrivenStriatum,
        EventDrivenCerebellum, EventRegionConfig,
        create_event_driven_brain,
    )

Author: Thalia Project
Date: December 2025
"""

# Event system core
from thalia.events.system import (
    # Event types
    Event,
    EventType,
    SpikePayload,
    Connection,
    # Event scheduling
    EventScheduler,
    RegionInterface,
    get_axonal_delay,
    AXONAL_DELAYS,
)

# Parallel execution
from thalia.events.parallel import (
    ParallelExecutor,
    RegionWorkerConfig,
)

# Region adapters
from thalia.events.adapters import (
    # Base classes
    EventRegionConfig,
    EventDrivenRegionBase,
    # Region adapters
    EventDrivenCortex,
    EventDrivenHippocampus,
    EventDrivenPFC,
    EventDrivenStriatum,
    EventDrivenCerebellum,
    # Factory
    create_event_driven_brain,
)

__all__ = [
    # Event types
    "Event",
    "EventType",
    "SpikePayload",
    "Connection",
    # Event scheduling
    "EventScheduler",
    "RegionInterface",
    "get_axonal_delay",
    "AXONAL_DELAYS",
    # Parallel execution
    "ParallelExecutor",
    "RegionWorkerConfig",
    # Base classes
    "EventRegionConfig",
    "EventDrivenRegionBase",
    # Region adapters
    "EventDrivenCortex",
    "EventDrivenHippocampus",
    "EventDrivenPFC",
    "EventDrivenStriatum",
    "EventDrivenCerebellum",
    # Factory
    "create_event_driven_brain",
]
