"""
Event-Driven Neural Simulation Framework.

This module provides the foundation for asynchronous, event-driven brain simulation
where regions operate independently and communicate via spike events with realistic
axonal delays.

Key Concepts:
=============

1. EVENT-DRIVEN PROCESSING:
   - Regions only compute when they receive events (spikes or neuromodulators)
   - Between events, membrane potentials decay analytically (no wasted computation)
   - This mirrors how real neurons work: they're event-driven, not clock-driven

2. AXONAL DELAYS:
   - Spike propagation takes time (1-20ms depending on distance)
   - Different pathways have different delays
   - This creates natural temporal dynamics and phase relationships

3. GLOBAL RHYTHMS (Theta):
   - Generated centrally (medial septum in biology)
   - Broadcast to regions with appropriate delays
   - Regions receive theta at different phases (biologically accurate!)

4. PARALLELISM:
   - Events at the same time can be processed in parallel
   - Different regions can run on different processes/machines
   - Scales naturally with number of regions

Architecture:
=============

    ┌─────────────────────────────────────────────────────────┐
    │                    EVENT SCHEDULER                       │
    │  • Priority queue sorted by event time                   │
    │  • Dispatches events to appropriate regions              │
    │  • Handles theta rhythm broadcasting                     │
    └─────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ Region  │          │ Region  │          │ Region  │
    │  Actor  │◄────────►│  Actor  │◄────────►│  Actor  │
    └─────────┘          └─────────┘          └─────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                    Spike events with delays

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
import torch


class EventType(Enum):
    """Types of events in the simulation."""
    SPIKE = "spike"           # Spike activity from a region
    THETA = "theta"           # Theta rhythm phase update
    DOPAMINE = "dopamine"     # Dopamine neuromodulator signal
    SENSORY = "sensory"       # External sensory input
    REWARD = "reward"         # Reward signal
    CLOCK = "clock"           # Periodic clock tick (for regions that need it)


@dataclass(order=True)
class Event:
    """An event in the simulation.

    Events are ordered by time for priority queue processing.
    The payload contains the actual data and is not used for ordering.
    """
    time: float                          # Simulated time in ms
    event_type: EventType = field(compare=False)
    source: str = field(compare=False)   # Source region/system
    target: str = field(compare=False)   # Target region
    payload: Any = field(compare=False)  # Event-specific data


@dataclass
class SpikePayload:
    """Payload for spike events."""
    spikes: torch.Tensor                 # Spike vector (binary)
    source_layer: Optional[str] = None   # e.g., "L5", "CA1" for layer-specific routing


@dataclass
class ThetaPayload:
    """Payload for theta rhythm events."""
    phase: float                         # Current theta phase (0 to 2π)
    frequency: float = 8.0               # Theta frequency in Hz
    encoding_strength: float = 0.0       # Encoding modulation (high at trough)
    retrieval_strength: float = 0.0      # Retrieval modulation (high at peak)


# =============================================================================
# Trial Phase Enum (moved from theta_dynamics.py)
# =============================================================================

class TrialPhase(Enum):
    """Phase of a memory task trial.
    
    These phases map to different theta-modulated computations:
    - ENCODE: Sample presentation, Hebbian learning enabled (theta trough)
    - DELAY: No stimulus, CA3 maintains via recurrence (theta continues)
    - RETRIEVE: Test presentation, NMDA comparison (theta peak)
    """
    ENCODE = "encode"
    DELAY = "delay"
    RETRIEVE = "retrieve"


@dataclass
class DopaminePayload:
    """Payload for dopamine events."""
    level: float                         # Dopamine level (-1 to +1 typically)
    is_burst: bool = False               # True for phasic burst
    is_dip: bool = False                 # True for phasic dip


@dataclass
class Connection:
    """A connection between two regions with axonal delay.

    Represents an axonal pathway with:
    - Source and target regions
    - Conduction delay (based on distance/myelination)
    - Optional layer-specific routing
    """
    source: str                          # Source region name
    target: str                          # Target region name
    delay_ms: float                      # Axonal conduction delay
    source_layer: Optional[str] = None   # Source layer (e.g., "L5")
    target_layer: Optional[str] = None   # Target layer (e.g., "L4")
    weight: float = 1.0                  # Connection strength


# Standard axonal delays based on neuroscience literature
# Values in milliseconds
AXONAL_DELAYS = {
    # Sensory pathways
    ("sensory", "cortex"): 5.0,          # Thalamic relay

    # Cortical local circuits
    ("cortex_L4", "cortex_L23"): 1.0,    # Feedforward
    ("cortex_L23", "cortex_L5"): 1.0,    # Feedforward
    ("cortex_L23", "cortex_L23"): 0.5,   # Lateral/recurrent

    # Cortico-subcortical
    ("cortex", "hippocampus"): 3.0,      # Via entorhinal cortex
    ("cortex", "striatum"): 5.0,         # Direct projection
    ("cortex", "pfc"): 6.0,              # Long-range cortical

    # Hippocampal internal
    ("hippocampus_DG", "hippocampus_CA3"): 1.0,
    ("hippocampus_CA3", "hippocampus_CA1"): 1.0,
    ("hippocampus_CA3", "hippocampus_CA3"): 0.5,  # Recurrent

    # Hippocampal outputs
    ("hippocampus", "pfc"): 5.0,
    ("hippocampus", "cortex"): 4.0,      # Replay pathway

    # PFC
    ("pfc", "cortex"): 8.0,              # Top-down feedback
    ("pfc", "striatum"): 4.0,
    ("pfc", "hippocampus"): 5.0,

    # Striatum
    ("striatum", "motor"): 5.0,

    # Theta rhythm (from medial septum)
    ("septum", "hippocampus"): 1.0,      # Fastest (direct)
    ("septum", "cortex"): 5.0,
    ("septum", "pfc"): 8.0,
    ("septum", "striatum"): 6.0,
}


def get_axonal_delay(source: str, target: str) -> float:
    """Get axonal delay between two regions.

    Falls back to a default based on whether regions are
    "close" (same system) or "far" (different systems).
    """
    # Try exact match
    if (source, target) in AXONAL_DELAYS:
        return AXONAL_DELAYS[(source, target)]

    # Try without layer specification
    source_base = source.split("_")[0] if "_" in source else source
    target_base = target.split("_")[0] if "_" in target else target

    if (source_base, target_base) in AXONAL_DELAYS:
        return AXONAL_DELAYS[(source_base, target_base)]

    # Default based on whether same region or different
    if source_base == target_base:
        return 0.5  # Local circuit
    else:
        return 5.0  # Long-range default


class EventScheduler:
    """Priority queue scheduler for simulation events.

    Manages the global event queue and dispatches events
    to appropriate handlers. Supports:
    - Event priorities based on time
    - Batch processing of simultaneous events
    - Parallel event processing (future)
    """

    def __init__(self):
        self._queue: List[Event] = []
        self._current_time: float = 0.0
        self._event_count: int = 0

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def schedule(self, event: Event) -> None:
        """Add an event to the queue."""
        heapq.heappush(self._queue, event)
        self._event_count += 1

    def schedule_many(self, events: List[Event]) -> None:
        """Add multiple events to the queue."""
        for event in events:
            self.schedule(event)

    def pop_next(self) -> Optional[Event]:
        """Get the next event (earliest time)."""
        if self._queue:
            event = heapq.heappop(self._queue)
            self._current_time = event.time
            return event
        return None

    def pop_simultaneous(self, tolerance_ms: float = 0.01) -> List[Event]:
        """Get all events at the current time (within tolerance).

        This is useful for batch processing events that are
        effectively simultaneous, which can be parallelized.
        """
        if not self._queue:
            return []

        # Get the earliest time
        first = heapq.heappop(self._queue)
        self._current_time = first.time
        batch = [first]

        # Collect all events within tolerance
        while self._queue and self._queue[0].time <= first.time + tolerance_ms:
            batch.append(heapq.heappop(self._queue))

        return batch

    def peek_time(self) -> Optional[float]:
        """Look at the time of the next event without removing it."""
        if self._queue:
            return self._queue[0].time
        return None

    def clear(self) -> None:
        """Clear all pending events."""
        self._queue.clear()
        self._current_time = 0.0


class RegionInterface(ABC):
    """Abstract interface for brain regions in event-driven simulation.

    Each region must implement:
    - process_event(): Handle incoming events and return output events
    - get_state(): Return current state for monitoring
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this region."""
        pass

    @abstractmethod
    def process_event(self, event: Event) -> List[Event]:
        """Process an incoming event and return output events.

        This is the core computation method. It should:
        1. Apply any membrane decay since last update
        2. Process the incoming event (spikes, theta, etc.)
        3. Generate output spikes if threshold exceeded
        4. Return events to be scheduled (with appropriate delays)
        """
        pass

    @abstractmethod
    def get_connections(self) -> List[Connection]:
        """Return list of outgoing connections from this region."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return current state for monitoring/debugging."""
        pass

    def reset_state(self) -> None:
        """Reset region to initial state."""
        pass


class EventDrivenSimulation:
    """Main simulation controller for event-driven brain simulation.

    Orchestrates:
    - Event scheduling and dispatch
    - Theta rhythm generation
    - Region coordination
    - Monitoring and logging
    """

    def __init__(
        self,
        regions: Dict[str, RegionInterface],
    ):
        self.regions = regions
        self.scheduler = EventScheduler()

        # Monitoring
        self._event_history: List[Tuple[float, Event]] = []
        self._spike_counts: Dict[str, int] = {name: 0 for name in regions}

    def inject_sensory_input(
        self,
        pattern: torch.Tensor,
        target: str = "cortex",
        time: Optional[float] = None,
    ) -> None:
        """Inject sensory input as an event.

        Args:
            pattern: Input pattern (will be converted to spikes)
            target: Target region (default: cortex)
            time: When to deliver (default: now)
        """
        event_time = time if time is not None else self.scheduler.current_time
        delay = get_axonal_delay("sensory", target)

        event = Event(
            time=event_time + delay,
            event_type=EventType.SENSORY,
            source="sensory_input",
            target=target,
            payload=SpikePayload(spikes=pattern),
        )
        self.scheduler.schedule(event)

    def inject_reward(
        self,
        reward: float,
        time: Optional[float] = None,
    ) -> None:
        """Inject a reward signal (converted to dopamine).

        Args:
            reward: Reward value (positive = burst, negative = dip)
            time: When to deliver (default: now)
        """
        event_time = time if time is not None else self.scheduler.current_time

        # Dopamine goes to all regions that need it
        for target in ["striatum", "pfc", "hippocampus"]:
            if target in self.regions:
                delay = get_axonal_delay("vta", target)  # VTA → target

                event = Event(
                    time=event_time + delay,
                    event_type=EventType.DOPAMINE,
                    source="reward_system",
                    target=target,
                    payload=DopaminePayload(
                        level=reward,
                        is_burst=reward > 0.5,
                        is_dip=reward < -0.5,
                    ),
                )
                self.scheduler.schedule(event)

    def run_until(self, end_time: float) -> Dict[str, Any]:
        """Run simulation until specified time.

        Processes all events in order, advancing time as needed.

        Args:
            end_time: Stop when simulation time reaches this value

        Returns:
            Dict with simulation statistics
        """
        events_processed = 0

        while not self.scheduler.is_empty:
            # Check if next event is past end time
            next_time = self.scheduler.peek_time()
            if next_time is None or next_time > end_time:
                break

            # Get next event
            event = self.scheduler.pop_next()

            # Process the event
            if event.target in self.regions:
                region = self.regions[event.target]
                output_events = region.process_event(event)

                # Schedule output events
                self.scheduler.schedule_many(output_events)

                # Track spike counts
                if event.event_type == EventType.SPIKE:
                    payload = event.payload
                    if isinstance(payload, SpikePayload):
                        self._spike_counts[event.target] += int(payload.spikes.sum().item())

            events_processed += 1

            # Optional: store event history for debugging
            # self._event_history.append((event.time, event))

        return {
            "events_processed": events_processed,
            "final_time": self.scheduler.current_time,
            "spike_counts": self._spike_counts.copy(),
        }

    def step(self) -> Optional[Event]:
        """Process a single event.

        Returns the processed event, or None if queue is empty.
        """
        event = self.scheduler.pop_next()
        if event is None:
            return None

        if event.target in self.regions:
            region = self.regions[event.target]
            output_events = region.process_event(event)
            self.scheduler.schedule_many(output_events)

        return event

    def get_region_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of all regions."""
        return {name: region.get_state() for name, region in self.regions.items()}

    def reset_state(self) -> None:
        """Reset simulation to initial state."""
        self.scheduler.clear()
        for region in self.regions.values():
            region.reset_state()
        self._spike_counts = {name: 0 for name in self.regions}
        self._event_history.clear()
