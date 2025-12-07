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


# =============================================================================
# Theta Configuration (consolidated from theta_dynamics.py)
# =============================================================================

@dataclass
class ThetaConfig:
    """Configuration for theta oscillations.
    
    Theta (6-10 Hz) is critical for:
    1. Separating encoding vs retrieval in time (phase separation)
    2. Organizing memory sequences (phase precession)
    3. Coordinating hippocampal-cortical communication
    """
    frequency_hz: float = 8.0            # Theta frequency (6-10 Hz typical)
    dt_ms: float = 1.0                   # Default timestep in ms
    
    # Phase offsets for encoding/retrieval separation
    # Encoding is maximal at trough (0°), retrieval at peak (180°)
    encoding_phase_offset: float = 0.0   # radians
    retrieval_phase_offset: float = math.pi  # radians
    
    # Modulation depth (0 = no modulation, 1 = full modulation)
    modulation_depth: float = 0.8
    
    # Minimum activity level (prevents complete silencing)
    min_activity: float = 0.1


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


class ThetaGenerator:
    """Unified theta rhythm generator and tracker.
    
    This class serves two purposes:
    1. LOCAL TRACKING: Track theta phase for sequential simulations (like ThetaState)
    2. EVENT GENERATION: Generate theta events for event-driven simulations
    
    In biology, theta (6-10 Hz) is generated by the medial septum
    and synchronizes hippocampus, cortex, and other regions.
    Different regions lock to different phases, creating
    "theta sequences" important for memory.
    
    Theta oscillations naturally separate encoding from retrieval:
    - At theta trough (phase ≈ 0): Encoding is strongest
    - At theta peak (phase ≈ π): Retrieval is strongest
    
    Example (local tracking):
        theta = ThetaGenerator(frequency_hz=8.0)
        for t in range(100):
            theta.advance()  # Advance by default dt
            enc_mod = theta.encoding_strength
            ret_mod = theta.retrieval_strength
    
    Example (event generation):
        theta = ThetaGenerator(frequency_hz=8.0, connected_regions=["cortex", "hippocampus"])
        events = theta.advance_to(10.0)  # Get events for time 10ms
    """

    def __init__(
        self,
        frequency_hz: float | ThetaConfig = 8.0,
        connected_regions: Optional[List[str]] = None,
        config: Optional[ThetaConfig] = None,
    ):
        """Initialize theta generator.
        
        Args:
            frequency_hz: Theta frequency in Hz (6-10 typical), OR
                         a ThetaConfig for backward compatibility with old ThetaState API
            connected_regions: Regions to send theta events to (for event mode)
            config: Optional ThetaConfig for advanced settings (alternative to first arg)
        
        Examples:
            # New style
            theta = ThetaGenerator(frequency_hz=8.0)
            
            # Old ThetaState style (backward compatible)
            theta = ThetaGenerator(ThetaConfig(frequency_hz=8.0))
        """
        # Handle backward compatibility: ThetaState took ThetaConfig as first arg
        if isinstance(frequency_hz, ThetaConfig):
            self.config = frequency_hz
        elif config is not None:
            self.config = config
        else:
            self.config = ThetaConfig(frequency_hz=frequency_hz)
        
        self.frequency = self.config.frequency_hz
        self.period_ms = 1000.0 / self.frequency  # ~125ms for 8Hz
        
        # Phase state
        self._phase = 0.0
        self.time = 0.0
        
        # Precompute phase increment for advance()
        self._phase_increment = (
            2.0 * math.pi * self.frequency * self.config.dt_ms / 1000.0
        )

        # Connected regions for event generation
        self.connected_regions = connected_regions or [
            "hippocampus", "cortex", "pfc", "striatum"
        ]
    
    # =========================================================================
    # Phase Properties
    # =========================================================================
    
    @property
    def phase(self) -> float:
        """Current theta phase in radians [0, 2π)."""
        return self._phase
    
    @phase.setter
    def phase(self, value: float) -> None:
        """Set theta phase, wrapping to [0, 2π)."""
        self._phase = value % (2.0 * math.pi)
    
    # =========================================================================
    # Modulation Strengths
    # =========================================================================
    
    @property
    def encoding_strength(self) -> float:
        """Encoding strength based on theta phase.
        
        Maximal at theta trough (phase = 0, 2π), minimal at peak (phase = π).
        Returns value in [min_activity, 1.0].
        """
        cfg = self.config
        phase_with_offset = self._phase + cfg.encoding_phase_offset
        modulation = 0.5 * (1.0 + math.cos(phase_with_offset))  # [0, 1]
        
        # Apply modulation depth and minimum
        strength = cfg.min_activity + (1.0 - cfg.min_activity) * modulation * cfg.modulation_depth
        return strength
    
    @property
    def retrieval_strength(self) -> float:
        """Retrieval strength based on theta phase.
        
        Maximal at theta peak (phase = π), minimal at trough (phase = 0).
        Returns value in [min_activity, 1.0].
        """
        cfg = self.config
        phase_with_offset = self._phase + cfg.retrieval_phase_offset
        modulation = 0.5 * (1.0 + math.cos(phase_with_offset))  # [0, 1]
        
        # Apply modulation depth and minimum
        strength = cfg.min_activity + (1.0 - cfg.min_activity) * modulation * cfg.modulation_depth
        return strength
    
    # =========================================================================
    # Local Tracking Methods (for sequential simulations)
    # =========================================================================
    
    def advance(self, dt_ms: Optional[float] = None) -> None:
        """Advance theta phase by one timestep.
        
        Args:
            dt_ms: Override timestep (uses config.dt_ms if None)
        """
        if dt_ms is not None:
            increment = 2.0 * math.pi * self.frequency * dt_ms / 1000.0
        else:
            increment = self._phase_increment
        
        self._phase = (self._phase + increment) % (2.0 * math.pi)
        self.time += dt_ms if dt_ms is not None else self.config.dt_ms
    
    def reset_state(self) -> None:
        """Reset theta phase to zero (align to optimal encoding).
        
        Called at the start of a new trial to synchronize theta
        with task timing.
        """
        self._phase = 0.0
    
    def align_to_encoding(self) -> None:
        """Align theta phase to optimal encoding (trough)."""
        self._phase = 0.0
    
    def align_to_retrieval(self) -> None:
        """Align theta phase to optimal retrieval (peak)."""
        self._phase = math.pi
    
    # =========================================================================
    # Event Generation Methods (for event-driven simulations)
    # =========================================================================

    def advance_to(self, new_time: float) -> List[Event]:
        """Advance theta to new time and generate events.

        Returns theta events for each connected region,
        scheduled with appropriate axonal delays.
        """
        self.time = new_time
        # Phase in radians (0 to 2π)
        self._phase = (2.0 * math.pi * self.frequency * self.time / 1000.0) % (2.0 * math.pi)

        # Generate events for all connected regions
        events: List[Event] = []
        for region in self.connected_regions:
            delay = get_axonal_delay("septum", region)

            event = Event(
                time=self.time + delay,
                event_type=EventType.THETA,
                source="theta_generator",
                target=region,
                payload=ThetaPayload(
                    phase=self._phase,
                    frequency=self.frequency,
                    encoding_strength=self.encoding_strength,
                    retrieval_strength=self.retrieval_strength,
                ),
            )
            events.append(event)

        return events

    def get_phase_at(self, time: float) -> float:
        """Get theta phase at a specific time."""
        return (2.0 * math.pi * self.frequency * time / 1000.0) % (2.0 * math.pi)
    
    # =========================================================================
    # Tensor Methods (for batch operations)
    # =========================================================================
    
    def get_phase_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Get theta phase as a tensor for batch operations."""
        return torch.tensor([self._phase], device=device)
    
    def get_modulation_tensors(
        self, 
        device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get encoding and retrieval modulation as tensors."""
        return (
            torch.tensor([self.encoding_strength], device=device),
            torch.tensor([self.retrieval_strength], device=device),
        )


# Alias for backward compatibility with theta_dynamics.py
ThetaState = ThetaGenerator


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
        theta_frequency: float = 8.0,
    ):
        self.regions = regions
        self.scheduler = EventScheduler()
        self.theta = ThetaGenerator(
            frequency_hz=theta_frequency,
            connected_regions=list(regions.keys()),
        )

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

        # Schedule initial theta events
        theta_events = self.theta.advance_to(0.0)
        self.scheduler.schedule_many(theta_events)

        # Schedule periodic theta updates
        theta_interval = 1.0  # Update theta every 1ms
        next_theta_time = theta_interval

        while not self.scheduler.is_empty:
            # Check if next event is past end time
            next_time = self.scheduler.peek_time()
            if next_time is None or next_time > end_time:
                break

            # Get next event
            event = self.scheduler.pop_next()

            # Schedule theta updates as needed
            while next_theta_time <= event.time:
                theta_events = self.theta.advance_to(next_theta_time)
                self.scheduler.schedule_many(theta_events)
                next_theta_time += theta_interval

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
        self.theta = ThetaGenerator(
            frequency_hz=self.theta.frequency,
            connected_regions=list(self.regions.keys()),
        )
        for region in self.regions.values():
            region.reset_state()
        self._spike_counts = {name: 0 for name in self.regions}
        self._event_history.clear()
