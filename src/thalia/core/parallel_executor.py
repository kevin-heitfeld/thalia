"""
Parallel Execution Framework for Event-Driven Brain Simulation.

This module enables true parallel execution of brain regions across
multiple CPU cores using Python's multiprocessing.

Architecture:
=============

    ┌─────────────────────────────────────────────────────────────────┐
    │                     MAIN PROCESS (Orchestrator)                  │
    │  • Schedules events by time                                      │
    │  • Distributes events to region processes                        │
    │  • Collects output events and schedules them                     │
    │  • Generates theta rhythm                                        │
    └─────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │    REGION 0   │       │    REGION 1   │       │    REGION 2   │
    │   (Process)   │       │   (Process)   │       │   (Process)   │
    │   - Cortex    │       │ - Hippocampus │       │     - PFC     │
    └───────────────┘       └───────────────┘       └───────────────┘
            │                       │                       │
            └───────────────────────┴───────────────────────┘
                        Output events returned to main

Key Design Decisions:
=====================

1. MAIN PROCESS ORCHESTRATION:
   - Main process handles event scheduling (priority queue)
   - Theta generation stays in main process for simplicity
   - Output events collected and re-scheduled by main

2. BATCH PROCESSING:
   - Events within tolerance (e.g., 0.1ms) are batched together
   - Each batch distributed to all regions in parallel
   - Synchronization barrier before next batch

3. REGION PROCESSES:
   - Each region runs in its own process
   - Receives events via input queue
   - Sends output events via output queue
   - Processes events sequentially within each batch

4. PICKLE-FRIENDLY:
   - Region states must be serializable
   - Use shared memory for large tensors if needed

Benefits:
=========
- True parallel execution on multi-core CPUs
- Scales with number of brain regions
- Event-driven (no wasted computation)
- Realistic asynchronous dynamics

Limitations:
============
- Inter-process communication overhead
- Must serialize events (pickle)
- GPU tensors need special handling

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process, Queue, Event as MPEvent
from queue import Empty
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import torch

from .event_system import (
    Event, EventType, EventScheduler,
    SpikePayload, RegionInterface, get_axonal_delay,
)


@dataclass
class RegionWorkerConfig:
    """Configuration for a region worker process."""
    name: str
    region_class: type
    region_config: Any
    device: str = "cpu"


class RegionWorker:
    """Worker process for a single brain region.

    Runs in a separate process and:
    - Receives events from input queue
    - Processes events through the region
    - Sends output events to output queue
    """

    def __init__(
        self,
        name: str,
        input_queue: Queue,
        output_queue: Queue,
        control_event: MPEvent,
        region: RegionInterface,
    ):
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.control_event = control_event
        self.region = region
        self._running = True

    def run(self) -> None:
        """Main event loop for the worker."""
        while self._running:
            # Wait for control event (signals batch ready)
            self.control_event.wait()
            self.control_event.clear()

            # Process all events in input queue
            events_processed = 0
            output_events = []

            while True:
                try:
                    event = self.input_queue.get_nowait()

                    # Check for shutdown signal
                    if event is None:
                        self._running = False
                        break

                    # Process the event
                    outputs = self.region.process_event(event)
                    output_events.extend(outputs)
                    events_processed += 1

                except Empty:
                    break

            # Send output events back to main process
            for event in output_events:
                self.output_queue.put(event)

            # Signal that we're done with this batch
            self.output_queue.put(("DONE", self.name, events_processed))


def worker_process(
    name: str,
    input_queue: Queue,
    output_queue: Queue,
    control_event: MPEvent,
    region_creator: callable,
) -> None:
    """Entry point for worker process.

    Creates the region inside the process to avoid pickle issues.
    """
    try:
        # Create region inside the worker process
        region = region_creator()

        worker = RegionWorker(
            name=name,
            input_queue=input_queue,
            output_queue=output_queue,
            control_event=control_event,
            region=region,
        )
        worker.run()

    except Exception as e:
        import sys
        import traceback
        print(f"[{name}] ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
class _ParallelExecutor:
    """Internal: Event-driven simulation with parallel region execution.

    This is an internal implementation detail of EventDrivenBrain.
    Use EventDrivenBrain with parallel=True instead of using this directly.

    Distributes events to region processes running in parallel,
    collects output events, and schedules them appropriately.
    """

    def __init__(
        self,
        region_creators: Dict[str, Callable[[], RegionInterface]],
        batch_tolerance_ms: float = 0.1,
    ):
        """Initialize parallel simulation.

        Args:
            region_creators: Dict mapping region names to callables that
                            create RegionInterface instances. Using callables
                            avoids pickling the actual region objects.
            batch_tolerance_ms: Events within this tolerance are batched
        """
        self.region_names = list(region_creators.keys())
        self.batch_tolerance = batch_tolerance_ms

        # Event scheduling (in main process)
        self.scheduler = EventScheduler()

        # Per-region queues and control events
        self.input_queues: Dict[str, Queue] = {}
        self.output_queues: Dict[str, Queue] = {}
        self.control_events: Dict[str, MPEvent] = {}
        self.processes: Dict[str, Process] = {}

        # Create worker processes
        for name, creator in region_creators.items():
            self.input_queues[name] = Queue()
            self.output_queues[name] = Queue()
            self.control_events[name] = MPEvent()

            process = Process(
                target=worker_process,
                args=(
                    name,
                    self.input_queues[name],
                    self.output_queues[name],
                    self.control_events[name],
                    creator,
                ),
                daemon=True,
            )
            self.processes[name] = process

        # Monitoring
        self._spike_counts: Dict[str, int] = {name: 0 for name in self.region_names}
        self._events_processed = 0

    def start(self) -> None:
        """Start all worker processes."""
        for process in self.processes.values():
            process.start()

    def stop(self) -> None:
        """Stop all worker processes."""
        # Send shutdown signal
        for name in self.region_names:
            self.input_queues[name].put(None)
            self.control_events[name].set()

        # Wait for processes to finish
        for process in self.processes.values():
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()

    def inject_sensory_input(
        self,
        pattern: torch.Tensor,
        target: str = "cortex",
        time: Optional[float] = None,
    ) -> None:
        """Inject sensory input as an event."""
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
        """Inject reward signal (converted to dopamine)."""
        from .event_system import DopaminePayload

        event_time = time if time is not None else self.scheduler.current_time

        for target in ["striatum", "pfc", "hippocampus"]:
            if target in self.region_names:
                delay = get_axonal_delay("vta", target)
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

    def _process_batch(self, events: List[Event]) -> List[Event]:
        """Process a batch of events in parallel.

        Distributes events to appropriate region processes,
        waits for all to complete, collects output events.
        """
        if not events:
            return []

        # Distribute events to region queues
        events_by_region: Dict[str, List[Event]] = {name: [] for name in self.region_names}
        for event in events:
            if event.target in events_by_region:
                events_by_region[event.target].append(event)

        # Put events in queues
        regions_with_work = []
        for name, region_events in events_by_region.items():
            for event in region_events:
                self.input_queues[name].put(event)
            if region_events:
                regions_with_work.append(name)

        # Signal all workers to process
        for name in regions_with_work:
            self.control_events[name].set()

        # Collect output events from all workers
        output_events = []
        done_count = 0

        while done_count < len(regions_with_work):
            for name in regions_with_work:
                try:
                    result = self.output_queues[name].get(timeout=0.01)

                    if isinstance(result, tuple) and result[0] == "DONE":
                        done_count += 1
                    else:
                        output_events.append(result)

                except Empty:
                    continue

        # Drain remaining output events
        for name in regions_with_work:
            while True:
                try:
                    result = self.output_queues[name].get_nowait()
                    if isinstance(result, Event):
                        output_events.append(result)
                except Empty:
                    break

        return output_events

    def run_until(self, end_time: float) -> Dict[str, Any]:
        """Run simulation until specified time.

        Processes events in batches, distributing to parallel workers.
        """
        while True:
            # Check if we're done
            next_time = self.scheduler.peek_time()
            if next_time is None or next_time > end_time:
                break

            # Get batch of simultaneous events
            batch = self.scheduler.pop_simultaneous(self.batch_tolerance)

            if not batch:
                break

            # Process batch in parallel
            output_events = self._process_batch(batch)

            # Schedule output events
            for event in output_events:
                self.scheduler.schedule(event)

                # Track spike counts
                if event.event_type == EventType.SPIKE:
                    if isinstance(event.payload, SpikePayload):
                        # Count spikes going TO this target
                        if event.target in self._spike_counts:
                            self._spike_counts[event.target] += int(event.payload.spikes.sum().item())

            self._events_processed += len(batch)

        return {
            "events_processed": self._events_processed,
            "final_time": self.scheduler.current_time,
            "spike_counts": self._spike_counts.copy(),
        }
