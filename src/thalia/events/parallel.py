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

import atexit
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from multiprocessing.synchronize import Event as MPEvent
from queue import Empty
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable

# Ensure we use spawn method for consistency across platforms
# This is required for Windows and makes the code more predictable
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=False)

from thalia.events.system import (
    Event, EventType, EventScheduler,
    SpikePayload, RegionInterface, get_axonal_delay,
)


# =============================================================================
# Tensor Serialization for Multiprocessing
# =============================================================================

def serialize_event(event: Event) -> Event:
    """Serialize event for multiprocessing (move GPU tensors to CPU).

    Args:
        event: Event with potential GPU tensors

    Returns:
        Event with CPU tensors that can be pickled
    """
    if isinstance(event.payload, SpikePayload):
        # Move spikes to CPU for pickling
        spikes = event.payload.spikes
        if spikes.is_cuda:
            event = Event(
                time=event.time,
                event_type=event.event_type,
                source=event.source,
                target=event.target,
                payload=SpikePayload(
                    spikes=spikes.cpu(),
                    source_layer=event.payload.source_layer,
                ),
            )
    return event


def deserialize_event(event: Event, device: str) -> Event:
    """Deserialize event and move tensors to target device.

    Args:
        event: Event with CPU tensors
        device: Target device ("cpu" or "cuda")

    Returns:
        Event with tensors on target device
    """
    if isinstance(event.payload, SpikePayload) and device != "cpu":
        event = Event(
            time=event.time,
            event_type=event.event_type,
            source=event.source,
            target=event.target,
            payload=SpikePayload(
                spikes=event.payload.spikes.to(device),
                source_layer=event.payload.source_layer,
            ),
        )
    return event


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
            # Wait for control event with timeout (signals batch ready or allows checking shutdown)
            triggered = self.control_event.wait(timeout=0.1)
            if not triggered:
                # Timeout - check for shutdown signal
                try:
                    event = self.input_queue.get_nowait()
                    if event is None:
                        self._running = False
                        break
                    else:
                        # Put it back, we'll process it when control event is set
                        self.input_queue.put(event)
                except Empty:
                    pass
                continue

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

            # Signal that we're done with this batch (only if still running)
            if self._running:
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
    import logging
    import traceback

    # Set up logging for this worker
    logging.basicConfig(
        level=logging.WARNING,
        format=f'%(asctime)s - {name} - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(name)

    try:
        logger.info(f"Worker {name} starting...")

        # Create region inside the worker process
        region = region_creator()
        logger.info(f"Worker {name} created region: {type(region).__name__}")

        worker = RegionWorker(
            name=name,
            input_queue=input_queue,
            output_queue=output_queue,
            control_event=control_event,
            region=region,
        )
        logger.info(f"Worker {name} entering event loop...")
        worker.run()
        logger.info(f"Worker {name} exiting normally")

    except Exception as e:
        logger.error(f"Worker {name} crashed: {e}")
        logger.error(traceback.format_exc())
        # Put error in output queue so main process knows
        try:
            output_queue.put(("ERROR", name, str(e)))
        except:
            pass
        raise


class ParallelExecutor:
    """Event-driven simulation with parallel region execution.

    Used by DynamicBrain when parallel=True to distribute event
    processing across multiple CPU cores.

    Distributes events to region processes running in parallel,
    collects output events, and schedules them appropriately.
    """

    def __init__(
        self,
        region_creators: Dict[str, Callable[[], RegionInterface]],
        batch_tolerance_ms: float = 0.1,
        device: str = "cpu",
    ):
        """Initialize parallel simulation.

        Args:
            region_creators: Dict mapping region names to callables that
                            create RegionInterface instances. Using callables
                            avoids pickling the actual region objects.
            batch_tolerance_ms: Events within this tolerance are batched
            device: Device for regions ("cpu" recommended for parallel mode)
        """
        self.region_names = list(region_creators.keys())
        self.batch_tolerance = batch_tolerance_ms
        self.device = device

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
            self.control_events[name] = mp.Event()  # Use factory function

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

        # Register cleanup handler to ensure workers are stopped
        atexit.register(self.stop)

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
        # Serialize before scheduling (in case pattern is on GPU)
        event = serialize_event(event)
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

        # Put events in queues (serialize for pickling)
        regions_with_work = []
        for name, region_events in events_by_region.items():
            for event in region_events:
                # Serialize event (move GPU tensors to CPU for pickling)
                serialized_event = serialize_event(event)
                self.input_queues[name].put(serialized_event)
            if region_events:
                regions_with_work.append(name)

        # Signal all workers to process
        for name in regions_with_work:
            self.control_events[name].set()

        # Collect output events from all workers
        output_events = []
        done_regions = set()

        # Wait for each region to signal done (with timeout)
        timeout_per_region = 5.0  # 5 seconds per region

        while len(done_regions) < len(regions_with_work):
            for name in regions_with_work:
                if name in done_regions:
                    continue

                try:
                    result = self.output_queues[name].get(timeout=timeout_per_region)

                    if isinstance(result, tuple) and result[0] == "DONE":
                        done_regions.add(name)
                    elif isinstance(result, tuple) and result[0] == "ERROR":
                        # Worker crashed
                        error_msg = result[2] if len(result) > 2 else "Unknown error"
                        raise RuntimeError(f"Worker process {name} crashed: {error_msg}")
                    else:
                        # Deserialize event (move tensors to target device)
                        deserialized_event = deserialize_event(result, self.device)
                        output_events.append(deserialized_event)

                except Empty:
                    # Timeout - check if worker is still alive
                    worker = self.processes.get(name)
                    if worker and not worker.is_alive():
                        raise RuntimeError(f"Worker process {name} died unexpectedly")
                    # Otherwise, continue waiting
                    continue

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


# =============================================================================
# Module-Level Region Creator Functions (Pickle-able)
# =============================================================================
# These functions MUST be defined at module level (not inside a class or
# function) so they can be pickled by multiprocessing on Windows (spawn mode).
#
# Each creator function:
# 1. Imports the necessary region class
# 2. Creates an EventDriven* adapter with the region
# 3. Returns a RegionInterface that can process events
#
# NOTE: These functions take raw config dicts instead of config objects
# to avoid pickling issues with complex config classes.
# =============================================================================

def create_thalamus_region(
    config_dict: Dict[str, Any],
    device: str = "cpu",
) -> RegionInterface:
    """Create event-driven thalamus region (module-level for pickling).

    Args:
        config_dict: Dictionary with thalamus configuration:
            - name: Region name
            - n_input: Input size
            - n_output: Output size
            - output_targets: List of target region names
        device: Device to use

    Returns:
        EventDrivenThalamus instance
    """
    from thalia.events.adapters import EventDrivenThalamus, EventRegionConfig
    from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig

    # Build thalamus config
    thalamus_config = ThalamicRelayConfig(
        n_input=config_dict.get("n_input", 784),
        n_output=config_dict.get("n_output", 256),
        device=device,
        dt_ms=1.0,
    )

    # Create thalamus
    thalamus = ThalamicRelay(thalamus_config)

    # Build event config
    event_config = EventRegionConfig(
        name=config_dict.get("name", "thalamus"),
        output_targets=config_dict.get("output_targets", ["cortex"]),
        device=device,
    )

    # Create and return adapter
    return EventDrivenThalamus(config=event_config, thalamus=thalamus)


def create_cortex_region(
    config_dict: Dict[str, Any],
    device: str = "cpu",
) -> RegionInterface:
    """Create event-driven cortex region (module-level for pickling).

    Args:
        config_dict: Dictionary with cortex configuration:
            - name: Region name
            - n_layers: Number of cortical layers
            - layer_sizes: List of layer sizes
            - output_targets: List of target region names
        device: Device to use ("cpu" or "cuda")

    Returns:
        EventDrivenCortex instance
    """
    from thalia.events.adapters import EventDrivenCortex, EventRegionConfig
    from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig

    # Build cortex config (LayeredCortexConfig uses n_input/n_output)
    n_output = config_dict.get("n_output", 256)
    # Calculate layer sizes to satisfy n_output = l23_size + l5_size
    # Use 1.67:1 ratio (l23:l5) for clean split
    l5_size = int(n_output * 0.375)  # 37.5% → 96 for n_output=256
    l23_size = n_output - l5_size     # Remaining → 160 for n_output=256
    l4_size = l23_size                # Same as l23
    l6_size = int(n_output * 0.3125)  # 31.25% → 80 for n_output=256

    cortex_config = LayeredCortexConfig(
        n_input=config_dict.get("n_input", 784),
        n_output=n_output,
        l4_size=l4_size,
        l23_size=l23_size,
        l5_size=l5_size,
        l6_size=l6_size,
        device=device,
        dt_ms=1.0,
    )    # Create cortex
    cortex = LayeredCortex(cortex_config)

    # Build event config
    event_config = EventRegionConfig(
        name=config_dict.get("name", "cortex"),
        output_targets=config_dict.get("output_targets", ["hippocampus", "pfc", "striatum"]),
        device=device,
    )

    # Create and return adapter
    return EventDrivenCortex(config=event_config, cortex=cortex)


def create_hippocampus_region(
    config_dict: Dict[str, Any],
    device: str = "cpu",
) -> RegionInterface:
    """Create event-driven hippocampus region (module-level for pickling).

    Args:
        config_dict: Dictionary with hippocampus configuration:
            - name: Region name
            - dg_size: Dentate gyrus size
            - ca3_size: CA3 size
            - ca1_size: CA1 size
            - output_targets: List of target region names
        device: Device to use

    Returns:
        EventDrivenHippocampus instance
    """
    from thalia.events.adapters import EventDrivenHippocampus, EventRegionConfig
    from thalia.regions.hippocampus import Hippocampus, HippocampusConfig

    # Build hippocampus config
    hc_config = HippocampusConfig(
        n_input=config_dict.get("n_input", 256),
        n_output=config_dict.get("ca1_size", 200),
        ec_l3_input_size=config_dict.get("n_input", 256),
        device=device,
        dt_ms=1.0,
    )

    # Create hippocampus
    hippocampus = Hippocampus(hc_config)    # Build event config
    event_config = EventRegionConfig(
        name=config_dict.get("name", "hippocampus"),
        output_targets=config_dict.get("output_targets", ["pfc", "cortex"]),
        device=device,
    )

    # Create and return adapter
    return EventDrivenHippocampus(config=event_config, hippocampus=hippocampus)


def create_pfc_region(
    config_dict: Dict[str, Any],
    device: str = "cpu",
) -> RegionInterface:
    """Create event-driven PFC region (module-level for pickling).

    Args:
        config_dict: Dictionary with PFC configuration:
            - name: Region name
            - n_neurons: Number of PFC neurons
            - n_input: Input size
            - output_targets: List of target region names
        device: Device to use

    Returns:
        EventDrivenPFC instance
    """
    from thalia.events.adapters import EventDrivenPFC, EventRegionConfig
    from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig

    # Build PFC config
    pfc_config = PrefrontalConfig(
        n_output=config_dict.get("n_neurons", 128),
        n_input=config_dict.get("n_input", 256),
        device=device,
        dt_ms=1.0,
    )

    # Create PFC
    pfc = Prefrontal(pfc_config)    # Build event config
    event_config = EventRegionConfig(
        name=config_dict.get("name", "pfc"),
        output_targets=config_dict.get("output_targets", ["cortex", "striatum", "hippocampus"]),
        device=device,
    )

    # Create and return adapter
    return EventDrivenPFC(config=event_config, pfc=pfc)


def create_striatum_region(
    config_dict: Dict[str, Any],
    device: str = "cpu",
) -> RegionInterface:
    """Create event-driven striatum region (module-level for pickling).

    Args:
        config_dict: Dictionary with striatum configuration:
            - name: Region name
            - n_actions: Number of actions
            - n_input: Input size
            - output_targets: List of target region names
        device: Device to use

    Returns:
        EventDrivenStriatum instance
    """
    from thalia.events.adapters import EventDrivenStriatum, EventRegionConfig
    from thalia.regions.striatum import Striatum, StriatumConfig

    # Build striatum config (n_output is number of actions)
    striatum_config = StriatumConfig(
        n_output=config_dict.get("n_actions", 10),
        n_input=config_dict.get("n_input", 256),
        neurons_per_action=config_dict.get("neurons_per_action", 20),
        device=device,
        dt_ms=1.0,
    )    # Create striatum
    striatum = Striatum(striatum_config)

    # Build event config
    event_config = EventRegionConfig(
        name=config_dict.get("name", "striatum"),
        output_targets=config_dict.get("output_targets", []),
        device=device,
    )

    # Create and return adapter
    return EventDrivenStriatum(config=event_config, striatum=striatum)


def create_cerebellum_region(
    config_dict: Dict[str, Any],
    device: str = "cpu",
) -> RegionInterface:
    """Create event-driven cerebellum region (module-level for pickling).

    Args:
        config_dict: Dictionary with cerebellum configuration:
            - name: Region name
            - n_purkinje: Number of Purkinje cells
            - n_input: Input size
            - output_targets: List of target region names
        device: Device to use

    Returns:
        EventDrivenCerebellum instance
    """
    from thalia.events.adapters import EventDrivenCerebellum, EventRegionConfig
    from thalia.regions.cerebellum import Cerebellum, CerebellumConfig

    # Build cerebellum config
    cerebellum_config = CerebellumConfig(
        n_output=config_dict.get("n_purkinje", 100),
        n_input=config_dict.get("n_input", 256),
        device=device,
        dt_ms=1.0,
    )    # Create cerebellum
    cerebellum = Cerebellum(cerebellum_config)

    # Build event config
    event_config = EventRegionConfig(
        name=config_dict.get("name", "cerebellum"),
        output_targets=config_dict.get("output_targets", []),
        device=device,
    )

    # Create and return adapter
    return EventDrivenCerebellum(config=event_config, cerebellum=cerebellum)


def create_region_creator(
    region_type: str,
    config_dict: Dict[str, Any],
    device: str = "cpu",
) -> Callable[[], RegionInterface]:
    """Create a region creator function for the specified region type.

    This returns a reference to the appropriate module-level creator function,
    with config bound via functools.partial (which IS picklable).

    Args:
        region_type: Type of region ("cortex", "hippocampus", etc.)
        config_dict: Configuration dictionary for the region
        device: Device to use

    Returns:
        Callable that creates the region when called

    Example:
        >>> creator = create_region_creator("cortex", {"n_layers": 3}, "cpu")
        >>> region = creator()  # Creates the cortex
    """
    from functools import partial

    creators = {
        "thalamus": create_thalamus_region,
        "cortex": create_cortex_region,
        "hippocampus": create_hippocampus_region,
        "pfc": create_pfc_region,
        "striatum": create_striatum_region,
        "cerebellum": create_cerebellum_region,
    }

    if region_type not in creators:
        raise ValueError(f"Unknown region type: {region_type}. Valid types: {list(creators.keys())}")

    # Return a partial application (which IS picklable)
    return partial(creators[region_type], config_dict, device)
