# Distributed Computation Support (Parallel Mode)

**Status**: ✅ Implemented (December 2025)
**Priority**: Tier 3 - Major Restructuring

## Overview

Thalia now supports true parallel execution of brain regions across multiple CPU cores using Python's multiprocessing. Each region runs in its own process, communicating via event-driven message passing with realistic axonal delays.

This implementation enables biologically-plausible distributed computation while maintaining the event-driven architecture's benefits.

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     MAIN PROCESS (Orchestrator)                  │
│  • Schedules events by time                                      │
│  • Distributes events to region processes                        │
│  • Collects output events and schedules them                     │
│  • Manages theta rhythm and neuromodulators                      │
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
```

### Key Components

1. **ParallelExecutor** (`events/parallel.py`)
   - Manages worker processes for each brain region
   - Distributes events via multiprocessing queues
   - Batches simultaneous events for parallel processing
   - Handles tensor serialization (GPU → CPU → GPU)

2. **RegionWorker** (`events/parallel.py`)
   - Runs in separate process
   - Processes events through region's `process_event()` method
   - Returns output events via queue

3. **Tensor Serialization** (`events/parallel.py`)
   - `serialize_event()`: Moves GPU tensors to CPU for pickling
   - `deserialize_event()`: Moves tensors back to target device
   - Transparent to user code

4. **Region Creators** (`events/parallel.py`)
   - Module-level factory functions (pickle-compatible)
   - Creates EventDriven* adapters in worker processes
   - Configured via dictionaries (not complex objects)

## Usage

### Enabling Parallel Mode

```python
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

# Create config with parallel=True
config = ThaliaConfig(
    global_=GlobalConfig(
        device="cpu",  # REQUIRED for parallel mode
        dt_ms=1.0,
    ),
    brain=BrainConfig(
        sizes=RegionSizes(
            input_size=784,
            cortex_size=256,
            hippocampus_size=200,
            pfc_size=128,
            n_actions=10,
        ),
        parallel=True,  # Enable parallel execution
        encoding_timesteps=15,
    ),
)

# Create brain (automatically initializes parallel executor)
brain = DynamicBrain.from_thalia_config(config)

# Use normally - parallel execution is transparent
input_pattern = torch.rand(784) > 0.5
result = brain.forward(input_pattern.float(), n_timesteps=15)

# Select action
action, confidence = brain.select_action()

# Deliver reward
brain.deliver_reward(external_reward=1.0)

# Cleanup (automatically called on del, but can be explicit)
del brain
```

### Sequential vs Parallel Comparison

```python
import time

# Sequential mode
config_seq = ThaliaConfig(
    global_=GlobalConfig(device="cpu"),
    brain=BrainConfig(
        sizes=RegionSizes(...),
        parallel=False,
    ),
)
)

# Sequential execution (single process)
brain_seq = DynamicBrain.from_thalia_config(config_seq)

start = time.time()
for _ in range(100):
    brain_seq.forward(input_pattern, n_timesteps=10)
seq_time = time.time() - start

# Parallel mode
config_par = ThaliaConfig(
    global_=GlobalConfig(device="cpu"),
    brain=BrainConfig(
        sizes=RegionSizes(...),
        parallel=True,
    ),
)
)

# Parallel execution (multi-process)
brain_par = DynamicBrain.from_thalia_config(config_par)

start = time.time()
for _ in range(100):
    brain_par.forward(input_pattern, n_timesteps=10)
par_time = time.time() - start

print(f"Sequential: {seq_time:.2f}s")
print(f"Parallel:   {par_time:.2f}s")
print(f"Speedup:    {seq_time/par_time:.2f}x")
```

## Performance Characteristics

### When Parallel Mode Helps

✅ **Good candidates for parallel mode:**
- Large brain models (>1000 neurons per region)
- Long simulations (>1000 timesteps)
- CPU-bound workloads
- Multi-core machines (4+ cores)
- Sparse connectivity (less inter-region communication)

❌ **Poor candidates for parallel mode:**
- Small models (<500 neurons per region)
- Short simulations (<100 timesteps)
- GPU-accelerated training
- Machines with few cores (1-2 cores)
- Dense connectivity (high IPC overhead)

### Overhead Sources

1. **Inter-Process Communication (IPC)**
   - Event serialization (pickle)
   - Queue operations
   - ~0.1-1ms per event batch

2. **Process Startup**
   - Region initialization in each worker
   - ~0.5-2s initial overhead
   - Amortized over long simulations

3. **Tensor Serialization**
   - GPU → CPU transfers for pickling
   - Automatic, but adds latency
   - Use CPU device for best performance

### Expected Speedups

| Model Size | Cores | Typical Speedup |
|-----------|-------|-----------------|
| Small     | 4     | 1.0-1.5x        |
| Medium    | 4     | 1.5-2.5x        |
| Large     | 4     | 2.0-3.5x        |
| Large     | 8     | 3.0-6.0x        |

## Limitations

### 1. Device Constraints

**Parallel mode requires CPU device:**
```python
# ✅ Correct
config = ThaliaConfig(
    global_=GlobalConfig(device="cpu"),
    brain=BrainConfig(parallel=True),
)

# ❌ Wrong (will auto-convert to CPU with warning)
config = ThaliaConfig(
    global_=GlobalConfig(device="cuda"),
    brain=BrainConfig(parallel=True),
)
```

**Reason**: GPU tensors cannot be pickled directly. While automatic serialization is implemented, it adds overhead. For GPU acceleration, use sequential mode.

### 2. Windows Multiprocessing

Windows uses `spawn` instead of `fork`, requiring:
- All region creators must be module-level functions
- Configurations must be picklable (use dicts, not complex objects)
- Main code must be in `if __name__ == "__main__"` guard

```python
# train.py
from thalia.config import ThaliaConfig
from thalia.core.dynamic_brain import DynamicBrain

def main():
    config = ThaliaConfig(...)
    brain = DynamicBrain.from_thalia_config(config)

    # Training loop
    for epoch in range(10):
        result = brain.forward(input_pattern, n_timesteps=15)
        # ...

if __name__ == "__main__":
    main()  # REQUIRED for Windows parallel mode
```

### 3. State Synchronization

**Region states are NOT shared between processes:**
- Each region maintains local state
- State updates via event passing only
- No shared memory arrays (by design)

This is **biologically accurate** (neurons communicate via spikes, not shared memory) but means:
- Cannot directly access region state from main process during execution
- Must use events for all inter-region communication
- State inspection happens between forward passes

### 4. Debugging Challenges

**Parallel mode complicates debugging:**
- Multiple processes make stepping difficult
- Exceptions in workers may be silent
- Use sequential mode for development/debugging

```python
# Development/debugging
config = ThaliaConfig(
    brain=BrainConfig(parallel=False),  # Sequential mode
)

# Production (after debugging)
config = ThaliaConfig(
    brain=BrainConfig(parallel=True),  # Parallel mode
)
```

### 5. Non-Determinism

**Parallel execution introduces timing variability:**
- Event processing order may vary slightly
- Spike counts will differ between runs
- Use same random seed for comparable results, but expect variations

```python
# Results will vary slightly in parallel mode
torch.manual_seed(42)
result1 = brain.forward(input_pattern, n_timesteps=10)

torch.manual_seed(42)
result2 = brain.forward(input_pattern, n_timesteps=10)

# Sequential mode: result1 == result2 (deterministic)
# Parallel mode: result1 ≈ result2 (similar, not identical)
```

## Implementation Details

### Tensor Serialization

Events with GPU tensors are automatically converted for pickling:

```python
# events/parallel.py
def serialize_event(event: Event) -> Event:
    """Move GPU tensors to CPU for pickling."""
    if isinstance(event.payload, SpikePayload):
        if event.payload.spikes.is_cuda:
            event = Event(
                ...
                payload=SpikePayload(
                    spikes=event.payload.spikes.cpu(),
                    ...
                ),
            )
    return event

def deserialize_event(event: Event, device: str) -> Event:
    """Move tensors back to target device."""
    if isinstance(event.payload, SpikePayload):
        if device != "cpu":
            event = Event(
                ...
                payload=SpikePayload(
                    spikes=event.payload.spikes.to(device),
                    ...
                ),
            )
    return event
```

### Region Creator Pattern

Module-level factory functions for pickling:

```python
# events/parallel.py
def create_cortex_region(config_dict: Dict, device: str) -> RegionInterface:
    """Create cortex region (module-level for pickling)."""
    from thalia.events.adapters import EventDrivenCortex, EventRegionConfig
    from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig

    # Build config from dict
    cortex_config = LayeredCortexConfig(
        n_layers=config_dict["n_layers"],
        layer_sizes=config_dict["layer_sizes"],
        ...
    )

    # Create region
    cortex = LayeredCortex(cortex_config)

    # Wrap in event-driven adapter
    event_config = EventRegionConfig(
        name=config_dict["name"],
        output_targets=config_dict["output_targets"],
        device=device,
    )

    return EventDrivenCortex(config=event_config, cortex=cortex)

# Create factory that can be pickled
creator = lambda: create_cortex_region(config_dict, "cpu")
```

### Batch Processing

Events within tolerance are batched for parallel execution:

```python
class ParallelExecutor:
    def __init__(self, ..., batch_tolerance_ms=0.1):
        self.batch_tolerance = batch_tolerance_ms  # 0.1ms default

    def run_until(self, end_time):
        while True:
            # Get all events within 0.1ms
            batch = self.scheduler.pop_simultaneous(self.batch_tolerance)

            # Process in parallel
            output_events = self._process_batch(batch)

            # Schedule outputs
            for event in output_events:
                self.scheduler.schedule(event)
```

## Testing

Run parallel mode tests:

```bash
# All parallel tests
pytest tests/unit/test_parallel_execution.py -v

# Specific test categories
pytest tests/unit/test_parallel_execution.py::TestTensorSerialization -v
pytest tests/unit/test_parallel_execution.py::TestParallelExecution -v
pytest tests/unit/test_parallel_execution.py::TestRegionCreators -v

# With coverage
pytest tests/unit/test_parallel_execution.py --cov=thalia.events.parallel -v
```

## Best Practices

### 1. Profile Before Parallelizing

```python
import cProfile
import pstats

# Profile sequential mode first
pr = cProfile.Profile()
pr.enable()

brain = DynamicBrain.from_thalia_config(config)
for _ in range(100):
    brain.forward(input_pattern, n_timesteps=10)

pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumtime')
stats.print_stats(20)

# If region computation dominates (>80%), parallel mode may help
# If IPC/overhead dominates, parallel mode won't help
```

### 2. Use Appropriate Batch Tolerance

```python
# Default: 0.1ms (good for most cases)
executor = ParallelExecutor(region_creators, batch_tolerance_ms=0.1)

# Smaller tolerance: More precise timing, less batching
executor = ParallelExecutor(region_creators, batch_tolerance_ms=0.01)

# Larger tolerance: More batching, faster but less precise
executor = ParallelExecutor(region_creators, batch_tolerance_ms=1.0)
```

### 3. Monitor IPC Overhead

```python
result = brain.forward(input_pattern, n_timesteps=100)

# Check if IPC overhead is acceptable
overhead_ratio = result["events_processed"] / (n_timesteps * len(regions))
if overhead_ratio > 10:
    print("Warning: High IPC overhead, consider sequential mode")
```

### 4. Development Workflow

```python
# 1. Develop in sequential mode (easier debugging)
config_dev = ThaliaConfig(
    brain=BrainConfig(parallel=False),
)

# 2. Validate in parallel mode (check for equivalence)
config_test = ThaliaConfig(
    brain=BrainConfig(parallel=True),
)

# 3. Benchmark and decide
# If speedup < 1.5x, use sequential mode
# If speedup > 2x, use parallel mode for production
```

## Troubleshooting

### Issue: "BrokenPipeError" or "Queue is closed"

**Cause**: Worker process crashed or was terminated

**Solution**:
```python
# Check worker exceptions (visible in stderr)
# Add explicit cleanup
try:
    brain.forward(input_pattern, n_timesteps=10)
except BrokenPipeError:
    print("Worker crashed - check stderr for exception")
    brain._parallel_executor.stop()
```

### Issue: Slow performance (worse than sequential)

**Cause**: Model too small or high IPC overhead

**Solution**:
```python
# 1. Check model size
if sum(region.impl.n_neurons for region in brain.adapters.values()) < 1000:
    print("Model too small for parallel mode")
    # Use sequential mode instead

# 2. Profile to identify bottleneck
# 3. Increase batch tolerance to reduce IPC
```

### Issue: "pickle.PicklingError"

**Cause**: Non-picklable object in config or region

**Solution**:
```python
# Use simple config dicts, not complex objects
# Ensure all config classes are @dataclass with simple types
# Move lambdas to module-level functions
```

### Issue: Different results in parallel vs sequential

**Cause**: Expected due to timing variations

**Solution**:
```python
# Parallel mode is not bit-exact deterministic
# Use statistical comparison instead
def compare_results(result_seq, result_par, tolerance=0.5):
    for region in ["cortex", "hippocampus", "pfc"]:
        seq_count = result_seq["spike_counts"][region]
        par_count = result_par["spike_counts"][region]

        if seq_count > 0:
            ratio = par_count / seq_count
            assert tolerance < ratio < (1/tolerance), f"{region} results too different"
```

## Future Enhancements

### Planned (Not Yet Implemented)

1. **GPU Tensor Sharing** (via CUDA IPC)
   - Share GPU tensors between processes
   - Eliminate CPU serialization overhead
   - Requires CUDA IPC and careful synchronization

2. **Dynamic Load Balancing**
   - Monitor worker utilization
   - Redistribute regions to balance load
   - Adaptive batching based on network activity

3. **Distributed Mode** (Multi-Machine)
   - TCP/IP communication instead of queues
   - Scale beyond single machine
   - Requires network serialization protocol

4. **Async Parallel Execution**
   - Non-blocking event submission
   - Pipeline execution across timesteps
   - Higher throughput for streaming applications

## References

- **Implementation**: `src/thalia/events/parallel.py`
- **Integration**: `src/thalia/core/brain.py` (lines 950-1050)
- **Tests**: `tests/unit/test_parallel_execution.py`
- **ADR**: `docs/decisions/adr-XXX-parallel-execution.md` (to be created)

## See Also

- [Event-Driven Architecture](../design/event_system.md)
- [Brain Regions](../architecture/ARCHITECTURE_OVERVIEW.md#brain-regions)
- [Performance Tuning](../patterns/performance.md)
- [Testing Guide](../../CONTRIBUTING.md#testing)
