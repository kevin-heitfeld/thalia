# ADR-014: Distributed Computation via Multiprocessing

**Status**: Accepted
**Date**: 2025-12-12
**Priority**: Tier 3 - Major Restructuring

## Context

Thalia's event-driven architecture enables natural parallelism - each brain region operates independently and communicates via spike events with axonal delays. However, the initial implementation processed all regions sequentially in a single process, leaving multi-core CPUs underutilized.

To scale to larger brain models and improve training throughput, we need true parallel execution where each region runs in its own process, communicating via message passing.

### Requirements

1. **True Parallelism**: Each region runs independently on separate CPU cores
2. **Biological Accuracy**: Maintain event-driven architecture with realistic delays
3. **Transparency**: Same API for sequential and parallel modes
4. **Scalability**: Support 2-16 processes without significant overhead
5. **Portability**: Work on Windows, Linux, and macOS

### Constraints

- Python's Global Interpreter Lock (GIL) prevents multi-threading
- Multiprocessing requires serializable (picklable) objects
- GPU tensors cannot be directly pickled
- Windows uses `spawn` (not `fork`) for multiprocessing
- Inter-process communication (IPC) adds overhead

## Decision

We implement **process-based parallelism** using Python's `multiprocessing` module with the following architecture:

### 1. Main Process Orchestration

The main process acts as orchestrator:
- Maintains global event scheduler (priority queue)
- Distributes events to worker processes via queues
- Collects output events and re-schedules them
- Manages theta rhythm and neuromodulator broadcasting

**Rationale**: Centralized scheduling ensures correct event ordering and simplifies synchronization.

### 2. Worker Process Per Region

Each brain region runs in a separate worker process:
- Receives input events via multiprocessing.Queue
- Processes events through `RegionInterface.process_event()`
- Sends output events back via output queue
- Maintains local state (no shared memory)

**Rationale**: Process isolation matches biological reality (neurons don't share memory) and enables true parallelism.

### 3. Batch Processing

Events within a time tolerance (default 0.1ms) are batched:
- Main process collects simultaneous events
- Distributes batch to all relevant workers in parallel
- Waits for all workers to complete before next batch

**Rationale**: Batching reduces IPC overhead while maintaining temporal precision. 0.1ms tolerance is below biological spike jitter (~1ms).

### 4. Tensor Serialization

GPU tensors are automatically converted for pickling:
- `serialize_event()`: GPU → CPU before sending to worker
- `deserialize_event()`: CPU → target device in worker
- Transparent to user code

**Rationale**: Pickle cannot serialize CUDA tensors. Automatic conversion maintains transparency while enabling GPU support (future: shared memory optimization).

### 5. Module-Level Region Creators

Region initialization uses module-level factory functions:
```python
def create_cortex_region(config_dict: Dict, device: str) -> RegionInterface:
    # Create region from simple dict (picklable)
    # Return EventDrivenRegion adapter
    ...
```

**Rationale**: Windows `spawn` requires picklable functions. Module-level factories work across all platforms.

### 6. CPU-Only Default

Parallel mode defaults to CPU device:
- GPU tensors cause serialization overhead
- Multi-GPU not yet implemented
- CPU parallelism more predictable

**Rationale**: Optimize for common case (CPU parallelism). GPU support is a future enhancement.

## Implementation

### Core Components

1. **ParallelExecutor** (`events/parallel.py`)
   - Manages worker processes and queues
   - Implements batch processing logic
   - Handles tensor serialization

2. **RegionWorker** (`events/parallel.py`)
   - Worker process event loop
   - Processes events sequentially within each batch
   - Signals completion to main process

3. **Region Creators** (`events/parallel.py`)
   - `create_cortex_region()`
   - `create_hippocampus_region()`
   - `create_pfc_region()`
   - `create_striatum_region()`
   - `create_cerebellum_region()`

4. **Brain Integration** (`core/brain.py`)
   - `_init_parallel_executor()`: Setup worker processes
   - `forward()`: Delegates to ParallelExecutor or sequential

### Configuration

```python
from thalia.config import ThaliaConfig, BrainConfig

# Enable parallel mode
config = ThaliaConfig(
    global_=GlobalConfig(device="cpu"),  # CPU required
    brain=BrainConfig(
        parallel=True,  # Enable multiprocessing
        ...
    ),
)

brain = DynamicBrain.from_thalia_config(config)
```

## Consequences

### Positive

1. **True Parallelism**: Scales linearly with number of cores (up to ~4-6x speedup)
2. **Biological Accuracy**: Maintains event-driven architecture and axonal delays
3. **Transparent API**: Same `forward()` interface for sequential and parallel
4. **Scalable**: Supports 2-16 processes with acceptable overhead
5. **Debuggable**: Can switch to sequential mode for debugging

### Negative

1. **IPC Overhead**: ~0.1-1ms per event batch (negligible for large models)
2. **CPU-Only**: GPU acceleration requires sequential mode (for now)
3. **Startup Cost**: ~0.5-2s to spawn worker processes (amortized over training)
4. **Non-Determinism**: Slight timing variations between runs (biologically realistic!)
5. **Windows Requirement**: Must use `if __name__ == "__main__"` guard

### Neutral

1. **Memory Usage**: N×(region memory) for N processes (expected)
2. **State Isolation**: Cannot directly access region state during execution (by design)
3. **Complexity**: More complex than sequential, but abstracted behind API

## Performance Characteristics

### When to Use Parallel Mode

✅ **Good candidates:**
- Large models (>1000 neurons per region)
- Long simulations (>1000 timesteps)
- Multi-core machines (4+ cores)
- CPU-bound workloads

❌ **Poor candidates:**
- Small models (<500 neurons per region)
- Short simulations (<100 timesteps)
- GPU-accelerated training
- Dense inter-region connectivity

### Expected Speedups

| Model Size | Cores | Typical Speedup |
|-----------|-------|-----------------|
| Small     | 4     | 1.0-1.5x        |
| Medium    | 4     | 1.5-2.5x        |
| Large     | 4     | 2.0-3.5x        |
| Large     | 8     | 3.0-6.0x        |

### Overhead Analysis

1. **IPC Overhead**: 5-10% for typical workloads
2. **Serialization**: <1% (CPU tensors), 5-15% (GPU tensors)
3. **Startup**: 0.5-2s (one-time cost)
4. **Synchronization**: <1% (batching minimizes)

## Alternatives Considered

### 1. Threading (Rejected)

**Pros**: Lower overhead, shared memory
**Cons**: GIL prevents true parallelism, not biologically accurate

**Verdict**: Rejected - GIL defeats the purpose

### 2. Ray/Distributed Framework (Deferred)

**Pros**: Better load balancing, multi-machine support
**Cons**: Heavy dependency, harder to debug, overkill for single machine

**Verdict**: Consider for future multi-machine scaling (ADR-XXX)

### 3. MPI (Message Passing Interface) (Rejected)

**Pros**: High performance, proven for HPC
**Cons**: Complex API, requires MPI installation, poor Windows support

**Verdict**: Rejected - multiprocessing sufficient for single-machine

### 4. Shared Memory + Locks (Rejected)

**Pros**: Faster than message passing
**Cons**: Complex synchronization, not biologically accurate, hard to debug

**Verdict**: Rejected - violates biological plausibility (neurons use spikes, not shared memory)

### 5. Async/Await (Rejected)

**Pros**: Lightweight, Pythonic
**Cons**: Still single-threaded (GIL), no true parallelism

**Verdict**: Rejected - doesn't solve scalability problem

## Testing Strategy

1. **Unit Tests** (`test_parallel_execution.py`)
   - Tensor serialization correctness
   - Region creator pickling
   - Batch processing logic

2. **Integration Tests** (`test_parallel_execution.py`)
   - Parallel vs sequential equivalence (statistical)
   - Multi-pass execution
   - Error handling and cleanup

3. **Performance Tests** (manual)
   - Speedup measurements
   - Scalability analysis
   - Overhead profiling

4. **Regression Tests**
   - Existing brain tests run in both modes
   - Ensures API compatibility

## Migration Guide

### Existing Code (Sequential)

```python
# No changes needed - sequential mode is default
config = ThaliaConfig(
    brain=BrainConfig(parallel=False),  # or omit (default)
)
brain = DynamicBrain.from_thalia_config(config)
```

### Enabling Parallel Mode

```python
# Minimal change to enable parallelism
config = ThaliaConfig(
    global_=GlobalConfig(device="cpu"),  # Required
    brain=BrainConfig(parallel=True),    # Enable
)
brain = DynamicBrain.from_thalia_config(config)

# Training scripts need __main__ guard on Windows
if __name__ == "__main__":
    main()  # Your training loop
```

### Gradual Migration

1. **Phase 1**: Develop in sequential mode
2. **Phase 2**: Validate in parallel mode (check for errors)
3. **Phase 3**: Benchmark (measure actual speedup)
4. **Phase 4**: Deploy parallel mode if speedup > 1.5x

## Future Enhancements

### Short Term (3-6 months)

1. **Adaptive Batching**: Adjust tolerance based on network activity
2. **Worker Monitoring**: Track utilization, detect crashes
3. **Better Error Messages**: Surface worker exceptions clearly

### Medium Term (6-12 months)

1. **GPU Tensor Sharing**: CUDA IPC for zero-copy GPU communication
2. **Dynamic Load Balancing**: Redistribute regions based on load
3. **Checkpoint Support**: Save/load in parallel mode

### Long Term (12+ months)

1. **Distributed Mode**: Multi-machine via TCP/IP (Ray/MPI)
2. **Async Parallel**: Pipeline execution across timesteps
3. **Hardware Acceleration**: Custom CUDA kernels for IPC

## References

- **Implementation PR**: [Link to PR]
- **Performance Analysis**: `docs/design/parallel_execution.md`
- **API Documentation**: `src/thalia/events/parallel.py`
- **Tests**: `tests/unit/test_parallel_execution.py`

## Related ADRs

- **ADR-001**: Simulation Backend (PyTorch enables GPU support)
- **ADR-003**: Clock-Driven vs Event-Driven (event-driven enables parallelism)
- **ADR-006**: Temporal Coding (spike timing preserved in parallel mode)
- **ADR-013**: Explicit Pathway Projections (pathways work in parallel mode)

## Approval

**Proposed**: 2025-12-12
**Accepted**: 2025-12-12
**Implemented**: 2025-12-12

**Stakeholders**:
- Core Team: Approved
- Contributors: Review period (2 weeks)

**Status**: ✅ Implemented and Documented
