# Architecture Decision Records

This document tracks key architectural decisions for the THALIA project.

---

## ADR-001: Simulation Backend

**Date**: 2025-11-28  
**Status**: Accepted

### Context
We need to choose between pure NumPy and PyTorch for the simulation backend. The choice affects performance, ease of development, and future scalability.

### Decision
**Use PyTorch with GPU acceleration.**

### Rationale
- GPU is necessary for performance with large-scale networks (millions of neurons)
- PyTorch provides autograd for surrogate gradient training
- `torch.compile()` (PyTorch 2.0+) offers significant speedups for simulation loops
- Easy transition between CPU debugging and GPU training
- Strong ecosystem for neural network operations

### Consequences
- Requires CUDA-capable GPU for full performance
- Adds PyTorch as a core dependency
- Learning curve for those unfamiliar with PyTorch
- Can fall back to CPU for development/testing

---

## ADR-002: Numerical Precision

**Date**: 2025-11-28  
**Status**: Accepted

### Context
Need to decide on floating-point precision for membrane potentials, weights, and spike computations.

### Decision
**Use Float32 (single precision) as default, with quantization as future optimization.**

### Rationale
- Float32 is sufficient for neuron dynamics (biological neurons are far noisier)
- Halves memory usage compared to Float64
- Better GPU performance (especially on consumer GPUs)
- Quantization (int8/int16) can be explored for extreme scaling
- Matches PyTorch defaults and deep learning conventions

### Consequences
- Potential numerical issues in very long simulations (monitor for drift)
- Need to test stability of learning rules at this precision
- Future work: implement optional quantized inference mode

---

## ADR-003: Clock-Driven Simulation (Initial)

**Date**: 2025-11-28  
**Status**: Accepted

### Context
SNN simulation can be clock-driven (fixed timestep) or event-driven (spike-triggered). Need to choose initial approach.

### Decision
**Start with clock-driven simulation, migrate to hybrid later.**

### Rationale
- Simpler to implement and debug
- Better GPU parallelization (regular computation pattern)
- Easier integration with PyTorch
- Event-driven is more efficient for sparse activity but complex to implement correctly

### Consequences
- Less efficient for very sparse networks
- Fixed dt=1ms provides good balance of accuracy vs. speed
- Will need refactoring for neuromorphic hardware targets

---

## Template for Future Decisions

```markdown
## ADR-XXX: Title

**Date**: YYYY-MM-DD  
**Status**: Proposed | Accepted | Deprecated | Superseded

### Context
What is the issue that we're seeing that motivates this decision?

### Decision
What is the change that we're proposing and/or doing?

### Rationale
Why is this the best choice? What alternatives were considered?

### Consequences
What becomes easier or more difficult because of this change?
```
