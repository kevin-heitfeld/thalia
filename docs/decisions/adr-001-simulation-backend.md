# ADR-001: Simulation Backend

**Date**: 2025-11-28  
**Status**: Accepted

### Context
We need to choose between pure NumPy and PyTorch for the simulation backend. The choice affects performance, ease of development, and future scalability.

### Decision
**Use PyTorch with GPU acceleration.**

### Rationale
- GPU is necessary for performance with large-scale networks (millions of neurons)
- Efficient tensor operations for spike-based processing
- `torch.compile()` (PyTorch 2.0+) offers significant speedups for simulation loops
- Easy transition between CPU debugging and GPU training
- Strong ecosystem for neural network operations

### Consequences
- Requires CUDA-capable GPU for full performance
- Adds PyTorch as a core dependency
- Learning curve for those unfamiliar with PyTorch
- Can fall back to CPU for development/testing
