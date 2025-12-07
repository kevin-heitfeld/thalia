# ADR-003: Clock-Driven Simulation

**Date**: 2025-11-28  
**Status**: Accepted

### Context
We must choose the simulation scheduling model: event-driven vs clock-driven (fixed timestep). Biological plausibility and implementation complexity trade off here.

### Decision
Use a clock-driven fixed-timestep simulation loop as the canonical execution model.

### Rationale
- Simpler to implement and reason about for multi-region interactions
- Matches the timestep-based design in most neuroscience models
- Deterministic execution aids reproducibility and testing

### Consequences
- May be less efficient for extremely sparse spike trains compared to event-driven methods
- Easier to implement oscillators, STP, and other time-dependent models
