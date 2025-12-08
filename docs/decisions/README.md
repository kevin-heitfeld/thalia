# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting important technical decisions made during the development of Thalia.

## Active Decisions

- **[ADR-001: Simulation Backend](./adr-001-simulation-backend.md)** - Use PyTorch with GPU acceleration
- **[ADR-002: Numeric Precision](./adr-002-numeric-precision.md)** - Mixed precision with float32 for critical state
- **[ADR-003: Clock-Driven Simulation](./adr-003-clock-driven.md)** - Fixed timestep simulation loop
- **[ADR-004: Bool Spikes](./adr-004-bool-spikes.md)** - Use bool tensors for spike representation (8× memory savings)
- **[ADR-005: No Batch Dimension](./adr-005-no-batch-dimension.md)** - Remove batch dimension (single-brain architecture)
- **[ADR-006: Temporal Coding](./adr-006-temporal-coding.md)** - Sensory pathways use temporal/latency coding
- **[ADR-007: PyTorch Consistency](./adr-007-pytorch-consistency.md)** - Use forward() instead of encode()/decode()

## ADR Format

Each ADR includes:
- **Date** - When the decision was made
- **Status** - Accepted, Proposed, Deprecated, or Superseded
- **Context** - The problem or decision to be made
- **Decision** - What was decided
- **Rationale** - Why this decision was made
- **Consequences** - Trade-offs and implications

## Related Documentation

- **[Architecture](../architecture/)** - High-level system architecture
- **[Design Docs](../design/)** - Detailed design specifications
