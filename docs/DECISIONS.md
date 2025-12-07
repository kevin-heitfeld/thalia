# Architecture Decision Records

This document tracks key architectural decisions for the Thalia project.

**Note**: Individual ADRs have been moved to the [`decisions/`](./decisions/) directory. See the [decisions README](./decisions/README.md) for the full list.

## Active Decisions

- **[ADR-001: Simulation Backend](./decisions/adr-001-simulation-backend.md)** - Use PyTorch with GPU acceleration
- **[ADR-002: Numeric Precision](./decisions/adr-002-numeric-precision.md)** - Mixed precision with float32 for critical state
- **[ADR-003: Clock-Driven Simulation](./decisions/adr-003-clock-driven.md)** - Fixed timestep simulation loop

## Adding New ADRs

To add a new architecture decision record:

1. Create a new file in `decisions/` following the naming convention `adr-NNN-short-title.md`
2. Use the template below
3. Update this file and `decisions/README.md` with a link to the new ADR

### ADR Template

```markdown
# ADR-XXX: Title

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
