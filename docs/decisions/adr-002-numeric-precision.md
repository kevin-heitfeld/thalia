# ADR-002: Numeric Precision

**Date**: 2025-11-28  
**Status**: Accepted

### Context
Choosing numeric precision impacts memory footprint and compute speed. Mixed precision can reduce memory and increase throughput but may hurt numerical stability for certain learning rules.

### Decision
Adopt mixed precision where safe (AMP for forward pass) and keep critical state (eligibility traces, accumulators) in float32.

### Rationale
- Mixed precision provides memory and speed benefits on modern GPUs
- Critical small-value accumulators (eligibility traces) are sensitive to reduced precision
- PyTorch's AMP makes integration straightforward

### Consequences
- Need explicit checks in learning rules to keep critical state in float32
- Increased testing to validate numerical stability
