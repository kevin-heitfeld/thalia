# ADR-017: Connection Key Structure for Multi-Source Projections

**Status:** Proposed
**Date:** 2026-02-14
**Deciders:** Architecture Team

## Context

Brain.connections stores AxonalProjection instances that route spikes from source regions to target regions.

Each AxonalProjection can have **multiple sources** (e.g., cortex→striatum:d1, hippocampus→striatum:d1, pfc→striatum:d1 all feed into the same target population).

**Current Implementation (BUGGY):**
```python
connections: Dict[Tuple[RegionName, SpikesSourceKey], AxonalProjection]
# Key: (first_source, "target:population")
# Example: ("cortex", "striatum:d1")
```

**The Bug:**
```python
for (src, _tgt), pathway in self.connections.items():
    if src in outputs:
        source_output = {src: outputs[src]}
        pathway.write_and_advance(source_output)
```

This code assumes `src` is the ONLY source for the pathway. For multi-source projections, it only writes spikes from the first source! The other sources (hippocampus, pfc) are silently dropped.

## Decision

Change the connection key structure to **target-only**:

```python
connections: Dict[SpikesSourceKey, AxonalProjection]
# Key: "target:population"
# Example: "striatum:d1"
```

This makes it clear that:
1. Each AxonalProjection serves ONE target population
2. The projection may have multiple sources (stored in `projection.sources`)
3. We must iterate over `projection.sources` to write from all sources

**Updated write logic:**
```python
for target_key, pathway in self.connections.items():
    # Write from ALL sources that contributed to this projection
    for source_spec in pathway.sources:
        src = source_spec.region_name
        if src in outputs:
            source_output = {src: outputs[src]}
            pathway.write_and_advance(source_output)
```

## Consequences

### Positive
- **Fixes multi-source projection bug**: All sources now correctly write to pathways
- **Clearer semantics**: Key represents the target, not a (source, target) pair
- **Simpler lookups**: Can directly access projection by target name
- **Validates against real biology**: Multiple brain regions DO converge on single targets

### Negative
- **Breaking change**: Existing code that iterates `connections.items()` must be updated
- **Slightly more complex write logic**: Must iterate over `pathway.sources`

## Implementation

1. Update [brain_builder.py](../../src/thalia/brain/brain_builder.py):
   ```python
   conn_key = f"{target_name}:{target_population}"  # Remove source from key
   ```

2. Update [brain.py](../../src/thalia/brain/brain.py):
   ```python
   connections: Dict[SpikesSourceKey, AxonalProjection]

   # Write step (multi-source aware)
   for target_key, pathway in self.connections.items():
       for source_spec in pathway.sources:
           src = source_spec.region_name
           if src in outputs:
               pathway.write_and_advance({src: outputs[src]})
   ```

3. Update `_connection_modules` registration for parameter tracking

4. Update validation scripts and tests

## References

- [ADR-015: Population-Based Routing](adr-015-population-based-routing.md)
- Biology: Convergent projections are fundamental (e.g., corticostriatal + hippocampostriatal + prefrontostriatal → striatum D1/D2)
