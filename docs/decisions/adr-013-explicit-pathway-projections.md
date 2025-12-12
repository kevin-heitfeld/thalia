# ADR-013: Explicit Pathway Projections

**Status**: Accepted  
**Date**: 2025-12-12  
**Authors**: Thalia Project

## Context

When regions communicate across different dimensional spaces (e.g., PFC with 32 neurons sending to cortex L2/3 with 96 neurons), there are two possible approaches:

1. **Implicit projection in adapters**: Region adapters handle dimensional mismatches internally with projection layers
2. **Explicit projection via pathways**: Pathways handle all dimensional transformations, adapters only accept matching dimensions

The original implementation mixed both approaches:
- Cortex adapter had a `_top_down_projection` for PFC→L2/3 (implicit)
- PathwayManager had a `SpikingAttentionPathway` for the same connection (explicit)

This created conflicts where:
- PFC spikes (32D) → Attention pathway (transforms to 96D) → Cortex adapter (tries to project 96D with a 32→96 layer) → dimension mismatch error

## Decision

**When regions have different sizes, dimensional transformations MUST be handled explicitly via pathways. Adapters should NOT contain projection layers for cross-region communication.**

Pathways are **required** when:
- Source and target region sizes differ (e.g., PFC 32 → Cortex L2/3 96)
- Specialized connectivity is needed (attention, replay, etc.)
- Synaptic dynamics/delays are required

Direct connections (without pathways) are acceptable when:
- Source and target sizes match exactly
- Simple pass-through without transformation is sufficient
- No synaptic plasticity or delays needed

Adapters' role is to:
1. Apply temporal dynamics (decay)
2. Route to appropriate internal processing (e.g., which cortical layer)
3. Create output events

Pathways' role is to:
1. Dimensional transformation (n_input → n_output)
2. Synaptic dynamics (STDP, STP, delays)
3. Temporal coding and spike timing

## Rationale

### Clarity and Single Responsibility
- **Pathways** = connectivity and transformation
- **Adapters** = event handling and internal routing
- No overlap, no confusion about which component handles projection

### Biological Plausibility
- Real neural pathways (e.g., cortico-cortical projections) handle dimensional mismatches through connectivity patterns and convergence/divergence
- The pathway weight matrix naturally represents this transformation
- Adapters represent the local dendritic integration, which operates on whatever arrives

### Maintainability
- Only one place to look for dimensional transformations (pathways)
- Adapter code is simpler and more uniform
- Easier to debug dimension mismatches (check pathway configuration)

### Composability
- Pathways can be swapped or reconfigured without modifying adapters
- Specialized pathways (attention, replay) naturally handle their own transformations
- Clear dependency: adapters depend on pathways, not vice versa

## Consequences

### Positive
- ✅ Simplified adapter code (remove conditional projection logic)
- ✅ Clear error messages when dimensions don't match
- ✅ Pathway configurations document all dimensional transformations
- ✅ Easier to add new specialized pathways
- ✅ No hidden projection layers in adapters

### Negative
- ⚠️ Slightly more verbose pathway configuration when sizes differ
  - Mitigation: Configuration is explicit and self-documenting
  
### Neutral
- ℹ️ When region sizes match, direct connections work without pathways
  - Pathways are optional for same-size connections (can add for synaptic dynamics/delays)
  - Pathways are **required** when dimensional transformation is needed

### Migration
Existing code changes:
1. Remove `_top_down_projection` from `EventDrivenCortex`
2. Remove size-checking and conditional projection logic
3. Add assertion that received spikes match expected size
4. Ensure all PathwayManager pathways have correct `n_input`/`n_output`

## Examples

### Before (Mixed Approach)
```python
# In cortex adapter __init__
if pfc_size > 0:
    self._top_down_projection = nn.Linear(pfc_size, cortex.l23_size)

# In _process_spikes
if source == "pfc":
    if self._top_down_projection is not None:
        projected = self._top_down_projection(input_spikes)
        # ...
```

### After (Explicit Pathway)
```python
# In PathwayManager
self.attention = SpikingAttentionPathway(
    SpikingAttentionPathwayConfig(
        n_input=self._sizes['pfc'],        # 32
        n_output=self._sizes['cortex_l23'], # 96
        device=self.device,
    )
)

# In cortex adapter _process_spikes
if source == "pfc":
    # Pathway already transformed to L2/3 size
    assert input_spikes.shape[0] == self.impl.l23_size, \
        f"PFC input must match L2/3 size via pathway projection"
    self._pending_top_down = torch.sigmoid(input_spikes.float())
```

## Related

- **ADR-005**: No Batch Dimension - All tensors are 1D, simplifying dimension checks
- **ADR-009**: Pathway-Neuron Consistency - Pathways use same neuron models as regions
- **ADR-010**: Region Axonal Delays - Delays handled by pathways, not adapters

## References

- `src/thalia/pathways/manager.py` - Pathway configuration
- `src/thalia/events/adapters/base.py` - Adapter base class
- `src/thalia/pathways/spiking_pathway.py` - Base pathway with n_input/n_output
