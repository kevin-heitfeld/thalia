# Pathway Selection Guide: AxonalProjection vs SpikingPathway

**Date**: December 17, 2025  
**Status**: ✅ Production Ready (v2.0 Architecture)  
**Related**: See `REFACTOR_EXPLICIT_AXONS_SYNAPSES.md` for implementation details

## Executive Summary

Thalia v2.0 introduces explicit separation of **axons** (transmission) from **synapses** (integration). This guide helps you choose the right pathway type for biological accuracy and code clarity.

**Quick Decision Tree:**
```
Does the connection contain actual neural populations?
├─ YES → Use SpikingPathway (e.g., thalamic relay, interneurons)
└─ NO  → Use AxonalProjection (recommended for most inter-region connections)
```

---

## Two Architectural Patterns

### Pattern 1: AxonalProjection (Recommended)

**What it represents**: Pure axon bundles (like corpus callosum, fornix, internal capsule)

**Characteristics**:
- ✅ NO weights (synapses belong to target)
- ✅ NO neurons (just transmission)
- ✅ NO learning (plasticity at synapses, not axons)
- ✅ Has delays (axonal conduction time)
- ✅ Concatenation (multi-source routing)

**Biology**: Real axonal tracts transmit action potentials with delays but don't compute or learn. Synapses are located at target dendrites.

**Usage**:
```python
# Single source
builder.connect("cortex", "striatum", 
                pathway_type="axonal", 
                source_port="l5")

# Multi-source (automatically merged into one AxonalProjection)
builder.connect("cortex", "striatum", pathway_type="axonal")
builder.connect("hippocampus", "striatum", pathway_type="axonal")
builder.connect("pfc", "striatum", pathway_type="axonal")
# → Creates single AxonalProjection with 3 sources

# Custom delay
builder.connect("hippocampus", "pfc",
                pathway_type="axonal",
                axonal_delay_ms=3.0)
```

**When to use**:
- Long-range inter-region connections (cortex ↔ hippocampus)
- Direct projections with no intermediate processing
- Any connection where synapses belong to target region (standard case)

---

### Pattern 2: SpikingPathway (Special Cases)

**What it represents**: Pathway with actual neural populations

**Characteristics**:
- ✅ Has weights (synaptic connections)
- ✅ Has neurons (LIF, fast-spiking interneurons, etc.)
- ✅ Has learning (STDP, BCM, three-factor, etc.)
- ✅ Has delays (optional)
- ✅ Computation (filtering, gating, transformation)

**Biology**: Some pathways contain real neurons that process information (thalamic relay cells, striatal feedforward interneurons).

**Usage**:
```python
# Don't use SpikingPathway for direct connections anymore!
# builder.connect("cortex", "striatum", "spiking")  # OLD WAY

# Instead: Thalamus IS a region with neurons
builder.add_component("thalamus", "thalamus", n_input=784, n_output=256)
builder.connect("thalamus", "cortex", pathway_type="axonal")  # NEW WAY
```

**When to use**:
- **Relay stations**: Thalamic relay (has real thalamic neurons)
- **Pathway interneurons**: Striatal feedforward inhibition (FSIs)
- **Pathway-specific learning**: Different plasticity rules per pathway
- **Intermediate computation**: Filtering, gating, or transformation needed

---

## Real-World Examples

### Example 1: Sensorimotor Architecture (Correct v2.0)

```python
def _build_sensorimotor(builder):
    # REGIONS (have neurons)
    builder.add_component("thalamus", "thalamus", n_input=128, n_output=128)
    builder.add_component("cortex", "cortex", n_output=500)
    builder.add_component("hippocampus", "hippocampus", n_output=200)
    builder.add_component("pfc", "prefrontal", n_output=300)
    builder.add_component("striatum", "striatum", n_output=150)
    builder.add_component("cerebellum", "cerebellum", n_output=100)

    # CONNECTIONS (use axonal projections)
    # Why axonal? These are pure long-range projections.
    # Synapses are owned by target regions (at their dendrites).
    
    builder.connect("thalamus", "cortex", pathway_type="axonal")
    builder.connect("cortex", "hippocampus", pathway_type="axonal")
    builder.connect("hippocampus", "cortex", pathway_type="axonal")
    builder.connect("cortex", "pfc", pathway_type="axonal")
    
    # Multi-source to striatum (automatically merged)
    builder.connect("cortex", "striatum", pathway_type="axonal")
    builder.connect("hippocampus", "striatum", pathway_type="axonal")
    builder.connect("pfc", "striatum", pathway_type="axonal")
    
    builder.connect("striatum", "pfc", pathway_type="axonal")
    builder.connect("cortex", "cerebellum", pathway_type="axonal")
    builder.connect("pfc", "cerebellum", pathway_type="axonal")
    builder.connect("cerebellum", "cortex", pathway_type="axonal")
```

**Result**: Clear separation of axons (routing) from synapses (integration).

---

### Example 2: When SpikingPathway IS Correct

#### Case 1: Thalamic Relay Station
```python
# CORRECT: Thalamus is a REGION (has relay neurons)
builder.add_component("thalamus", "thalamus", n_input=784, n_output=256)
builder.connect("thalamus", "cortex", pathway_type="axonal")

# WRONG: Don't use SpikingPathway to represent thalamus
# builder.connect("retina", "cortex", "spiking")  # Missing thalamus!
```

**Why**: Thalamic relay neurons filter sensory input (center-surround), gate attention (TRN), and switch between burst/tonic modes. This is a REGION, not a pathway.

#### Case 2: Striatal Feedforward Inhibition
```python
# If modeling FSIs INSIDE the corticostriatal pathway
# (This is advanced circuit modeling)
builder.connect("cortex", "striatum", 
                pathway_type="spiking",  # Has FSI neurons
                neurons="fast_spiking_interneuron")
```

**Why**: Fast-spiking interneurons provide feedforward inhibition to MSNs. They are actual neurons along the pathway, not just axons.

#### Case 3: Pathway-Specific Plasticity
```python
# Rare case: Different learning rules per pathway
builder.connect("hippocampus", "cortex",
                pathway_type="spiking",
                learning_rule="acetylcholine_gated_stdp")

builder.connect("pfc", "cortex",
                pathway_type="spiking",
                learning_rule="dopamine_gated_stdp")
```

**Why**: If learning rules vary by source (not just target), the pathway needs its own plasticity.

---

## Common Mistakes

### ❌ Mistake 1: Using SpikingPathway for Everything

```python
# OLD WAY (v1.x - confusing)
builder.connect("cortex", "striatum", "spiking")  # Has neurons? Why?
builder.connect("hippocampus", "pfc", "spiking")  # More neurons? Confusing!
```

**Problem**: Creates "ghost synapses" in pathways. Where are the real corticostriatal synapses? In the pathway or striatum?

**Fix**:
```python
# NEW WAY (v2.0 - clear)
builder.connect("cortex", "striatum", "axonal")      # Just axons
builder.connect("hippocampus", "pfc", "axonal")      # Just axons
# Synapses are owned by striatum and pfc (at their dendrites)
```

---

### ❌ Mistake 2: Using AxonalProjection for Relay Stations

```python
# WRONG: Missing thalamus entirely
builder.connect("retina", "cortex", "axonal")  # Where's the thalamus?
```

**Problem**: Thalamus is a real brain region with relay neurons. You can't skip it.

**Fix**:
```python
# CORRECT: Thalamus is a region
builder.add_component("thalamus", "thalamus", n_input=784, n_output=256)
builder.connect("thalamus", "cortex", "axonal")
```

---

### ❌ Mistake 3: Double Synapses

```python
# OLD: Both pathway AND region have weights
pathway = SpikingPathway(n_input=224, n_output=224)  # Weights [224, 224]
striatum.d1_weights = [70, 224]                       # Weights [70, 224]
striatum.d2_weights = [70, 224]                       # More weights!
```

**Problem**: Which weights are "the" corticostriatal synapses? Ownership unclear.

**Fix**:
```python
# NEW: Only striatum has weights
projection = AxonalProjection(sources=[...])  # NO weights
striatum.afferent_synapses.weights = [70, 224]  # THE synapses (at MSN dendrites)
```

---

## Migration Checklist

When updating existing code:

- [ ] Identify all inter-region connections
- [ ] For each connection, ask: "Does this pathway have neurons?"
  - NO → Change to `pathway_type="axonal"`
  - YES → Keep as `pathway_type="spiking"` OR make it a region
- [ ] Update presets (sensorimotor, minimal, etc.)
- [ ] Test that multi-source connections still work
- [ ] Verify checkpoint compatibility (if needed)

---

## Benefits of v2.0 Architecture

### Biological Accuracy
- ✅ Axons ≠ Synapses (explicit separation)
- ✅ Synapses located at target dendrites (matches biology)
- ✅ Clear ownership of weights (no ambiguity)
- ✅ Accurate `n_input` vs `n_output` semantics

### Code Clarity
- ✅ Easier to understand where synapses are
- ✅ Simpler growth (only target resizes)
- ✅ Better checkpointing (weights in regions)
- ✅ Less coupling between components

### Performance
- ✅ Fewer weight matrices (no double synapses)
- ✅ Simpler routing (no extra neurons in pathways)
- ✅ Faster growth (no dual resize)

---

## Default Recommendation

**For new development**: Use `pathway_type="axonal"` for all inter-region connections unless you have a specific biological reason to use SpikingPathway.

**Rule of thumb**: If you can't name the neuron population in the pathway (e.g., "thalamic relay neurons", "fast-spiking interneurons"), use AxonalProjection.

---

## See Also

- `REFACTOR_EXPLICIT_AXONS_SYNAPSES.md` - Implementation details
- `ARCHITECTURE_OVERVIEW.md` - System architecture
- `../patterns/component-parity.md` - Component design patterns
- `brain_builder.py` - BrainBuilder implementation with examples

---

**Questions?** See the "Pathway Selection Guide" section in `REFACTOR_EXPLICIT_AXONS_SYNAPSES.md` for additional examples and biological justification.
