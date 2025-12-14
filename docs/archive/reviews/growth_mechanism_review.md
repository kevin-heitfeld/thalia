# Growth Mechanism Review & Analysis

**Date**: December 14, 2025  
**Status**: ⚠️ HISTORICAL DOCUMENT - Migration Complete

---

## ⚠️ NOTE: This is a Historical Design Document

This document describes the **OLD growth API** and the reasoning behind migrating to the unified API. It is preserved for historical context.

**Current Status (December 2025)**:
- ✅ Migration to unified API (`grow_input()` / `grow_output()`) is **complete**
- ✅ Old API methods (`add_neurons`, `grow_source`, `grow_target`) have been **removed**
- ✅ See `docs/architecture/UNIFIED_GROWTH_API.md` for current implementation

---

## Executive Summary

### The Question
**Why do pathways need `grow_source()` and `grow_target()` while regions only need `add_neurons()`?**

### The Answer (REVISED)
**Initial hypothesis** (WRONG): "Regions grow symmetrically, pathways grow asymmetrically"

**Corrected finding** (RIGHT): **Both regions AND pathways need bidirectional growth!**
- Regions originally only had `add_neurons()` → **INCOMPLETE IMPLEMENTATION**
- Regions were **missing** `grow_input()` method
- Once added, both follow the same pattern → **unification is natural**

### Key Discoveries

1. **Both have `n_input ≠ n_output`**: Regions transform dimensionality just like pathways
2. **Both need bidirectional growth**: Input dimension AND output dimension can grow
3. **Old asymmetry was a bug**: Regions couldn't handle upstream growth
4. **Unification achieved**: Renamed to `grow_input()` / `grow_output()` for both

### Implementation Complete

✅ **Implement unified growth API**:
```python
# ALL neural components:
component.grow_input(n_new)   # Expand input dimension
component.grow_output(n_new)  # Expand output dimension
```

This makes the architecture **more consistent**, **more complete**, and **easier to understand**.

---

## CRITICAL CLARIFICATION: What ARE Pathways?

### Common Misconception

**Wrong assumption**: "Pathways are just weight matrices connecting two regions"

**Reality**: **Pathways ARE neural populations with their own neurons!**

### Pathway Neuron Populations

```python
# From SpikingPathway.__init__:
self.neurons = self._create_neurons()  # ConductanceLIF neurons!

# Pathways have:
self.neurons = ConductanceLIF(n_neurons=config.n_output, ...)
self.synaptic_current = torch.zeros(config.n_output, ...)
self.firing_rate_estimate = torch.zeros(config.n_output, ...)
```

**Key insight**: Pathways have `n_output` neurons that:
- Receive weighted input from `n_input` source neurons
- Integrate via LIF dynamics
- Spike and send to target region
- Learn via STDP

### Why Pathways Need Neurons

**Biological reality**: Axonal projections are NOT passive wires!

```
Source Region (256 neurons)
    ↓ [weights 128×256]
Pathway Neurons (128 LIF neurons)  ← ACTIVE COMPUTATION HERE!
    - Integrate weighted input
    - Apply temporal filtering  
    - Spike with delays (2-10ms)
    - Learn via STDP
    ↓
Target Region (128 neurons)
```

**Pathways provide**:
1. **Dimensionality transformation**: 256 inputs → 128 outputs
2. **Temporal filtering**: Synaptic current integration
3. **Learning substrate**: STDP modifies weights based on pathway neuron spikes
4. **Axonal delays**: Realistic transmission times
5. **Nonlinearity**: Spike thresholding

### Comparison: Pathways vs Regions

| Aspect | Regions | Pathways |
|--------|---------|----------|
| **Has neurons?** | ✅ Yes (`n_output` neurons) | ✅ Yes (`n_output` neurons) |
| **Has input weights?** | ✅ Yes (`[n_output, n_input]`) | ✅ Yes (`[n_output, n_input]`) |
| **Population 1** | Internal neurons | Pathway neurons |
| **Population 2** | ❌ None (input is external) | ❌ None (input is external) |
| **Owns neurons?** | ✅ Yes | ✅ Yes |
| **Controls input size?** | ❌ No (determined by source) | ❌ No (determined by source) |

**Critical realization**: **Pathways do NOT have "two populations"!**

They have:
- ONE neuron population (`n_output` pathway neurons)  
- ONE input dimension (`n_input` from source region)
- Same structure as regions!

### Answer to Your Question

**Q: "Why do pathways need two populations while regions bake input size into weights? Couldn't pathways do the same?"**

**A: They DO do the same! Pathways don't have two populations.**

Both regions and pathways:
- Have ONE neuron population (output dimension)
- Receive external input (input dimension) 
- Store input in weight matrix dimensions
- Need to handle growth of BOTH dimensions

**The confusion arose because**:
- Methods named `grow_source` / `grow_target` sound like separate populations
- But they really mean `grow_input` / `grow_output` (dimension growth)
- Both components follow the exact same pattern!

### Biological Evidence

**Real axonal projections have active neurons**:

1. **Thalamocortical pathway**: Thalamic relay neurons actively gate and filter sensory input
2. **Corticothalamic pathway**: Layer 6 pyramidal neurons provide active feedback
3. **Hippocampal pathways**: Perforant path has active neurons in entorhinal cortex
4. **Long-range cortical**: Area-to-area projections have active neurons at both ends

**The "pathway neurons" are often the OUTPUT neurons of the source region**, but in our implementation, pathways have their own dedicated neuron population for:
- Dimensionality transformation (e.g., 256 cortex → 128 hippocampus)
- Independent learning dynamics
- Separate modulation (pathway-specific plasticity)

---

## Current Implementation Analysis### 1. Pathway Growth: Bidirectional (Source + Target)

**Why Bidirectional?**
```python
# Pathway weight matrix: [n_output, n_input]
#                         [target,   source]
weights = torch.randn(target_neurons, source_neurons)
```

Pathways have TWO independently-sized dimensions:
- **n_input** (source region size) - grows with `grow_source()`
- **n_output** (target region size) - grows with `grow_target()`

**When each is needed**:
```
Source region grows:
  cortex: 256 → 300 neurons
  ↓
  cortex_to_hippocampus pathway must grow_source()
  weights: [128, 256] → [128, 300]
           [target stays same, source grows]

Target region grows:
  hippocampus: 128 → 150 neurons
  ↓
  cortex_to_hippocampus pathway must grow_target()
  weights: [128, 300] → [150, 300]
           [target grows, source stays same]
```

### 2. Region Growth: Output-Only (add_neurons expands output dimension)

**CRITICAL INSIGHT**: Regions also have `n_input ≠ n_output`!

```python
# Region weight matrices are NOT square!
# Cortex example:
config = LayeredCortexConfig(
    n_input=784,   # MNIST input (28x28)
    n_output=128   # Cortex output size
)

# Internal weights:
w_input_l4: [l4_size, 784]     # [~51, 784] - NOT square!
w_l4_l23: [l23_size, l4_size]  # [~51, ~51] - square (internal)
w_l23_l5: [l5_size, l23_size]  # [~26, ~51] - NOT square!
```

**Why only add_neurons() (not grow_input)?**

Regions have **external input** (from pathways) vs **internal neurons**:
- `n_input`: External input dimension (from upstream pathways, FIXED by source)
- `n_output`: Internal neuron population (CAN grow)
- Region doesn't control its input size - the **source region** does!

**Example - Cortex add_neurons()**:
```python
# Cortex has n_input=784 (sensory input), n_output=128 (neurons)
cortex.add_neurons(20)  # Add 20 neurons

# What grows:
✅ n_output: 128 → 148 neurons
✅ All internal weight OUTPUT dimensions:
   w_input_l4: [51, 784] → [~60, 784]  (rows grow, cols FIXED)
   w_l4_l23: [51, 51] → [~60, ~60]    (both grow - internal layer)
   w_l23_l5: [26, 51] → [~30, ~60]    (output rows grow, input cols grow)

❌ n_input: 784 stays FIXED (external input from sensory pathway)
❌ Input dimension of external weights (cols) FIXED

# To grow input dimension, the SOURCE must grow:
# If visual_pathway.grow_target(20), then cortex receives 804 inputs
```

**Key Difference**:
- **Pathways**: Connect two INDEPENDENT populations → need both grow methods
- **Regions**: Have EXTERNAL input + INTERNAL population → only grow population

---

## Issues & Concerns

### ✅ Non-Issues (Working as Designed)

1. **"Pathways have 2 methods, regions have 1"**
   - **Not an issue**: Reflects architectural difference
   - Pathways connect different populations (asymmetric growth)
   - Regions grow their own population (symmetric growth)

2. **"Inconsistent naming"**
   - **Minor issue**: `add_neurons()` vs `grow_target()/grow_source()`
   - **Justification**: Different semantics (whole population vs directional)

### ⚠️ Actual Issues Found

#### Issue 1: Confusing Semantics of `add_neurons()` for Pathways

**Current behavior**:
```python
# For pathways, add_neurons() delegates to grow_target()
pathway.add_neurons(n_new=10)
# ↓ Actually grows target dimension only!
# weights: [80, 100] → [90, 100]
```

**Problem**: Name suggests "add neurons to pathway" but actually means "grow target side only"

**Recommendation**:
```python
# Option A: Deprecate add_neurons() for pathways (breaking change)
pathway.grow_target(10)  # Clear and explicit

# Option B: Keep for backward compatibility but warn
@deprecated("Use grow_target() or grow_source() for clarity")
def add_neurons(self, n_new): ...
```

**Current Status**: ✅ We already marked it DEPRECATED in docstring

#### Issue 2: Region Growth Doesn't Update Connected Pathways Automatically

**Current workflow**:
```python
# Manual coordination required
region.add_neurons(100)
# Then manually:
for input_pathway in region.input_pathways:
    input_pathway.grow_target(100)
for output_pathway in region.output_pathways:
    output_pathway.grow_source(100)
```

**Problem**: Easy to forget pathway updates, causing dimension mismatches

**Solution**: ✅ **Already implemented!** `GrowthCoordinator.coordinate_growth()`
```python
coordinator.coordinate_growth('cortex', n_new=100)
# ✅ Grows cortex
# ✅ Automatically grows all connected pathways
```

**Remaining Issue**: Users might call `region.add_neurons()` directly, bypassing coordinator

**Recommendation**: Document the pattern strongly, possibly add warnings

#### Issue 3: No Validation of Dimension Consistency

**Current state**: No automatic check that pathway dimensions match regions

**Example failure scenario**:
```python
cortex.add_neurons(100)  # cortex now 356 neurons
# Oops, forgot to grow cortex_to_hippocampus pathway!
# pathway still expects 256 input neurons
# Next forward pass: RuntimeError: dimension mismatch
```

**Recommendation**: Add dimension validation in `EventDrivenBrain`
```python
def validate_connectivity(self):
    """Validate all pathway dimensions match regions."""
    for pathway_name, pathway in self.pathways.items():
        source_size = self.regions[pathway.source].n_output
        target_size = self.regions[pathway.target].n_input

        if pathway.n_input != source_size:
            raise DimensionMismatchError(...)
        if pathway.n_output != target_size:
            raise DimensionMismatchError(...)
```

#### Issue 4: Synapse Count Calculation Uses Old Dimensions

**In `GrowthCoordinator`**:
```python
# Get source size BEFORE growth
n_source = pathway.config.n_input  # ← OLD size
pathway.grow_target(n_new=10)
# Calculate synapses with OLD source size
n_synapses = n_new * n_source  # ← Should use NEW source size? No!
```

**Wait, is this correct?**

Actually, **YES** - this is correct! We want OLD dimensions because:
```
# Before: weights [80, 100]
# After:  weights [90, 100]
# Synapses added: 10 new rows × 100 columns = 1000 new synapses
# Using OLD n_source=100 is correct!
```

**Status**: ✅ Not an issue, working as intended

#### Issue 5: Missing Growth Support for Some Components

**Currently supports growth**:
- ✅ SpikingPathway
- ✅ SpikingAttentionPathway
- ✅ SpikingReplayPathway
- ✅ LayeredCortex
- ✅ Striatum
- ✅ Hippocampus (trisynaptic)
- ✅ Cerebellum

**Missing growth support**:
- ❌ SensoryPathway (VisualPathway, AuditoryPathway, LanguagePathway)
- ❌ Thalamus
- ❌ PrefrontalCortex (might have it, need to check)

**Recommendation**:
- Document which components support growth
- Add growth to sensory pathways if needed
- Add validation to reject growth for unsupported components

---

## Architectural Questions

### Q1: Should we unify the interface?

**CRITICAL REALIZATION**: Both regions AND pathways need TWO growth methods!

**Current (INCOMPLETE) design**:
```python
# Regions: Only grow output ❌ MISSING grow_input!
region.add_neurons(n_new)  # Grows output dimension only
# Missing: region.grow_input(n_new)  # Should expand input dimension

# Pathways: Can grow either dimension ✅
pathway.grow_target(n_new)  # Target region grew
pathway.grow_source(n_new)  # Source region grew
```

**Fixed design - Option A: Unified symmetric interface (RECOMMENDED)**
```python
# BOTH regions and pathways need bidirectional growth!

# Regions:
region.grow_output(n_new)  # Grow neuron population (current: add_neurons)
region.grow_input(n_new)   # Expand input weights when upstream grows (NEW!)

# Pathways:
pathway.grow_output(n_new) # Grow target dimension (current: grow_target)
pathway.grow_input(n_new)  # Grow source dimension (current: grow_source)
```

**Pros**:
- **Perfectly unified!** Same interface for both component types
- Clear semantics: input dimension vs output dimension
- Self-documenting: immediately obvious what each method does
- Architecturally correct: both need to handle growth from both ends

**Cons**:
- Breaking change: need to rename existing methods
- Backward compatibility requires deprecation period

**Option B: Keep current names but add missing methods**
```python
# Regions:
region.add_neurons(n_new)  # Grow neuron population (output)
region.grow_input(n_new)   # Expand input weights (NEW!)

# Pathways:
pathway.grow_target(n_new) # Grow target dimension
pathway.grow_source(n_new) # Grow source dimension
```

**Pros**:
- No breaking changes
- Backward compatible
- Less refactoring needed

**Cons**:
- Inconsistent naming (add_neurons vs grow_target/grow_source)
- Less obvious that regions and pathways follow same pattern
- `add_neurons` doesn't make clear it's output-only

**Option C: Dimension-explicit naming**
```python
# Make input/output explicit in ALL method names

# Regions:
region.grow_output_dimension(n_new)
region.grow_input_dimension(n_new)

# Pathways:
pathway.grow_output_dimension(n_new)
pathway.grow_input_dimension(n_new)
```

**Pros**: Maximally explicit, impossible to confuse
**Cons**: Verbose method names

**Recommendation**: ✅ **Option A - Unified symmetric interface**

**Rationale**:
1. **Architecturally accurate**: Both regions and pathways have bidirectional growth
2. **Maximally clear**: `grow_input()` and `grow_output()` are self-documenting
3. **Unified pattern**: Same API for all neural components
4. **Future-proof**: Easy to understand for new code

**Migration path**:
```python
# Phase 1: Add new methods, deprecate old
region.add_neurons(10)  # DeprecationWarning: Use grow_output()
region.grow_output(10)  # New method
pathway.grow_target(10)  # DeprecationWarning: Use grow_output()
pathway.grow_output(10)  # New method

# Phase 2: Update all code to new API
# Phase 3: Remove deprecated methods in v3.0
```

### Q2: Should GrowthCoordinator be mandatory?

**Current**: Users can call `region.add_neurons()` directly, bypassing coordinator

**Options**:
1. **Permissive** (current): Allow direct calls, coordinator is optional helper
2. **Recommended**: Add warnings when direct calls are used
3. **Strict**: Make regions raise error if grown without coordinator

**Recommendation**: **Option 2 - Recommended pattern**
```python
def add_neurons(self, n_new):
    warnings.warn(
        "Direct region growth bypasses pathway coordination. "
        "Consider using GrowthCoordinator.coordinate_growth() instead.",
        UserWarning
    )
    self._grow_impl(n_new)
```

---

## Recommended Improvements

### Priority 1: Critical (Do Now)

1. **❌ BROKEN: Add `grow_input()` method to regions** ✅ Critical missing feature!
   ```python
   region.grow_input(n_new)  # Expand input weight dimension
   ```
   **Why critical**: Currently regions can't handle upstream growth, will crash with dimension mismatch

2. **Add dimension validation** ✅ Easy win
   ```python
   brain.validate_connectivity()  # Check all pathways match regions
   ```

3. **Document growth patterns** ✅ Already doing this
   - When to use coordinator vs direct growth
   - Which components support growth
   - Dimension mismatch troubleshooting

### Priority 2: Important (Soon)

3. **Add growth support to missing components**
   - SensoryPathway subclasses
   - Thalamus
   - Document which don't support growth (and why)

4. **Add warnings for direct region growth**
   ```python
   region.add_neurons(10)  # Warning: Consider using GrowthCoordinator
   ```

5. **Add growth metrics to diagnostics**
   ```python
   diag = region.get_diagnostics()
   # diag['growth_history'] = [...]
   # diag['growth_capacity'] = 0.23  # 23% capacity used
   ```

### Priority 3: Nice to Have (Future)

6. **Automatic capacity monitoring**
   ```python
   brain.auto_grow = True  # Automatically grow when capacity > 85%
   ```

7. **Growth visualization**
   ```python
   brain.plot_growth_history()  # Timeline of all growth events
   ```

8. **Checkpoint growth compatibility checks**
   ```python
   # Warn if loading checkpoint with different capacity
   brain.load_checkpoint('old.pt')  # Warning: Checkpoint has 100 neurons, current has 150
   ```

---

## Conclusion

### Summary of Findings

✅ **What's Working Well**:
- Bidirectional pathway growth is correctly implemented
- GrowthCoordinator successfully coordinates region + pathway growth
- Growth preserves existing knowledge (weights, traces, neuron state)
- Comprehensive test coverage (8 passing tests)
- Synapse count calculations are correct
- **Architecture is biologically accurate**: `n_input ≠ n_output` matches real neural structures

⚠️ **What Needs Improvement**:
- Missing dimension validation (easy to break without noticing)
- Missing growth support for some components
- No warnings when bypassing coordinator
- Documentation could be clearer on when to use which method

🔍 **Key Insight - REVISED**:
The initial assumption was **WRONG**. Both regions AND pathways need **bidirectional growth**:
- **Regions**: Need `grow_output()` (internal neurons) AND `grow_input()` (when upstream grows)
- **Pathways**: Need `grow_output()` (target side) AND `grow_input()` (source side)
- **Unification is possible**: Both follow the same pattern → unified API is best!
- **Biology**: Real neurons have asymmetric input/output AND can grow both dimensions during development

### The Breakthrough Realization

**Initial understanding** (INCORRECT):
- "Regions only need one method because they grow symmetrically"
- "Pathways need two methods because they connect different populations"
- "This asymmetry is justified"

**Corrected understanding** (CORRECT):
- **Both** regions and pathways have `n_input ≠ n_output`
- **Both** need to handle growth from upstream (input) and internal (output)
- **Both** should have symmetric `grow_input()` and `grow_output()` methods
- The current asymmetry is an **incomplete implementation**, not a design feature!

**Impact**: This enables a **beautifully unified growth API** for all neural components.

### Corrected Understanding

**Your understanding is 100% CORRECT and reveals the path forward:**

✅ **Both regions AND pathways need bidirectional growth**:

| Component | Input Dimension Growth | Output Dimension Growth |
|-----------|------------------------|-------------------------|
| **Regions** | ❌ Missing: `grow_input()` | ✅ Has: `add_neurons()` |
| **Pathways** | ✅ Has: `grow_source()` | ✅ Has: `grow_target()` |

**What this means**:
1. The current API is **incomplete**, not fundamentally different
2. Regions need `grow_input()` to handle upstream growth
3. Once added, **both follow the same pattern** → unification is natural!
4. The "asymmetry" was a bug, not a feature

### Proposed Unified API

**Current (inconsistent)**:
```python
# Regions: 1 method (incomplete)
region.add_neurons(n_new)  # Output only

# Pathways: 2 methods (complete)
pathway.grow_target(n_new)  # Output
pathway.grow_source(n_new)  # Input
```

**Proposed (unified)**:
```python
# Both components: 2 methods (complete & consistent)

# Regions:
region.grow_output(n_new)  # Grow neuron population
region.grow_input(n_new)   # Expand input weights

# Pathways:
pathway.grow_output(n_new) # Grow target dimension
pathway.grow_input(n_new)  # Grow source dimension
```

**Benefits**:
- ✅ Unified API: Same interface for all neural components
- ✅ Self-documenting: `grow_input()` vs `grow_output()` is crystal clear
- ✅ Architecturally complete: Both handle bidirectional growth
- ✅ Easy to learn: One pattern works everywhere
- ✅ Future-proof: New components follow same pattern

❌ **Critical Bug Currently**:
**When do we update region weight matrices for bigger input?**
- **Answer**: We DON'T! This is missing!
- Regions currently NEVER expand input dimension
- Will crash if upstream regions grow and pass larger input

**Current state**:
```python
# What works ✅
region.add_neurons(20)  # Expands w [target, source] rows (target dimension)

# What's missing ❌
region.grow_input(20)   # Should expand w [target, source] cols (source dimension)
```

**Why it hasn't crashed yet**:
- Current curriculum only grows internal regions
- Input from sensory encoding is FIXED at initialization
- We've never actually grown upstream regions yet!### Recommendations

**Immediate Actions**:
1. Add `brain.validate_connectivity()` method
2. Document growth patterns in GROWTH_GUIDE.md
3. Add growth support to sensory pathways

**Future Enhancements**:
4. Add warnings for direct region growth
5. Add automatic capacity monitoring
6. Improve growth diagnostics

---

---

## CRITICAL MISSING PIECE: Region Input Dimension Growth

### The Problem You Identified

**YES - You found a real issue!** Regions currently do **NOT** grow their input dimension when pathways grow.

**Current flow**:
```python
# 1. Cortex grows
cortex.add_neurons(20)  # cortex: 128 → 148 neurons

# 2. GrowthCoordinator grows input pathways
visual_to_cortex.grow_target(20)  # weights: [128, 784] → [148, 784]

# 3. ❌ PROBLEM: Cortex internal weights still expect 784 inputs!
cortex.w_input_l4.shape  # Still [51, 784]
# But if visual pathway grew its SOURCE, we'd have 804 inputs!
```

**What SHOULD happen**:
```python
# If visual encoding region grows from 784 → 804:
visual_region.add_neurons(20)

# Output pathways grow
visual_to_cortex.grow_source(20)  # weights: [128, 784] → [128, 804]

# ❌ NOW BROKEN: Cortex receives 804 inputs but w_input_l4 still [51, 784]
# Next forward pass: RuntimeError: mat1 and mat2 shapes mismatch
```

### Why This Doesn't Crash Currently

**Regions currently NEVER grow their input dimension!**

Looking at `LayeredCortex.add_neurons()`:
```python
# 1. Expand input→L4 weights [l4, input]
# Add rows for new L4 neurons
new_input_l4 = new_weights_for(l4_growth, self.layer_config.n_input)  # ← Uses OLD n_input!
self.w_input_l4 = nn.Parameter(
    torch.cat([self.w_input_l4.data, new_input_l4], dim=0)  # ← Only adds ROWS (output dimension)
)
```

**Only the OUTPUT dimension grows, input dimension is FIXED.**

### Missing Implementation: `grow_input()`

Regions **SHOULD** have a method like:

```python
def grow_input(self, n_new: int) -> None:
    """Grow region's input dimension when upstream region grows.

    Called by GrowthCoordinator after growing input pathways.

    Example:
        visual_region grows 784 → 804
        ↓
        visual_to_cortex.grow_source(20)  # pathway: [128, 784] → [128, 804]
        ↓
        cortex.grow_input(20)  # cortex weights: [*, 784] → [*, 804]
    """
    old_n_input = self.config.n_input
    new_n_input = old_n_input + n_new

    # Expand ALL weights that receive external input
    # For LayeredCortex: w_input_l4 [l4, n_input] → [l4, n_input+n_new]
    new_input_cols = WeightInitializer.sparse_random(
        n_output=self.l4_size,
        n_input=n_new,
        sparsity=0.1,
        device=self.device
    )
    self.w_input_l4 = nn.Parameter(
        torch.cat([self.w_input_l4.data, new_input_cols], dim=1)  # Add COLUMNS
    )

    # Update config
    self.config.n_input = new_n_input
```

### Updated GrowthCoordinator Logic

```python
def coordinate_growth(self, region_name: str, n_new_neurons: int):
    # 1. Grow the region itself
    region.add_neurons(n_new_neurons)  # Grows output dimension

    # 2. Grow input pathways (their target dimension)
    for pathway_name, pathway in input_pathways:
        pathway.grow_target(n_new_neurons)

        # ✅ NEW: Tell target region its input grew
        # (No-op for now since we don't grow upstream yet)

    # 3. Grow output pathways (their source dimension)
    for pathway_name, pathway in output_pathways:
        pathway.grow_source(n_new_neurons)

        # ✅ NEW: Tell downstream regions their input grew!
        downstream_region = self._get_pathway_target(pathway_name)
        downstream_region.grow_input(n_new_neurons)
```

### Why This Hasn't Been a Problem

**Current curriculum training only grows within a fixed input size:**
- Input size set at initialization (e.g., 784 for MNIST)
- Only internal neurons grow (cortex, hippocampus, PFC)
- Input from sensory encoding never changes

**But it WILL break if**:
- Sensory encoding regions grow (e.g., add more visual features)
- Input resolution increases (e.g., 28×28 → 32×32 images)
- New input modalities added during training

### Biological Plausibility

**Is this biologically realistic?**

**YES!** Developmental neurogenesis happens in **sensory regions too**:
- **Olfactory bulb**: Continuous addition of granule cells from adult neural stem cells
- **Cortical columns**: Developmental expansion of cortical maps (barrel cortex grows with whisker addition)
- **Retina**: Peripheral vision develops over time in some species

When upstream areas add neurons, downstream areas must adapt their dendritic arbors!

## Appendix: Code Examples

### Example 1: Correct Usage Pattern

```python
from thalia.coordination.growth import GrowthCoordinator

# Create brain
brain = EventDrivenBrain.from_config(config)
coordinator = GrowthCoordinator(brain)

# Train initial capacity
for episode in range(1000):
    brain.train_episode(...)

# Detect saturation
metrics = coordinator.get_capacity_metrics(brain.regions['cortex'])
if metrics.growth_recommended:
    # Coordinated growth (✅ CORRECT)
    events = coordinator.coordinate_growth(
        region_name='cortex',
        n_new_neurons=metrics.growth_amount,
        reason=metrics.growth_reason
    )
    print(f"Grew cortex by {metrics.growth_amount} neurons")
    print(f"Also grew {len(events)-1} connected pathways")
```

### Example 2: Dimension Mismatch (Incorrect)

```python
# ❌ WRONG - Manual growth without coordination
brain.regions['cortex'].add_neurons(100)
# Oops! Pathways still expect old cortex size
# Next forward pass will crash with dimension mismatch
```

### Example 3: Dimension Validation (Proposed)

```python
# ✅ CORRECT - Validate after any manual changes
brain.regions['cortex'].add_neurons(100)
# Manual pathway updates...
for pathway_name in ['visual_to_cortex', 'cortex_to_hippocampus']:
    brain.pathways[pathway_name].grow_appropriately(...)

# Validate everything matches
try:
    brain.validate_connectivity()
    print("✓ All dimensions consistent")
except DimensionMismatchError as e:
    print(f"✗ Dimension mismatch: {e}")
```

---

## References

- `src/thalia/coordination/growth.py` - GrowthCoordinator implementation
- `src/thalia/pathways/spiking_pathway.py` - Pathway growth methods
- `tests/unit/test_bidirectional_growth.py` - Growth test suite
- `tests/unit/test_growth_coordinator.py` - Coordination tests
- ADR-008: Neural Component Consolidation (weight matrix conventions)

---

## Biological Plausibility

### Does `n_input ≠ n_output` Make Sense Biologically?

**YES - This is biologically accurate!** Real neural structures have asymmetric input/output:

**Example 1: Primary Visual Cortex (V1)**
```
Input:  ~1.2 million retinal ganglion cells (optic nerve fibers)
Output: ~140 million neurons in V1
Ratio:  ~120× expansion
```
Why? Dimensionality expansion for feature extraction (edge detectors, orientation columns)

**Example 2: Hippocampus DG → CA3**
```
Input:  ~200k entorhinal cortex neurons
DG:     ~1 million granule cells (5× expansion for pattern separation)
CA3:    ~300k pyramidal neurons (compression for pattern completion)
Output: 300k neurons
```
Why? Pattern separation needs expansion, pattern completion needs compression

**Example 3: Striatum**
```
Input:  ~10 million cortical neurons projecting to striatum
MSNs:   ~2.5 million medium spiny neurons
Ratio:  ~4:1 convergence (compression)
```
Why? Action selection requires convergence (many cortical patterns → few actions)

**Example 4: Cerebellum**
```
Input:  ~200k mossy fibers from cortex/brainstem
Granule cells: ~50 BILLION (250,000× expansion!)
Purkinje cells: ~15 million
Output: ~1 million deep nuclei neurons
```
Why? Massive expansion for fine motor learning, then compression for output

### Why Regions Have `n_input ≠ n_output`

**Computational Reasons**:
1. **Dimensionality expansion**: Extract features (V1, DG)
2. **Dimensionality reduction**: Compress, select, classify (Striatum, CA3)
3. **Transformation**: Map between different representational spaces
4. **Multi-scale processing**: Different spatial/temporal scales

**Biological Constraints**:
1. **Convergence/divergence ratios**: Neurons don't do 1:1 relay (that's for pathways)
2. **Population coding**: Multiple neurons encode single feature
3. **Receptive fields**: One input affects many neurons
4. **Sparse connectivity**: Not all inputs connect to all neurons

### Why Only Regions Grow Output (Not Input)

**Biological Principle**: Neurogenesis occurs WITHIN a region, not in its inputs

```
Adult Neurogenesis Sites:
✅ Hippocampal DG: Adds new granule cells (output neurons)
✅ Olfactory bulb: Adds new granule cells (output neurons)
✅ Striatum: Limited MSN neurogenesis (output neurons)
✅ Cortex: Gliogenesis, limited neurogenesis (output neurons)

❌ Afferent pathways: Do NOT spontaneously grow new input fibers
```

**When input dimension grows**:
- Source region undergoes neurogenesis → adds output neurons
- All pathways FROM that region must grow their source dimension
- Target regions receive more input but don't add input neurons

**Example**:
```python
# Hippocampus DG adds 100 new granule cells (neurogenesis)
hippocampus.dg.add_neurons(100)  # Internal growth

# DG → CA3 pathway must adapt
dg_to_ca3_pathway.grow_source(100)  # Pathway adapts to source growth

# CA3 receives more input but doesn't add "input neurons"
# CA3 neurons now receive from larger DG population
# If CA3 needs more neurons, it does its own add_neurons()
```

**Conclusion**: The current design is **biologically accurate**:
- Regions grow their own neuron population (`add_neurons`)
- Pathways adapt to growth at both ends (`grow_source`/`grow_target`)
- Input size is determined by source, not by target region
