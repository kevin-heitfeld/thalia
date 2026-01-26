# ADR-011: Large Region Files Justified by Biological Circuit Integrity

## Status

**ACCEPTED** - December 12, 2025

## Context

The architecture review (2025-12-12) identified large region files as targets for splitting:
- `hippocampus/trisynaptic.py`: ~1862 lines
- `cortex/layered_cortex.py`: ~1294 lines

The review recommended splitting these into submodules following the Striatum pattern (learning_component.py, exploration_component.py, etc.).

However, detailed code analysis reveals that these files are fundamentally different from Striatum in structure and purpose.

## Decision

**We will NOT split the main circuit implementation files** (`trisynaptic.py`, `layered_cortex.py`) into submodules.

The large file sizes are **justified by biological circuit integrity** and **not an antipattern**.

### Rationale

#### 1. Biological Circuit Cohesion

**Hippocampus trisynaptic.py forward() method (~700 lines)**:
- Implements the DG→CA3→CA2→CA1 circuit as a single narrative flow
- Each stage depends on the previous stage's output within the same timestep
- Theta modulation coordinates all three stages simultaneously
- Splitting would require artificial boundaries that don't exist biologically

```python
# The circuit flow is a single cohesive computation:
DG spikes → CA3 recurrence (theta-gated) → CA1 coincidence detection

# Each line depends on previous computations:
ca3_from_dg = matmul(w_dg_ca3, dg_spikes) * encoding_mod  # Theta-modulated
ca3_rec = matmul(w_ca3_ca3, ca3_spikes) * retrieval_mod   # Opposite gating!
ca1_coincidence = (ca3_to_ca1 * nmda_gating) + ca1_from_ec  # Match/mismatch
```

**LayeredCortex forward() method (~400 lines)**:
- Implements L4→L2/3→L5 canonical microcircuit
- L2/3 recurrence interdepends with L4 feedforward and L5 output
- Alpha/theta/gamma oscillator coupling affects all layers simultaneously
- Layer boundaries are biological, but the computation is interleaved

#### 2. Already Has Component Managers

Both regions ALREADY extract orthogonal concerns into components:

**Hippocampus components**:
- `learning_component.py`: Hebbian plasticity, synaptic scaling
- `memory_component.py`: Episode storage/retrieval
- `replay_engine.py`: Sequence replay logic
- `hindsight_relabeling.py`: HER integration

**What remains in trisynaptic.py**: The core circuit (DG/CA3/CA2/CA1 dynamics) - this IS the hippocampus!

**Cortex components**:
- Uses mixins: `LearningStrategyMixin`, `NeuromodulatorMixin`, `DiagnosticsMixin`
- Uses helpers: `LayerEIBalance`, `UnifiedHomeostasis`, `FeedforwardInhibition`

**What remains in layered_cortex.py**: The core circuit (L4/L2/3/L5 dynamics) - this IS the cortex!

#### 3. Striatum Pattern Doesn't Apply

**Why Striatum CAN be split**:
- D1 and D2 pathways are **physically separate** brain structures
- They compute **in parallel** (no sequential dependency)
- Learning component orchestrates both pathways **after** forward passes
- **Result**: Natural boundaries for extraction

**Why Hippocampus/Cortex CANNOT be split the same way**:
- DG→CA3→CA1 is a **sequential pipeline within a single timestep**
- L4→L2/3→L5 is a **cascading computation with feedback loops**
- Splitting would require passing 20+ intermediate tensors between components
- **Result**: Artificial complexity, no benefit

#### 4. Line Count Is Misleading

**Hippocampus trisynaptic.py breakdown**:
- Imports/docstrings: ~120 lines
- `__init__` (circuit initialization): ~200 lines (cannot split - establishes connections)
- `forward()` (biological circuit): ~700 lines (cannot split - cohesive narrative)
- Component managers: Already extracted!
- Utilities (`reset_state`, `grow_source`, `grow_output`, etc.): ~300 lines
- Properties/diagnostics: ~400 lines
- **Total**: ~1862 lines

**The forward() method is 700 lines because**:
- Theta modulation: ~50 lines (continuous encoding/retrieval gating)
- DG processing: ~80 lines (pattern separation with sparsity)
- CA3 processing: ~300 lines (recurrence, bistability, STP, gamma slots)
- CA1 processing: ~200 lines (coincidence detection, NMDA gating, EC plasticity)
- Trace updates: ~70 lines (eligibility traces for learning)

Each section is **irreducible** - removing ANY line breaks biological plausibility.

## Implications

### What We WILL Do

1. **Improve Documentation**:
   - Add section headers within forward() methods
   - Document biological rationale for each computation block
   - Link to relevant neuroscience papers

2. **Extract Pure Utilities** (if any):
   - Reusable math functions that don't depend on circuit state
   - Generic spike processing helpers (already done via mixins)

3. **Enhance Navigation**:
   - VSCode region markers (`# region DG Processing`)
   - Table of contents in file docstrings
   - Cross-references to related components

### What We WILL NOT Do

1. **Split forward() methods** into artificial components
2. **Extract circuit stages** (DG/CA3/CA2/CA1 or L4/L2/3/L5) into separate files
3. **Force component boundaries** that don't exist biologically

## Comparison: When to Split vs. When to Keep Cohesive

| Structure | Can Split? | Reason |
|-----------|-----------|--------|
| Striatum D1/D2 pathways | ✅ YES | Parallel, independent computation |
| Learning components | ✅ YES | Orthogonal concern, called after forward() |
| Hippocampus DG→CA3→CA1 | ❌ NO | Sequential pipeline, single timestep |
| Cortex L4→L2/3→L5 | ❌ NO | Cascading with feedback, interleaved |
| Replay engine | ✅ YES | Separate use case (consolidation), already done |
| Memory buffer | ✅ YES | Storage concern, orthogonal, already done |

## Alternative Considered and Rejected

**Alternative**: Split hippocampus into `dg_processor.py`, `ca3_processor.py`, `ca1_processor.py`

**Rejected because**:
- Each processor would need 15+ parameters from the main class
- CA3 processor would need to call DG processor output
- CA1 processor would need both CA3 and EC outputs
- Theta modulation affects all three - requires global coordination
- **Result**: More lines of code, worse readability, no benefit

**Code comparison**:

```python
# Current (cohesive):
def forward(input_spikes):
    encoding_mod = compute_theta_modulation()
    dg_spikes = process_dg(input_spikes, encoding_mod)
    ca3_spikes = process_ca3(dg_spikes, encoding_mod)  # Uses encoding_mod!
    ca1_spikes = process_ca1(ca3_spikes, input_spikes, encoding_mod)
    return ca1_spikes

# Split (artificial):
def forward(input_spikes):
    encoding_mod = compute_theta_modulation()
    dg_spikes = self.dg_processor.process(input_spikes, encoding_mod, self.w_ec_dg, self.dg_neurons, self.tri_config, ...)
    ca3_spikes = self.ca3_processor.process(dg_spikes, input_spikes, encoding_mod, self.w_dg_ca3, self.w_ec_ca3, self.w_ca3_ca3, self.ca3_neurons, self.state.ca3_persistent, self._ca3_activity_trace, self.tri_config, ...)
    ca1_spikes = self.ca1_processor.process(ca3_spikes, input_spikes, encoding_mod, self.w_ca3_ca1, self.w_ec_ca1, self.ca1_neurons, self.state.nmda_trace, self.tri_config, ...)
    return ca1_spikes
```

**Verdict**: Split version is WORSE (more parameters, less clear, no encapsulation benefit).

## Metrics

**Before attempting split**:
- `trisynaptic.py`: 1862 lines
- Component managers: Already extracted (3 files, ~800 lines)
- Forward() method: 700 lines (irreducible biological circuit)

**After this ADR**:
- No code changes needed
- Documentation improvements only
- Clarity increased by NOT forcing artificial splits

## Related Decisions

- **ADR-008**: Neural component consolidation (component managers ARE the split)
- **ADR-009**: Pathway-neuron consistency (pathways also large, justified)
- **Component Parity Principle**: Regions and pathways have same methods (forward() is fundamental)

## References

- Architecture Review 2025-12-12 (T2.4 recommendation)
- Copilot Instructions: "Maintain biological plausibility" (line 40)
- Striatum implementation: Successful component pattern for parallel structures
- Hippocampus trisynaptic circuit: Marr (1971), Treves & Rolls (1994)
- Cortical microcircuit: Douglas & Martin (2004), canonical circuit pattern

## Conclusion

**Large files are not inherently bad when they represent cohesive biological circuits.**

The Striatum pattern works because D1/D2 pathways are physically separate and compute in parallel. Hippocampus and Cortex do not have this property - their stages are sequential and interdependent within single timesteps.

**The correct abstraction level is**: Extract orthogonal concerns (learning, memory, replay) but keep the circuit cohesive.

**Verdict**: Current structure is appropriate. No refactoring needed. Documentation improvements sufficient.
