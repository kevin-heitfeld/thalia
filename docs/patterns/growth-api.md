# Growth API Reference

**Standard interface for dynamic neural region expansion in Thalia.**

All `NeuralRegion` subclasses implement the Growth API via `GrowthMixin`, providing a unified interface for curriculum-based developmental training. This document defines the contracts, behavior guarantees, and implementation patterns.

---

## Core Concepts

### What is Growth?

Growth allows brain regions to add capacity during training without discarding learned knowledge. This mirrors biological development where:
- **Neurogenesis**: New neurons are born and integrate into circuits
- **Synaptogenesis**: New connections form between existing neurons
- **Structural plasticity**: Dendrites and axons grow/retract

### Two Growth Operations

1. **`grow_output(n_new)`**: Add neurons (grow the region itself)
2. **`grow_source(source_name, new_size)`**: Expand specific input source

**Note**: Replaced `grow_input()` with `grow_source()` to support multi-source architecture where each input source has a separate weight matrix.

---

## Method Contracts

### `grow_output(n_new: int) -> None`

**Purpose**: Add neurons to the region, expanding its representational capacity.

**Effects**:
- ✅ Expands weight matrices (adds **rows**)
- ✅ Adds neurons via `neurons.grow_neurons(n_new)`
- ✅ Updates `config.n_output`
- ✅ Preserves **all** existing weights (no forgetting)
- ✅ New neurons initialized with biological strategies (WeightInitializer)

**Contract**:
```python
def grow_output(self, n_new: int) -> None:
    """Add n_new output neurons to region.

    Guarantees:
    - Old weights unchanged
    - New weights biologically initialized
    - Neuron dynamics preserve state
    - No breaking changes to existing circuits
    """
```

**Example**:
```python
# Before: 100 neurons
cortex = LayeredCortex(config)  # n_output=100

# Grow: Add 50 neurons
cortex.grow_output(50)

# After: 150 neurons
assert cortex.n_neurons == 150
assert cortex.config.n_output == 150
assert cortex.weights.shape[0] == 150  # Rows expanded
```

**Implementation Pattern**:
```python
def grow_output(self, n_new: int) -> None:
    """Standard implementation pattern."""
    # 1. Grow weight matrices (add rows)
    self.weights = self._expand_weights(
        self.weights,
        n_output=self.n_neurons + n_new,
        n_input=self.weights.shape[1],
        expansion_type="output",
    )

    # 2. Grow neuron population
    self.neurons.grow_neurons(n_new)

    # 3. Update config
    self.config.n_output += n_new
    self.n_neurons += n_new

    # 4. Grow region-specific structures
    # E.g., learning traces, neuromodulator receptors, etc.
```

---

### `grow_source(source_name: str, new_size: int) -> None`

**Purpose**: Expand capacity for specific input source in multi-source architecture.

**Replaces**: `grow_input(n_new)` - removed in v3.0+ in favor of per-source growth

**Applies to**: All regions with multi-source inputs (standard architecture)

**Effects**:
- ✅ Resizes weight matrix for specified source only
- ✅ Updates `input_sizes[source_name]`
- ✅ Preserves other sources completely untouched
- ❌ **Does NOT** add neurons

**Contract**:
```python
def grow_source(self, source_name: str, new_size: int) -> None:
    """Expand specific input source to new_size.

    Args:
        source_name: Input source identifier (e.g., 'thalamus', 'cortex:l5')
        new_size: New total size for this source

    Guarantees:
    - Other sources unchanged
    - Source-specific weights preserved and expanded
    - Total input dimension updated
    """
```

**Example**:
```python
# Before: Thalamus=100, Hippocampus=200
cortex = LayeredCortex(config)
cortex.input_sizes = {'thalamus': 100, 'hippocampus': 200}

# Grow: Expand thalamus source to 150
cortex.grow_source('thalamus', 150)

# After: Thalamus=150, Hippocampus=200 (unchanged)
assert cortex.input_sizes['thalamus'] == 150
assert cortex.input_sizes['hippocampus'] == 200  # Untouched!
assert cortex.synaptic_weights['thalamus'].shape == (n_neurons, 150)
assert cortex.synaptic_weights['hippocampus'].shape == (n_neurons, 200)  # Same!
```

**Implementation Pattern**:
```python
def grow_source(self, source_name: str, new_size: int) -> None:
    """Standard implementation from NeuralRegion base class."""
    if source_name not in self.synaptic_weights:
        raise ValueError(f"Unknown source: {source_name}")

    old_size = self.input_sizes[source_name]
    n_new = new_size - old_size

    # 1. Grow weight matrix for THIS SOURCE ONLY
    old_weights = self.synaptic_weights[source_name]  # [n_output, old_size]
    new_weights = self._expand_weights(
        old_weights,
        n_output=self.n_neurons,
        n_input=new_size,  # Expand columns
        expansion_type="input",
    )
    self.synaptic_weights[source_name] = new_weights  # [n_output, new_size]

    # 2. Update input size tracking
    self.input_sizes[source_name] = new_size
    self.n_input = sum(self.input_sizes.values())

    # 3. Update config
    self.config.n_input = self.n_input
```

**Multi-Source Architecture**:
- Each input source has **separate weight matrix** in `synaptic_weights` dict
- GrowthManager coordinates cascading growth:
  1. Source region grows output → calls `projection.grow_source(source_name, new_size)`
  2. Projection updates routing → calls `target_region.grow_source(source_name, new_size)`
- AxonalProjection also has `grow_source()` but NO `grow_output()` (pure routing)

---

## Weight Initialization

**All new weights MUST use `WeightInitializer` registry**:

```python
from thalia.components.synapses import WeightInitializer

# ✅ CORRECT: Use centralized initializer
new_weights = self._create_new_weights(
    n_output=100,
    n_input=50,
    initialization='xavier',
    sparsity=0.2,
)

# ❌ WRONG: Direct tensor creation
new_weights = torch.randn(100, 50, device=self.device) * 0.1
```

**Available strategies** (see `WeightInitializer` for full list):
- `xavier`: He initialization for ReLU-like activations
- `sparse_random`: Random with sparsity (fraction TO KEEP)
- `topographic`: Distance-based connectivity
- `gaussian`: Normal distribution with mean/std
- `uniform`: Uniform distribution in [low, high]

---

## Common Patterns

### Pattern 1: Symmetric Growth (Input + Output)

```python
def grow_region_symmetrically(region, n_new):
    """Grow both input and output dimensions."""
    region.grow_source('upstream_region', new_size)  # Accept more from specific source
    region.grow_output(n_new)  # Add more neurons
```

**Use Case**: Balanced expansion (e.g., sensory cortex growing with thalamus)

**Note**: Use `grow_source()` instead of `grow_input()` for per-source expansion

---

### Pattern 2: Layer-Specific Growth (Cortex)

```python
def grow_cortex_layer(cortex, layer_name, n_new):
    """Grow specific cortical layer."""
    if layer_name == 'l4':
        cortex.l4_neurons.grow_neurons(n_new)
        cortex.grow_source('thalamus', new_size)
    elif layer_name == 'l23':
        cortex.l23_neurons.grow_neurons(n_new)
        cortex.grow_source('l4', new_size)
    # ... etc
```

**Use Case**: Targeted layer expansion during curriculum stages

---

### Pattern 3: Pathway-Aware Growth (Striatum)

```python
def grow_striatum_pathways(striatum, n_new):
    """Grow D1/D2 pathways together."""
    striatum.d1_pathway.grow_output(n_new)  # Add D1 MSNs
    striatum.d2_pathway.grow_output(n_new)  # Add D2 MSNs

    # Update opponent inhibition
    striatum._update_d1d2_inhibition()
```

**Use Case**: Maintain balance in opponent pathways

---

### Pattern 4: Multi-Stage Growth

```python
# Stage 0: Small network (100 neurons)
brain = DynamicBrain(base_config)

# Stage 1: 2x growth (200 neurons)
for region in brain.components.values():
    region.grow_output(100)

# Stage 2: 4x growth (400 neurons)
for region in brain.components.values():
    region.grow_output(200)
```

**Use Case**: Curriculum-based training with progressive complexity

---

## Testing Growth

**Every region MUST test growth operations**:

```python
def test_grow_output():
    """Verify grow_output() preserves weights and adds neurons."""
    region = MyRegion(config)
    old_weights = region.weights.clone()
    old_n_neurons = region.n_neurons

    region.grow_output(50)

    # Check neurons added
    assert region.n_neurons == old_n_neurons + 50

    # Check old weights preserved
    assert torch.allclose(
        region.weights[:old_n_neurons, :],
        old_weights
    )

    # Check new weights initialized
    assert region.weights[old_n_neurons:, :].abs().sum() > 0

def test_grow_source():
    """Verify grow_source() preserves weights for specific source, no neurons added."""
    region = MyRegion(config)
    source_name = 'thalamus'
    old_weights = region.synaptic_weights[source_name].clone()
    old_n_neurons = region.n_neurons
    old_size = region.input_sizes[source_name]

    region.grow_source(source_name, old_size + 30)

    # Check NO new neurons
    assert region.n_neurons == old_n_neurons

    # Check old weights preserved
    assert torch.allclose(
        region.synaptic_weights[source_name][:, :old_size],
        old_weights
    )

    # Check new columns initialized
    assert region.synaptic_weights[source_name][:, old_size:].abs().sum() > 0

    # Check other sources unchanged
    for other_source in region.synaptic_weights:
        if other_source != source_name:
            assert region.synaptic_weights[other_source].shape == original_shapes[other_source]
```

---

## Troubleshooting

### Issue: "Weights not preserved after growth"

**Cause**: Using wrong `expansion_type` or incorrect slicing

**Fix**:
```python
# ✅ CORRECT: Specify expansion direction
self._expand_weights(weights, n_output, n_input, expansion_type="output")

# ❌ WRONG: Ambiguous expansion
weights = torch.cat([weights, new_weights], dim=0)  # Which dimension?
```

---

### Issue: "Device mismatch after growth"

**Cause**: New weights created on CPU instead of region's device

**Fix**:
```python
# ✅ CORRECT: Use _create_new_weights() (handles device)
new_weights = self._create_new_weights(n_output, n_input, ...)

# ❌ WRONG: Manual creation without device
new_weights = torch.randn(n_output, n_input)  # Defaults to CPU!
```

---

### Issue: "Learning traces not resized"

**Cause**: Forgot to grow region-specific structures

**Fix**:
```python
def grow_output(self, n_new):
    # ... grow weights and neurons ...

    # ✅ Don't forget region-specific structures!
    if hasattr(self, 'eligibility'):
        self.eligibility = self._expand_weights(
            self.eligibility, self.n_neurons, self.n_input, "output"
        )
```

---

## See Also

- **Implementation**: `src/thalia/mixins/growth_mixin.py` - GrowthMixin base implementation
- **Examples**: `src/thalia/regions/*/` - Region-specific growth methods
- **Testing**: `tests/unit/test_growth_mixin.py` - Growth utility tests
- **Constants**: `src/thalia/regulation/region_architecture_constants.py` - Growth scales
- **Curriculum**: `docs/CURRICULUM_QUICK_REFERENCE.md` - Stage-based training with growth
