# Size Specification Refactoring Plan

**Date**: January 10, 2026
**Status**: 🔵 PROPOSAL - Ready for Review
**Breaking Changes**: Yes - Acceptable (no external users)
**Priority**: HIGH - Blocking test fixes and causing architectural confusion

---

## Executive Summary

We have multiple inconsistent ways to specify and calculate sizes throughout the codebase. This creates confusion, bugs, and makes the system hard to reason about. This plan proposes a **single source of truth** approach with clear separation between:

1. **User-facing specification** (what you want to build)
2. **Derived computation** (how sizes are calculated)
3. **Runtime properties** (what actually exists)

---

## Current Problems

### 1. Inconsistent Size Calculation Methods

**Problem**: Two different functions calculate cortex layer sizes differently:

```python
# Method 1: region_sizes.py::compute_cortex_layer_sizes()
# Based on INPUT size + expansion ratios
def compute_cortex_layer_sizes(input_size: int):
    l4_size = int(input_size * 1.5)  # L4_TO_INPUT_RATIO = 1.5
    l23_size = int(l4_size * 2.0)    # L23_TO_L4_RATIO = 2.0
    l5_size = int(l23_size * 0.5)    # L5_TO_L23_RATIO = 0.5
    return {"l4_size": l4_size, "l23_size": l23_size, "l5_size": l5_size}

# Method 2: cortex/config.py::calculate_layer_sizes()
# Based on OUTPUT size (n_output) + different ratios
def calculate_layer_sizes(n_output: int):
    l4_size = int(n_output * 1.0)    # l4_ratio = 1.0
    l23_size = int(n_output * 1.5)   # l23_ratio = 1.5
    l5_size = int(n_output * 1.0)    # l5_ratio = 1.0
    l6a_size = int(n_output * 0.3)   # l6a_ratio = 0.3
    l6b_size = int(n_output * 0.2)   # l6b_ratio = 0.2
    return {...}
```

**Impact**:
- Method 1 doesn't compute L6a/L6b
- Different ratios produce different layer sizes for same base value
- Unclear which method to use when
- Tests use both methods inconsistently

### 2. Ambiguous "Base Size" Parameter

**Problem**: What does `cortex_size=128` mean?

```python
# In RegionSizes
cortex_size: int = 128
"""Output size of cortex. L2/3 and L5 layers will be sized relative to this."""

# But output_size is actually a computed property:
@property
def output_size(self) -> int:
    """L2/3 + L5 (dual output pathways)."""
    return self.l23_size + self.l5_size
```

**Confusion**: Is `cortex_size` the:
- Desired output size? (Then l23_size + l5_size should equal cortex_size)
- Base for ratio calculations? (Then cortex_size is NOT the output size)
- Total neuron count? (Then it should equal sum of all layers)

**Current reality**: It's used as a base multiplier, not a target. With default ratios (1.0:1.5:1.0:0.3:0.2), `cortex_size=128` produces `output_size=128*1.5 + 128*1.0 = 320`, not 128!

### 3. Input Size Inference Chicken-and-Egg

**Problem**: BrainBuilder needs to know output sizes to infer input sizes, but cortex needs to know input size to compute layer sizes.

```python
# Current pattern in from_thalia_config:
cortex_config = LayeredCortexConfig(
    l4_size=128,   # Hardcoded, doesn't account for actual inputs
    l23_size=192,
    l5_size=128,
    # ... but cortex will receive from thalamus AND hippocampus
)

# Later, builder infers:
cortex_input_size = thalamus_output + hippocampus_output  # 64 + 128 = 192

# But L4 was sized for 128, not 192!
# Result: Weight matrix mismatch
```

**Impact**: Tests fail with "input shape [192] but input_size=128" errors.

### 4. Multi-Source Pathway Confusion

**Problem**: Regions receive inputs from multiple sources, but config has single `input_size`.

```python
# Cortex receives from:
# - Thalamus (64 neurons)
# - Hippocampus CA1 (128 neurons)
# Total input: 192

# But LayeredCortexConfig has:
input_size: int = 128  # Which source? How to know total?
```

**Current solution**: Builder infers total input from connections, but then layer sizes already computed incorrectly.

### 5. Growth API Confusion

**Problem**: What do `grow_input(n)` and `grow_output(n)` mean for multi-layer regions?

```python
# Striatum has D1 and D2 pathways
striatum.grow_output(10)  # Add 10 what? Actions? D1 neurons? Total neurons?

# Cortex has 5 layers
cortex.grow_output(20)  # Add 20 to which layers? L2/3? L5? Both?
```

**Current solution**: Document "output" means specific layers (L2/3+L5 for cortex, D1+D2 for striatum), but still confusing.

---

## Root Causes

1. **Historical evolution**: Started with simple feedforward regions (n_input → n_output), added complexity (multi-source, multi-layer) without redesigning size API
2. **Multiple concerns mixed**: User intent (desired scale) + biological ratios + runtime reality all in same config
3. **Implicit dependencies**: Builder infers sizes but regions need sizes before building
4. **Inconsistent terminology**: "size", "n_input", "n_output", "n_neurons", "input_size", "output_size" used interchangeably

---

## Design Principles

### Principle 1: Explicit Over Implicit
**Prefer**: User specifies all layer sizes explicitly
**Avoid**: Auto-compute sizes from ratios unless explicitly requested

**Rationale**: Debugging and testing require knowing exact sizes. Magic calculations hide bugs.

### Principle 2: Separate Specification from Computation
**Separate**:
- What user wants (intent)
- How to compute it (ratios/formulas)
- What exists (runtime reality)

**Rationale**: Different use cases need different patterns. Keep them independent.

### Principle 3: Clear Defaults for Common Cases
**Provide**: High-level factory methods for standard architectures
**Support**: Low-level explicit control for custom configurations

**Rationale**: Both beginners and experts should be happy.

### Principle 4: No Ambiguous "Base" Parameters
**Replace**: `cortex_size` (ambiguous)
**With**: `target_output_size` or `scale_factor` (clear intent)

**Rationale**: Names should reveal purpose.

### Principle 5: Multi-Source is the Norm
**Design**: Assume regions receive from multiple sources
**Handle**: Single-source as special case of multi-source

**Rationale**: Most brain regions have convergent inputs.

---

## Proposed Architecture

### Phase 1: Unified Size Calculation (HIGHEST PRIORITY)

**Goal**: One canonical way to compute layer sizes with biological ratios.

#### 1.1 Single Source of Truth: `LayerSizeCalculator` Class

Create a single, well-documented calculator that handles ALL region types:

```python
# src/thalia/config/size_calculator.py

from dataclasses import dataclass
from typing import Dict, Literal

@dataclass
class BiologicalRatios:
    """Biologically-motivated ratios from neuroscience literature.

    All ratios documented with references.
    """
    # Hippocampus (Amaral & Witter, 1989)
    dg_to_ec: float = 4.0        # Pattern separation expansion
    ca3_to_dg: float = 0.5       # Pattern completion compression
    ca2_to_dg: float = 0.25      # Social memory hub
    ca1_to_ca3: float = 1.0      # Output comparison

    # Cortex (Douglas & Martin, 2004)
    # L4 → L2/3 → L5 → L6
    l4_to_input: float = 1.5     # Input expansion
    l23_to_l4: float = 2.0       # Processing expansion
    l5_to_l23: float = 0.5       # Output compression
    l6a_to_l23: float = 0.2      # Feedback to TRN
    l6b_to_l23: float = 0.13     # Feedback to relay

    # Striatum
    msn_to_cortex: float = 0.5   # Dimensionality reduction
    d1_to_total: float = 0.5     # D1/D2 balance

    # Cerebellum
    granule_to_purkinje: float = 4.0  # Expansion coding

    # Thalamus
    trn_to_relay: float = 0.3    # Inhibitory modulation


class LayerSizeCalculator:
    """Single source of truth for layer size calculations.

    All calculations based on documented biological ratios.
    Supports multiple specification patterns.
    """

    def __init__(self, ratios: BiologicalRatios = None):
        self.ratios = ratios or BiologicalRatios()

    # =========================================================================
    # CORTEX
    # =========================================================================

    def cortex_from_input(self, input_size: int) -> Dict[str, int]:
        """Calculate cortex layer sizes from INPUT size.

        Pattern: Input → L4 → L2/3 → L5 → L6a/L6b
        Use when you know what's connecting TO cortex.

        Args:
            input_size: Total input from all sources (thalamus + other cortex + hippocampus)

        Returns:
            {"l4_size": ..., "l23_size": ..., "l5_size": ..., "l6a_size": ..., "l6b_size": ...}
        """
        l4_size = int(input_size * self.ratios.l4_to_input)
        l23_size = int(l4_size * self.ratios.l23_to_l4)
        l5_size = int(l23_size * self.ratios.l5_to_l23)
        l6a_size = int(l23_size * self.ratios.l6a_to_l23)
        l6b_size = int(l23_size * self.ratios.l6b_to_l23)

        return {
            "l4_size": l4_size,
            "l23_size": l23_size,
            "l5_size": l5_size,
            "l6a_size": l6a_size,
            "l6b_size": l6b_size,
            "input_size": input_size,
            "output_size": l23_size + l5_size,  # Dual output pathways
            "total_neurons": l4_size + l23_size + l5_size + l6a_size + l6b_size,
        }

    def cortex_from_output(self, target_output_size: int) -> Dict[str, int]:
        """Calculate cortex layer sizes from target OUTPUT size.

        Pattern: Work backwards from desired output.
        Use when you know what you want cortex to OUTPUT.

        Args:
            target_output_size: Desired L2/3 + L5 output size

        Returns:
            Layer sizes that produce approximately target_output_size

        Note:
            Output = L2/3 + L5. With default ratios, L2/3 is ~67% and L5 is ~33%.
            So L2/3 ≈ target * 0.67, L5 ≈ target * 0.33
        """
        # With L2/3:L5 = 2:1, we have L2/3 + L5 = target
        # So L2/3 = target * 2/3, L5 = target * 1/3
        l23_size = int(target_output_size * 2 / 3)
        l5_size = int(target_output_size * 1 / 3)

        # Work backwards to L4 (L2/3 = L4 * 2.0, so L4 = L2/3 / 2.0)
        l4_size = int(l23_size / self.ratios.l23_to_l4)

        # Compute L6 from L2/3
        l6a_size = int(l23_size * self.ratios.l6a_to_l23)
        l6b_size = int(l23_size * self.ratios.l6b_to_l23)

        # Infer input size from L4
        input_size = int(l4_size / self.ratios.l4_to_input)

        return {
            "l4_size": l4_size,
            "l23_size": l23_size,
            "l5_size": l5_size,
            "l6a_size": l6a_size,
            "l6b_size": l6b_size,
            "input_size": input_size,
            "output_size": l23_size + l5_size,
            "total_neurons": l4_size + l23_size + l5_size + l6a_size + l6b_size,
        }

    def cortex_from_scale(self, scale_factor: int) -> Dict[str, int]:
        """Calculate cortex layer sizes from SCALE factor.

        Pattern: Scale all layers proportionally.
        Use when you want "small/medium/large" cortex without caring about specifics.

        Args:
            scale_factor: Base multiplier (e.g., 32, 64, 128, 256)

        Returns:
            Proportionally scaled layers
        """
        # Use scale_factor as L4 base, apply standard ratios
        l4_size = scale_factor
        l23_size = int(scale_factor * self.ratios.l23_to_l4)
        l5_size = int(l23_size * self.ratios.l5_to_l23)
        l6a_size = int(l23_size * self.ratios.l6a_to_l23)
        l6b_size = int(l23_size * self.ratios.l6b_to_l23)

        input_size = int(l4_size / self.ratios.l4_to_input)

        return {
            "l4_size": l4_size,
            "l23_size": l23_size,
            "l5_size": l5_size,
            "l6a_size": l6a_size,
            "l6b_size": l6b_size,
            "input_size": input_size,
            "output_size": l23_size + l5_size,
            "total_neurons": l4_size + l23_size + l5_size + l6a_size + l6b_size,
        }

    # =========================================================================
    # HIPPOCAMPUS
    # =========================================================================

    def hippocampus_from_input(self, ec_input_size: int) -> Dict[str, int]:
        """Calculate hippocampus layer sizes from entorhinal cortex input.

        Pattern: EC → DG → CA3 → CA2 → CA1

        Args:
            ec_input_size: Size of entorhinal cortex input

        Returns:
            {"dg_size": ..., "ca3_size": ..., "ca2_size": ..., "ca1_size": ...}
        """
        dg_size = int(ec_input_size * self.ratios.dg_to_ec)
        ca3_size = int(dg_size * self.ratios.ca3_to_dg)
        ca2_size = int(dg_size * self.ratios.ca2_to_dg)
        ca1_size = int(ca3_size * self.ratios.ca1_to_ca3)

        return {
            "dg_size": dg_size,
            "ca3_size": ca3_size,
            "ca2_size": ca2_size,
            "ca1_size": ca1_size,
            "input_size": ec_input_size,
            "output_size": ca1_size,
            "total_neurons": dg_size + ca3_size + ca2_size + ca1_size,
        }

    # =========================================================================
    # STRIATUM
    # =========================================================================

    def striatum_from_actions(
        self,
        n_actions: int,
        neurons_per_action: int = 10
    ) -> Dict[str, int]:
        """Calculate striatum sizes from number of actions.

        Pattern: Population coding with D1/D2 opponent pathways.

        Args:
            n_actions: Number of discrete actions
            neurons_per_action: Neurons per action (default: 10 for noise reduction)

        Returns:
            {"d1_size": ..., "d2_size": ..., "n_actions": ..., "neurons_per_action": ...}
        """
        if neurons_per_action == 1:
            d1_neurons_per_action = 1
            d2_neurons_per_action = 1
        else:
            # Split neurons between D1 (Go) and D2 (NoGo)
            d1_neurons_per_action = max(1, int(neurons_per_action * self.ratios.d1_to_total))
            d2_neurons_per_action = max(1, neurons_per_action - d1_neurons_per_action)

        d1_size = n_actions * d1_neurons_per_action
        d2_size = n_actions * d2_neurons_per_action

        return {
            "d1_size": d1_size,
            "d2_size": d2_size,
            "n_actions": n_actions,
            "neurons_per_action": neurons_per_action,
            "output_size": d1_size + d2_size,  # Both pathways output spikes
            "total_neurons": d1_size + d2_size,
        }

    # =========================================================================
    # CEREBELLUM
    # =========================================================================

    def cerebellum_from_output(self, purkinje_size: int) -> Dict[str, int]:
        """Calculate cerebellum sizes from Purkinje cell count.

        Args:
            purkinje_size: Number of Purkinje cells (= output size)

        Returns:
            {"purkinje_size": ..., "granule_size": ..., ...}
        """
        granule_size = int(purkinje_size * self.ratios.granule_to_purkinje)
        basket_size = int(purkinje_size * 0.1)  # ~10% basket cells
        stellate_size = int(purkinje_size * 0.05)  # ~5% stellate cells

        return {
            "purkinje_size": purkinje_size,
            "granule_size": granule_size,
            "basket_size": basket_size,
            "stellate_size": stellate_size,
            "output_size": purkinje_size,
            "total_neurons": purkinje_size + granule_size + basket_size + stellate_size,
        }

    # =========================================================================
    # THALAMUS
    # =========================================================================

    def thalamus_from_relay(self, relay_size: int) -> Dict[str, int]:
        """Calculate thalamus sizes from relay neuron count.

        Args:
            relay_size: Number of relay neurons (= output size)

        Returns:
            {"relay_size": ..., "trn_size": ...}
        """
        trn_size = int(relay_size * self.ratios.trn_to_relay)

        return {
            "relay_size": relay_size,
            "trn_size": trn_size,
            "output_size": relay_size,
            "total_neurons": relay_size + trn_size,
        }
```

#### 1.2 Deprecate Old Functions

Mark old functions as deprecated with clear migration path:

```python
# src/thalia/config/region_sizes.py

import warnings
from thalia.config.size_calculator import LayerSizeCalculator

_calculator = LayerSizeCalculator()

def compute_cortex_layer_sizes(input_size: int) -> dict:
    """DEPRECATED: Use LayerSizeCalculator.cortex_from_input() instead.

    This function will be removed in v0.3.0.
    """
    warnings.warn(
        "compute_cortex_layer_sizes is deprecated. "
        "Use LayerSizeCalculator().cortex_from_input() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _calculator.cortex_from_input(input_size)

# Similar for all other compute_* functions
```

#### 1.3 Remove Duplicate Functions

Delete `calculate_layer_sizes()` from `src/thalia/regions/cortex/config.py` - it duplicates functionality.

---

### Phase 2: Clear Config API (HIGH PRIORITY)

**Goal**: Make configs self-documenting with clear semantics.

#### 2.1 New Config Pattern: Required vs Optional

```python
# src/thalia/regions/cortex/config.py

@dataclass
class LayeredCortexConfig(NeuralComponentConfig):
    """Configuration for layered cortical microcircuit.

    SIZE SPECIFICATION:
    You MUST specify sizes in ONE of three ways:

    1. EXPLICIT (recommended for production):
       LayeredCortexConfig(
           l4_size=192, l23_size=288, l5_size=144, l6a_size=58, l6b_size=38
       )

    2. FROM INPUT (use when you know inputs):
       sizes = LayerSizeCalculator().cortex_from_input(input_size=192)
       LayeredCortexConfig(**sizes)

    3. FROM SCALE (use for prototyping):
       sizes = LayerSizeCalculator().cortex_from_scale(scale_factor=128)
       LayeredCortexConfig(**sizes)

    DO NOT mix patterns or leave sizes at 0 - you'll get validation errors.
    """

    # Layer sizes - ALL REQUIRED (no defaults)
    l4_size: int = field(default=0)
    l23_size: int = field(default=0)
    l5_size: int = field(default=0)
    l6a_size: int = field(default=0)
    l6b_size: int = field(default=0)

    # Input size - OPTIONAL (inferred by builder if not specified)
    input_size: int = 0

    # ... rest of config

    def __post_init__(self):
        """Validate that sizes are specified."""
        core_layers = [self.l4_size, self.l23_size, self.l5_size]

        if any(s == 0 for s in core_layers):
            raise ValueError(
                "All layer sizes must be specified explicitly. "
                "Use LayerSizeCalculator to compute from input_size, "
                "target_output_size, or scale_factor."
            )

        # input_size validation: Either specified OR will be inferred by builder
        # We can't validate here because we don't know sources yet
```

#### 2.2 Remove Ambiguous Parameters

**Remove**: `cortex_size` from `RegionSizes`
**Replace with**: Explicit layer sizes in config

```python
# OLD (ambiguous):
sizes = RegionSizes(cortex_size=128)  # What does this mean?

# NEW (explicit):
cortex_sizes = LayerSizeCalculator().cortex_from_scale(128)
cortex_config = LayeredCortexConfig(**cortex_sizes)
```

---

### Phase 3: Builder Input Size Inference (HIGH PRIORITY)

**Goal**: Handle multi-source inputs correctly.

#### 3.1 Two-Pass Building

```python
# src/thalia/core/brain_builder.py

class BrainBuilder:
    """Build brain with two-pass size inference."""

    def build(self) -> DynamicBrain:
        """Build brain with automatic input size inference.

        Pass 1: Create components with temporary placeholders
        Pass 2: Infer input sizes from connections and finalize
        """
        # PASS 1: Create components
        components = {}
        for name, spec in self._components.items():
            component = self._create_component_placeholder(name, spec)
            components[name] = component

        # PASS 2: Infer input sizes from connections
        input_sizes = self._infer_input_sizes(components, self._connections)

        # PASS 3: Finalize components with correct input sizes
        for name, component in components.items():
            if name in input_sizes:
                component.finalize_input_size(input_sizes[name])

        # PASS 4: Create connections
        connections = self._create_connections(components)

        return DynamicBrain(components, connections, ...)

    def _infer_input_sizes(
        self,
        components: Dict[str, NeuralRegion],
        connections: List[ConnectionSpec]
    ) -> Dict[str, int]:
        """Infer input sizes for each component from connection graph."""
        input_sizes = {}

        for spec in connections:
            target = spec.target
            source_output = components[spec.source].output_size

            if target not in input_sizes:
                input_sizes[target] = 0
            input_sizes[target] += source_output

        return input_sizes
```

#### 3.2 Region Finalization

Each region implements `finalize_input_size()`:

```python
# src/thalia/regions/cortex/layered_cortex.py

class LayeredCortex(NeuralRegion):
    """Layered cortical microcircuit."""

    def finalize_input_size(self, total_input_size: int) -> None:
        """Finalize input size after builder infers from connections.

        Args:
            total_input_size: Sum of all source output sizes

        Raises:
            ValueError: If total_input_size doesn't match expected input_size
        """
        if self.config.input_size == 0:
            # Input size wasn't specified, use inferred value
            object.__setattr__(self.config, "input_size", total_input_size)
        elif self.config.input_size != total_input_size:
            raise ValueError(
                f"Cortex input_size mismatch: "
                f"config specifies {self.config.input_size} "
                f"but connections provide {total_input_size}. "
                f"Either leave input_size=0 (auto-infer) or "
                f"ensure it matches sum of source outputs."
            )

        # Validate that L4 can handle the input
        self._validate_input_capacity(total_input_size)
```

---

### Phase 4: High-Level Factory Methods (MEDIUM PRIORITY)

**Goal**: Make common patterns easy.

```python
# src/thalia/config/brain_presets.py

from thalia.config.size_calculator import LayerSizeCalculator
from thalia.core.dynamic_brain import BrainBuilder

class BrainPresets:
    """High-level factory methods for standard brain architectures."""

    @staticmethod
    def default(
        input_size: int = 128,
        scale: Literal["small", "medium", "large"] = "medium",
        n_actions: int = 4,
    ) -> DynamicBrain:
        """Create default brain architecture.

        Architecture:
        - Sensory Input → Thalamus (relay) → Cortex (layers)
        - Cortex ↔ Hippocampus (bidirectional)
        - Cortex → Striatum (action selection)
        - Cortex → Prefrontal (working memory)

        Args:
            input_size: Size of sensory input
            scale: Overall brain size ("small", "medium", "large")
            n_actions: Number of discrete actions

        Returns:
            Configured brain ready for training
        """
        calc = LayerSizeCalculator()

        # Scale factors
        scale_map = {"small": 64, "medium": 128, "large": 256}
        base = scale_map[scale]

        # Thalamus (1:1 relay)
        thalamus_sizes = calc.thalamus_from_relay(relay_size=input_size)

        # Cortex (from thalamus input)
        cortex_sizes = calc.cortex_from_input(input_size=input_size)

        # Hippocampus (from cortex L2/3 output)
        hippo_input = cortex_sizes["l23_size"]
        hippocampus_sizes = calc.hippocampus_from_input(ec_input_size=hippo_input)

        # Striatum (from cortex L5 output)
        striatum_sizes = calc.striatum_from_actions(
            n_actions=n_actions,
            neurons_per_action=10
        )

        # Build
        builder = (BrainBuilder()
            .add_component("thalamus", "thalamus", **thalamus_sizes)
            .add_component("cortex", "layered_cortex", **cortex_sizes)
            .add_component("hippocampus", "hippocampus", **hippocampus_sizes)
            .add_component("striatum", "striatum", **striatum_sizes)
            .connect("thalamus", "cortex")
            .connect("cortex:l23", "hippocampus")  # Port-based routing
            .connect("hippocampus", "cortex")      # Feedback
            .connect("cortex:l5", "striatum")
        )

        return builder.build()
```

---

### Phase 5: Update Documentation (MEDIUM PRIORITY)

**Goal**: Make the new patterns clear.

#### 5.1 Update Getting Started Guide

Add section: "Understanding Size Specification"

#### 5.2 Create Migration Guide

Document: "Migrating from Old Size API"

#### 5.3 Update All Examples

Show new patterns in all examples and tutorials.

---

## Migration Plan

### Step 1: Add New System (No Breaking Changes)

1. Create `size_calculator.py` with `LayerSizeCalculator`
2. Add deprecation warnings to old functions
3. Update tests to use new patterns (keep old tests passing)

### Step 2: Update Core Components

1. Add `finalize_input_size()` to all regions
2. Update `BrainBuilder` with two-pass building
3. Update configs to require explicit sizes

### Step 3: Update High-Level APIs

1. Update `from_thalia_config()` to use calculator
2. Update `BrainPresets` to use calculator
3. Create migration examples

### Step 4: Remove Deprecated (Breaking)

1. Remove old `compute_*` functions
2. Remove ambiguous parameters (`cortex_size`)
3. Remove `calculate_layer_sizes()` duplicate

---

## Testing Strategy

### Unit Tests

- Test `LayerSizeCalculator` with known biological ratios
- Test each `from_*` method produces consistent results
- Test validation (e.g., can't leave sizes at 0)

### Integration Tests

- Test builder input size inference with multi-source
- Test finalization catches mismatches
- Test growth operations maintain consistency

### Regression Tests

- Ensure old tests still pass with deprecation warnings
- Verify no functional changes to existing brains
- Check performance (two-pass shouldn't be slower)

---

## Success Criteria

1. ✅ Single source of truth for size calculations
2. ✅ No ambiguous "base size" parameters
3. ✅ Multi-source input sizes inferred correctly
4. ✅ Growth API clearly documented
5. ✅ All tests passing with new patterns
6. ✅ Documentation updated with examples
7. ✅ Deprecation warnings for old API
8. ✅ No functional regressions

---

## Timeline

- **Week 1**: Phase 1 (Unified calculator) + Phase 2 (Config API)
- **Week 2**: Phase 3 (Builder inference) + Phase 4 (Factory methods)
- **Week 3**: Phase 5 (Documentation) + Testing
- **Week 4**: Cleanup + Remove deprecated API

**Total**: ~4 weeks for complete migration

---

## Design Decisions

1. **Should we support auto-compute in LayeredCortexConfig.__post_init__()?**
   - Pro: Convenient for simple cases
   - Con: Hides complexity, harder to debug
   - **DECISION**: ❌ No. Be explicit. Users must use LayerSizeCalculator to compute sizes.

2. **How to handle heterogeneous layer ratios?**
   - Some users may want non-standard ratios
   - **DECISION**: ✅ Calculator supports custom `BiologicalRatios`, OR user computes manually

3. **Should `input_size` in config be optional?**
   - Builder can always infer it
   - **DECISION**: ✅ Yes, optional. Builder fills it in during finalization.

4. **Backward compatibility with checkpoints?**
   - No existing checkpoints to migrate
   - **DECISION**: ✅ No migration code needed. Clean break from old API.

---

## Risks and Mitigation

### Risk 1: Breaking All Tests
**Mitigation**: Incremental migration. Keep old API with warnings. Update tests gradually.

### Risk 2: Two-Pass Building Too Complex
**Mitigation**: Thorough testing. Document clearly. Provide examples.

### Risk 3: Users Confused by New API
**Mitigation**: Clear documentation. Migration guide. Good defaults.

---

## Conclusion

This refactoring will:
- **Eliminate confusion** around size specification
- **Fix bugs** caused by inconsistent calculations
- **Make the system more intuitive** for new users
- **Maintain flexibility** for advanced users
- **Improve debuggability** with explicit sizes

The key insight: **Separate intent (what you want) from computation (how to get it) from reality (what exists).**

With this separation, each concern has a clear home, and the system becomes much easier to reason about.
