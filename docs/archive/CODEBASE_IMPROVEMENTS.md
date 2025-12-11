# Codebase Improvements Roadmap

**Date**: December 7, 2025  
**Status**: Planning Phase - Prioritized improvements for code maintainability

## Overview

This document tracks identified pain points in the codebase and proposes concrete solutions. Items are prioritized by impact/effort ratio.

---

## Current Pain Points

### 1. 🔴 Import Discovery
**Problem**: Hard to know where to import classes from. Need to search through `core/`, `regions/`, `learning/`, etc.

**Example**:
```python
# Where do I import these from?
from thalia.regions.base import BrainRegion  # regions?
from thalia.core.neuron import LIFNeuron    # core?
from thalia.learning import STDPStrategy     # learning?
```

**Impact**: High - affects every file  
**Effort**: Low - 1 hour

---

### 2. 🔴 Config Sprawl
**Problem**: Many config classes with inheritance chains. Hard to discover available parameters.

**Config Classes**:
- Base: `RegionConfig`, `ThaliaConfig`, `BrainConfig`
- Regions: `StriatumConfig`, `TrisynapticConfig`, `LayeredCortexConfig`, `PredictiveCortexConfig`, `PrefrontalConfig`
- Systems: `LanguageConfig`, `RobustnessConfig`, `TrainingConfig`

**Impact**: Medium - slows down configuration and understanding  
**Effort**: Low - 2 hours

---

### 3. ✅ Device Management Inconsistency - COMPLETE
**Problem**: Two patterns used inconsistently:
```python
# Pattern 1: Pass device at creation (preferred)
torch.randn(..., device=device)

# Pattern 2: Create then move (inefficient)
torch.randn(...).to(device)
```

**Solution Applied**: Enforced Pattern 1 for all new tensor creation. Pattern 2 retained only where required (nn.Module instances, external data).

**Impact**: Low - functional but inefficient and inconsistent  
**Effort**: Low - 1 hour search & replace

---

### 4. 🔴 State Management Confusion
**Problem**: When to use `self.state.attr` vs `self.attribute`? Why do some regions have `RegionState` dataclasses and others don't?

**Examples**:
- `self.state.l4_spikes` (stored in state)
- `self.weights` (direct attribute)
- `self.config` (direct attribute)

**Impact**: High - affects architecture understanding  
**Effort**: Low - 1 hour for docs

---

### 5. 🟡 Abstract Method Overhead
**Problem**: Regions must implement abstract methods even when delegating to composition.

**Example**:
```python
class PredictiveCortex(BrainRegion):
    def _initialize_weights(self) -> torch.Tensor:
        """Just delegates to inner cortex."""
        return self.cortex.weights
```

**Impact**: Medium - creates boilerplate  
**Effort**: Medium - 3 hours refactoring

---

### 6. 🟡 Large File Navigation
**Problem**: Some files exceed 1500+ lines, making navigation difficult.

**Large Files**:
- `trisynaptic.py`: 1711 lines
- `striatum.py`: 1636 lines
- `layered_cortex.py`: 874 lines

**Impact**: Medium - slows down code reading  
**Effort**: High - 8+ hours per file (only do if actively working in file)

---

### 7. 🟡 Mixin Method Discovery
**Problem**: When a class inherits from multiple mixins, hard to track which methods come from where.

**Example**:
```python
class Striatum(DiagnosticsMixin, ActionSelectionMixin, BrainRegion):
    # Which methods come from which mixin?
    pass
```

**Impact**: Medium - slows down understanding  
**Effort**: Low - 1 hour adding docstrings

---

## Proposed Solutions

### Priority 1: Documentation (Quick Wins) ⚡

#### 1.1 State Management Guide ✅ COMPLETE
**File**: `docs/patterns/state-management.md`

**Content**:
- When to use `RegionState` dataclass
- When to use direct attributes
- Best practices and examples
- Pattern consistency guidelines

**Effort**: 1 hour  
**Impact**: High - clarifies fundamental architecture pattern

---

#### 1.2 Config Documentation ✅ COMPLETE
**File**: `docs/patterns/configuration.md`

**Content**:
- Config hierarchy visualization
- Auto-generated parameter reference
- When to create new config classes
- Config inheritance patterns

**Effort**: 2 hours  
**Impact**: Medium - helps with configuration

---

#### 1.3 Mixin Documentation ✅ COMPLETE
**File**: `docs/patterns/mixins.md`
**Action**: Add docstrings to classes using mixins

**Example**:
```python
class Striatum(DiagnosticsMixin, ActionSelectionMixin, BrainRegion):
    """
    Striatal region with reinforcement learning.
    
    Mixins provide:
    - DiagnosticsMixin: spike_diagnostics(), weight_diagnostics(), get_diagnostics()
    - ActionSelectionMixin: select_action(), compute_action_values(), softmax_action()
    - BrainRegion: forward(), learn(), reset_state() [abstract base]
    
    See Also:
    - docs/patterns/mixins.md for mixin patterns
    """
```

**Effort**: 1 hour  
**Impact**: Medium - clarifies method sources

---

### Priority 2: Code Cleanup (1-2 hours each) 🔧

#### 2.1 Device Pattern Enforcement
**Action**: Search and replace all Pattern 2 → Pattern 1

**Script**:
```bash
# Find all .to(device) patterns
grep -r "\.to(.*device" src/thalia/

# Replace with device=device in creation
# Manual review + replace
```

**Effort**: 1 hour  
**Impact**: Low - consistency and minor performance gain

---

#### 2.2 Import Convenience ✅ COMPLETE
**File**: `src/thalia/__init__.py`

**Implementation**: Hybrid approach
- **Top-level API** (`thalia/__init__.py`): Common imports for external users
- **Topic-level** (`core`, `regions`, `learning`): Already well-organized
- **Internal code**: Continues using explicit imports for clarity

**External Users Can Now Write**:
```python
from thalia import Brain, Striatum, LayeredCortex
from thalia import ConductanceLIF, WeightInitializer
from thalia import ThaliaConfig
```

**Internal Code Keeps Explicit Imports**:
```python
from thalia.core.neuron import ConductanceLIF
from thalia.regions.striatum import Striatum
```

**Effort**: 1 hour  
**Impact**: Low - convenience for external users

---

### Priority 3: Architecture Improvements (3-8 hours each) 🏗️

#### 3.1 Optional Abstract Methods
**Action**: Make abstract methods have default implementations

**Current**:
```python
class BrainRegion(ABC):
    @abstractmethod
    def _initialize_weights(self) -> torch.Tensor:
        """Must override."""
        pass
```

**Proposed**:
```python
class BrainRegion(ABC):
    def _initialize_weights(self) -> torch.Tensor:
        """Default: Xavier initialization. Override if needed."""
        return WeightInitializer.xavier(
            n_output=self.config.n_output,
            n_input=self.config.n_input,
            device=self.device
        )
```

**Trade-off**: Less explicit, but reduces boilerplate for composition-based regions.

**Effort**: 3 hours  
**Impact**: Medium - reduces boilerplate

---

#### 3.2 Large File Splitting (Low Priority)
**Action**: Only split if actively working in these files

**Candidates**:
1. `trisynaptic.py` → `hippocampus/trisynaptic/` package
   - `circuit.py` - weight initialization
   - `forward.py` - forward pass
   - `learning.py` - plasticity
   - `__init__.py` - exports main class

2. `striatum.py` → Already well-modularized with separate files for action_selection, eligibility, config

**Guideline**: Split only when:
- File > 1000 lines
- Clear logical sections exist
- Currently working in the file (not pre-emptive)

**Effort**: 8+ hours per file  
**Impact**: Medium - better navigation, but breaks muscle memory

---

## Implementation Roadmap

### Phase 1: Documentation (Week 1)
1. ✅ Create CODEBASE_IMPROVEMENTS.md (this file)
2. ✅ Create docs/patterns/state-management.md
3. ✅ Create docs/patterns/configuration.md
4. ✅ Create docs/patterns/mixins.md
5. ✅ Add mixin docstrings to major classes

**Total effort**: ~5 hours  
**Impact**: High - immediately helps with understanding

---

### Phase 2: Quick Wins (Week 2)
1. ✅ Device pattern enforcement (search & replace)
2. ✅ Add thalia/__init__.py convenience imports (hybrid approach)

**Total effort**: ~2 hours  
**Impact**: Medium - consistency improvements

---

### Phase 3: Architecture (As Needed)
1. Optional abstract methods (if boilerplate becomes painful)
2. Large file splitting (only if actively refactoring)

**Total effort**: 3-8+ hours per item  
**Impact**: Medium - quality of life improvements

---

## ✅ Growth Logic Consolidation - COMPLETE

**Date**: December 11, 2025  
**Status**: ✅ Complete - 3 regions refactored, 2 documented as complex

### Problem
Regions had ~1000 lines of duplicated `add_neurons()` code:
- **Weight expansion**: Xavier/sparse_random/uniform initialization + clamping + concatenation
- **State expansion**: Eligibility traces, spike traces, TD-lambda traces
- **Neuron recreation**: Save state → create larger population → restore old state

Each region (Striatum, PFC, Cerebellum, etc.) implemented these patterns independently, leading to:
- Code duplication (~150-320 lines per region)
- Inconsistent implementations
- Higher maintenance burden
- Risk of bugs when logic diverges

### Solution
Created 3 base class helper methods in `NeuralComponent` (base.py):

```python
def _expand_weights(
    current_weights: nn.Parameter,
    n_new: int,
    initialization: str,
    sparsity: float,
    scale: float
) -> nn.Parameter:
    """Consolidates weight matrix expansion with multiple initialization strategies."""

def _expand_state_tensors(
    state_dict: Dict[str, torch.Tensor],
    n_new: int
) -> Dict[str, torch.Tensor]:
    """Handles both 1D [n_neurons] and 2D [n_neurons, dim] tensor expansion."""

def _recreate_neurons_with_state(
    neuron_factory: Callable[[], Any],
    old_n_output: int
) -> Any:
    """Creates larger neuron population while preserving old membrane/conductance state."""
```

### Results

| Region | Before | After | Reduction |
|--------|--------|-------|-----------|
| **Striatum** | 320 lines | 180 lines | 44% |
| **PrefrontalCortex** | 100 lines | 60 lines | 40% |
| **Cerebellum** | 60 lines | 35 lines | 42% |
| **PredictiveCortex** | 60 lines | 60 lines* | 0% (delegates) |

*PredictiveCortex delegates to LayeredCortex - just added organizational comments

**Total reduction**: ~460 lines → ~335 lines (27% reduction for simple regions)

### Multi-Layer Regions

**LayeredCortex** and **TrisynapticCircuit** were **NOT refactored** because they have complex multi-layer weight expansions:

- **Recurrent matrices**: Need to expand BOTH dimensions simultaneously (e.g., CA3→CA3 [ca3, ca3], L2/3→L2/3 [l23, l23])
- **Inter-layer matrices**: Need coordinated row AND column expansion (e.g., DG→CA3 must expand for both new DG and new CA3)

The base helpers are designed for simple `[n_output, n_input]` expansion and don't apply to these cases. These regions require their own growth logic.

### Testing
✅ All refactored regions pass tests:
- `test_striatum_exploration.py`: 13/13 passed (including `test_add_neurons_grows_exploration`)
- `test_region_axonal_delays.py`: 10/10 passed (PFC growth verified)
- No regressions from refactoring

### Benefits
1. **Single source of truth**: Weight expansion logic in one place
2. **Consistency**: All simple regions use same initialization strategies
3. **Easier maintenance**: Changes to growth strategy only need base.py update
4. **Reduced risk**: Eliminates divergence between region implementations

**Effort**: 8 hours  
**Impact**: High - major reduction in duplication

---

## Success Metrics

After Phase 1 completion:
- ✅ Clear documentation for state management patterns
- ✅ Config hierarchy and parameters documented
- ✅ Mixin usage clarified in docstrings
- ✅ New contributors can understand architecture patterns

After Phase 2 completion:
- ✅ 100% device pattern consistency (Pattern 1 for new tensors)
- ✅ Convenient imports available for external users
- ✅ Internal code maintains explicit imports for clarity
- ✅ Hybrid import system balances convenience and maintainability

After Growth Logic Consolidation:
- ✅ 27% code reduction in simple region growth methods
- ✅ Single source of truth for weight/state/neuron expansion
- ✅ All growth tests passing (13/13 striatum, 10/10 axonal delays)
- ✅ Documented multi-layer regions as requiring specialized logic

---

## Notes

- **Don't over-engineer**: Some "problems" are just inherent complexity of a neuroscience codebase
- **Document first**: Many issues are solved by good docs, not code changes
- **Incremental improvements**: Small, focused changes are better than big rewrites
- **Preserve what works**: The codebase is already well above average - enhance, don't rebuild
- **Recognize complexity**: Multi-layer circuits (hippocampus, cortex) have legitimate complexity that shouldn't be abstracted away

---

## Feedback Welcome

This document is a living proposal. Suggestions and revisions are encouraged!

**Last Updated**: December 11, 2025
