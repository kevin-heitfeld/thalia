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

### 3. 🟡 Device Management Inconsistency
**Problem**: Two patterns used inconsistently:
```python
# Pattern 1: Pass device at creation (preferred)
torch.randn(..., device=device)

# Pattern 2: Create then move (inefficient)
torch.randn(...).to(device)
```

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
**File**: `docs/patterns/state_management.md`

**Content**:
- When to use `RegionState` dataclass
- When to use direct attributes
- Best practices and examples
- Pattern consistency guidelines

**Effort**: 1 hour  
**Impact**: High - clarifies fundamental architecture pattern

---

#### 1.2 Config Documentation ✅ COMPLETE
**File**: `docs/patterns/config_guide.md`

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

#### 2.2 Import Convenience (Optional)
**File**: `src/thalia/__init__.py`

**Content**: Re-export commonly used classes for convenience

```python
# Public API - commonly used classes
from thalia.core import (
    LIFNeuron, LIFConfig,
    ConductanceLIF, ConductanceLIFConfig,
    WeightInitializer, InitStrategy,
    BrainRegionProtocol, NeuralPathway,
)

from thalia.regions import (
    BrainRegion, RegionConfig,
    Striatum, Hippocampus, Prefrontal,
    LayeredCortex, PredictiveCortex,
)
```

**Note**: Internal code continues using full paths for clarity.

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
2. ✅ Create docs/patterns/state_management.md
3. ✅ Create docs/patterns/config_guide.md
4. ✅ Create docs/patterns/mixins.md
5. ⏳ Add mixin docstrings to major classes (optional enhancement)

**Total effort**: ~5 hours  
**Impact**: High - immediately helps with understanding

---

### Phase 2: Quick Wins (Week 2)
1. ⏳ Device pattern enforcement (search & replace)
2. Optional: Add thalia/__init__.py convenience imports

**Total effort**: ~2 hours  
**Impact**: Medium - consistency improvements

---

### Phase 3: Architecture (As Needed)
1. Optional abstract methods (if boilerplate becomes painful)
2. Large file splitting (only if actively refactoring)

**Total effort**: 3-8+ hours per item  
**Impact**: Medium - quality of life improvements

---

## Success Metrics

After Phase 1 completion:
- ✅ Clear documentation for state management patterns
- ✅ Config hierarchy and parameters documented
- ✅ Mixin usage clarified in docstrings
- ✅ New contributors can understand architecture patterns

After Phase 2 completion:
- ✅ 100% device pattern consistency
- ✅ (Optional) Convenient imports available

---

## Notes

- **Don't over-engineer**: Some "problems" are just inherent complexity of a neuroscience codebase
- **Document first**: Many issues are solved by good docs, not code changes
- **Incremental improvements**: Small, focused changes are better than big rewrites
- **Preserve what works**: The codebase is already well above average - enhance, don't rebuild

---

## Feedback Welcome

This document is a living proposal. Suggestions and revisions are encouraged!

**Last Updated**: December 7, 2025
