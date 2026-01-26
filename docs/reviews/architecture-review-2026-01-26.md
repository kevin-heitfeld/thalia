# Architecture Review – 2026-01-26

## Executive Summary

This comprehensive architectural analysis of the Thalia codebase (focusing on `src/thalia/`) identifies strengths and opportunities for improvement across module organization, naming conventions, separation of concerns, and code maintainability.

**Key Findings:**
- **Strong foundation**: Excellent use of mixins, protocols, and registry pattern for extensibility
- **Good separation**: Learning strategies successfully extracted into pluggable pattern
- **Inconsistencies**: Weight initialization methods, magic numbers, and component organization need standardization
- **Growth opportunities**: Some regions exceed 1500+ lines and would benefit from component extraction
- **Pattern adherence**: Generally strong adherence to biological plausibility constraints, with WeightInitializer pattern mostly adopted

**Impact Overview:**
- **High-impact, low-disruption improvements**: 18 recommendations (17 pending, 1 completed ✅)
- **Moderate refactoring**: 12 recommendations (11 pending, 1 completed ✅)
- **Major restructuring**: 5 recommendations (long-term, multi-week efforts)

**Recent Completions (January 26, 2026):**
- ✅ **Tier 1.11 / 2.7**: Added checkpoint managers for Cerebellum, LayeredCortex, and ThalamicRelay regions
  - All regions now have uniform checkpoint/restore interface
  - See [docs/reviews/checkpoint-managers-implementation.md](docs/reviews/checkpoint-managers-implementation.md)

---

## Tier 1 - High Impact, Low Disruption

### 1.1 Magic Number Consolidation

**Current State**: Numeric literals scattered throughout code (0.01, 0.001, 0.0001) for learning rates, thresholds, noise levels, and activity detection.

**Locations**:
- [src/thalia/tasks/executive_function.py](src/thalia/tasks/executive_function.py#L224-L231): `0.01` for stimulus noise
- [src/thalia/training/curriculum/stage_monitoring.py](src/thalia/training/curriculum/stage_monitoring.py#L184): `0.01` for silent region detection
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L1316): `0.01` for theta oscillation detection
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L2335): `0.01` for softmax temperature
- [src/thalia/regions/prefrontal/goal_emergence.py](src/thalia/regions/prefrontal/goal_emergence.py#L227): `0.01` default learning rate
- [src/thalia/regions/multisensory.py](src/thalia/regions/multisensory.py#L1018): `0.01` for activity threshold
- [src/thalia/neuromodulation/mixin.py](src/thalia/neuromodulation/mixin.py#L321): `0.01` fallback learning rate
- [src/thalia/learning/homeostasis/synaptic_homeostasis.py](src/thalia/learning/homeostasis/synaptic_homeostasis.py#L110): `0.01` for minimum activity

**Proposed Change**: Extract to named constants in [src/thalia/constants/learning.py](src/thalia/constants/learning.py):
```python
# Learning rates
LEARNING_RATE_DEFAULT = 0.01
LEARNING_RATE_FAST = 0.001
LEARNING_RATE_SLOW = 0.0001

# Activity thresholds
ACTIVITY_THRESHOLD_SILENT = 0.01
ACTIVITY_THRESHOLD_MINIMAL = 0.001

# Noise and temperature
STIMULUS_NOISE_STD = 0.01
SOFTMAX_TEMPERATURE_DEFAULT = 0.01
MEMBRANE_NOISE_BIOLOGICAL = 0.002  # 2mV standard deviation
```

**Rationale**: Named constants improve code readability, make biological assumptions explicit, and enable global tuning without hunting through files.

**Impact**:
- **Files affected**: ~25 files across regions/, tasks/, training/, learning/
- **Breaking changes**: None (value-preserving refactor)
- **Maintainability**: High improvement in documenting biological assumptions

---

### 1.2 Weight Initialization Method Standardization

**Current State**: Inconsistent weight initialization methods across regions, with most using `WeightInitializer` but some inconsistencies in method naming.

**Duplication Pattern**:
```python
# Pattern 1 (correct): Using WeightInitializer registry
weights = WeightInitializer.sparse_random(n_out, n_in, sparsity=0.2, device=device)

# Pattern 2 (acceptable): Direct torch for demos/tests
encoded = torch.randn(size, device=device)  # OK in examples/tests

# Pattern 3 (needs attention): Repeated initialization logic in _initialize_weights
```

**Files with `_initialize_weights` methods** (potential consolidation):
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L524): `_initialize_weights`
- [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py#L364): `_initialize_weights`
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py#L374): `_initialize_weights`
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py#L585): `_init_weights` (naming inconsistency)
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py#L2247): `_initialize_weights`
- [src/thalia/regions/cerebellum/cerebellum.py](src/thalia/regions/cerebellum/cerebellum.py#L383): `_initialize_weights_tensor`

**Proposed Change**:
1. Standardize method name to `_initialize_weights` (not `_init_weights` or `_initialize_weights_tensor`)
2. Add base implementation in `NeuralRegion` mixin for common patterns
3. Document when direct `torch.randn` is acceptable (examples, tests, prototyping)

**Rationale**: Consistent naming improves discoverability. Base implementation reduces duplication. `WeightInitializer` usage is already strong (80%+ adoption).

**Impact**:
- **Files affected**: 6 region files
- **Breaking changes**: None (internal method renaming)
- **Maintainability**: Medium improvement

---

### 1.3 Neuron Constants Consolidation

**Current State**: Good use of constants module ([src/thalia/constants/neuron.py](src/thalia/constants/neuron.py)) but some values still hardcoded in regions.

**Examples**:
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L2329): `0.002` membrane noise (should use constant)
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L2342): `0.002` conductance noise
- [src/thalia/regions/prefrontal/prefrontal.py](src/thalia/regions/prefrontal/prefrontal.py#L758): `self.pfc_config.wm_noise_std` (good pattern)

**Proposed Change**:
- Move remaining hardcoded neuron parameters to [src/thalia/constants/neuron.py](src/thalia/constants/neuron.py)
- Add section for biological noise parameters:
```python
# Biological Noise (standard deviations in normalized units)
MEMBRANE_NOISE_STANDARD = 0.002  # 2mV biological fluctuation
CONDUCTANCE_NOISE_STANDARD = 0.002  # Synaptic variability
```

**Rationale**: Centralizes biological assumptions, improves documentation of parameter choices.

**Impact**:
- **Files affected**: 5-8 region files
- **Breaking changes**: None
- **Maintainability**: Low-medium improvement

---

### 1.4 Component Naming Consistency

**Current State**: Mostly consistent use of `*Component` and `*Manager` suffixes, but some variation.

**Examples** (good patterns):
- [src/thalia/regions/striatum/learning_component.py](src/thalia/regions/striatum/learning_component.py#L25): `StriatumLearningComponent`
- [src/thalia/regions/striatum/homeostasis_component.py](src/thalia/regions/striatum/homeostasis_component.py#L74): `StriatumHomeostasisComponent`
- [src/thalia/regions/striatum/checkpoint_manager.py](src/thalia/regions/striatum/checkpoint_manager.py#L58): `StriatumCheckpointManager`
- [src/thalia/regions/striatum/exploration.py](src/thalia/regions/striatum/exploration.py#L76): `ExplorationManager`

**Inconsistency**:
- [src/thalia/regions/striatum/exploration.py](src/thalia/regions/striatum/exploration.py#L361): `StriatumExplorationComponent` vs `ExplorationManager` (both exist)

**Proposed Change**: Clarify naming convention in documentation:
- `*Component`: Part of region, tightly coupled (learning, homeostasis)
- `*Manager`: Standalone service, loosely coupled (checkpoint, exploration)

**Rationale**: Clear naming convention aids navigation and understanding of component coupling.

**Impact**:
- **Files affected**: Documentation update + possible 2-3 class renames
- **Breaking changes**: Low (mostly internal)
- **Maintainability**: Low improvement (clarity)

---

### 1.5 Directory Organization - Language vs. Components

**Current State**: Ambiguous placement of language-related code:
- [src/thalia/language/](src/thalia/language/): encoder.py, decoder.py, position.py
- [src/thalia/components/coding/](src/thalia/components/coding/): (unclear relationship)

**Analysis**: Current structure suggests language processing is separate from "components", but language encoding/decoding IS a component type.

**Proposed Change**: Document or consolidate:
- **Option A** (minimal): Update [docs/architecture/ARCHITECTURE_OVERVIEW.md](docs/architecture/ARCHITECTURE_OVERVIEW.md) to clarify `language/` vs `components/` distinction
- **Option B** (moderate): Move `language/` into `components/language/` for consistency

**Rationale**: Clarifies whether "language" is a first-class system or a component category.

**Impact** (Option A):
- **Files affected**: 1 doc file
- **Breaking changes**: None
- **Maintainability**: Low improvement

**Impact** (Option B):
- **Files affected**: 3-5 files + import updates
- **Breaking changes**: Medium (import paths change)
- **Maintainability**: Medium improvement

**Recommendation**: Start with Option A (documentation), defer Option B until language system expands.

---

### 1.6 State Class Reset Method Duplication

**Current State**: Every region state class implements `reset()` method with similar pattern:

**Locations**:
- [src/thalia/regions/striatum/state.py](src/thalia/regions/striatum/state.py#L304): `StriatumState.reset()`
- [src/thalia/regions/thalamus/state.py](src/thalia/regions/thalamus/state.py#L123): `ThalamicRelayState.reset()`
- [src/thalia/regions/prefrontal/state.py](src/thalia/regions/prefrontal/state.py#L114): `PrefrontalState.reset()`
- [src/thalia/regions/hippocampus/state.py](src/thalia/regions/hippocampus/state.py#L189): `HippocampusState.reset()`
- [src/thalia/regions/cortex/state.py](src/thalia/regions/cortex/state.py#L190): `LayeredCortexState.reset()`
- [src/thalia/regions/cortex/state.py](src/thalia/regions/cortex/state.py#L373): `PredictiveCortexState.reset()`
- [src/thalia/regions/cerebellum/state.py](src/thalia/regions/cerebellum/state.py#L187): `CerebellumState.reset()`

**Pattern** (common across all):
```python
def reset(self) -> None:
    """Reset all state to initial values."""
    self.voltage.zero_()
    self.conductance_E.zero_()
    self.conductance_I.zero_()
    self.refractory_timer.zero_()
    self.adaptation.zero_()
    # ... region-specific state ...
```

**Proposed Change**: Add helper method in [src/thalia/core/region_state.py](src/thalia/core/region_state.py) `BaseRegionState`:
```python
def _reset_neuron_state(self) -> None:
    """Reset common neuron state tensors (voltage, conductances, etc.)."""
    if hasattr(self, 'voltage'):
        self.voltage.zero_()
    if hasattr(self, 'conductance_E'):
        self.conductance_E.zero_()
    # ... etc for common fields
```

Then state classes call:
```python
def reset(self) -> None:
    self._reset_neuron_state()  # Common state
    self.eligibility_trace.zero_()  # Region-specific
```

**Rationale**: Reduces duplication of common reset logic while preserving region-specific needs.

**Impact**:
- **Files affected**: 8 files (1 base + 7 state classes)
- **Breaking changes**: None (internal helper)
- **Maintainability**: Medium improvement

---

### 1.7 Standardize Diagnostic Key Names

**Current State**: Good use of diagnostic schema ([src/thalia/core/diagnostics_schema.py](src/thalia/core/diagnostics_schema.py)), but key names could be more consistent.

**Observation**: Pattern generally consistent (e.g., `region_name.firing_rate`, `region_name.sparsity`), but worth documenting standard key format.

**Proposed Change**: Add diagnostic key naming convention to [docs/patterns/component-parity.md](docs/patterns/component-parity.md):
```
Standard Diagnostic Key Format:
- {component_name}.{metric_name}
- Use snake_case for metric names
- Common metrics: firing_rate, sparsity, mean_weight, weight_change
- Prefix with subcomponent: hippocampus.ca3.firing_rate
```

**Rationale**: Prevents key name drift as system expands.

**Impact**:
- **Files affected**: 1-2 documentation files
- **Breaking changes**: None
- **Maintainability**: Low improvement (preventative)

---

### 1.8 Consolidate Activity Threshold Constants

**Current State**: Activity thresholds for detection scattered across files:
- [src/thalia/training/curriculum/stage_monitoring.py](src/thalia/training/curriculum/stage_monitoring.py#L184): `0.01` for silent regions
- [src/thalia/regions/multisensory.py](src/thalia/regions/multisensory.py#L1018): `0.01` for activity detection
- [src/thalia/learning/homeostasis/synaptic_homeostasis.py](src/thalia/learning/homeostasis/synaptic_homeostasis.py#L110): `0.01` for minimum activity
- [src/thalia/constants/learning.py](src/thalia/constants/learning.py#L37): `SILENCE_DETECTION_THRESHOLD = 0.001` (different value!)

**Inconsistency**: `SILENCE_DETECTION_THRESHOLD = 0.001` in constants but `0.01` used in actual code.

**Proposed Change**:
1. Audit usage to determine if `0.001` or `0.01` is correct threshold
2. Update all code to use constant from [src/thalia/constants/learning.py](src/thalia/constants/learning.py)
3. Add additional thresholds if needed:
```python
ACTIVITY_THRESHOLD_SILENT = 0.001  # <0.1% firing (truly silent)
ACTIVITY_THRESHOLD_MINIMAL = 0.01  # <1% firing (very low activity)
ACTIVITY_THRESHOLD_LOW = 0.05      # <5% firing (biologically sparse)
```

**Rationale**: Resolves actual inconsistency that could cause bugs (different thresholds in different systems).

**Impact**:
- **Files affected**: 5-6 files
- **Breaking changes**: None (fixes bug)
- **Maintainability**: High improvement (correctness)

---

### 1.9 Add Type Hints to Magic Number Utilities

**Current State**: [src/thalia/utils/core_utils.py](src/thalia/utils/core_utils.py) has good utility for phase preferences:
```python
def random_phase_preferences(n_neurons: int, device: str) -> torch.Tensor:
    return torch.rand(n_neurons, device=device) * (2 * math.pi)
```

**Observation**: Good pattern! Replaces magic number `2 * pi` pattern.

**Proposed Change**: Add constant for clarity:
```python
from thalia.constants.time import TWO_PI  # or add to constants

def random_phase_preferences(n_neurons: int, device: str) -> torch.Tensor:
    return torch.rand(n_neurons, device=device) * TWO_PI
```

**Rationale**: Further reduces magic numbers, though `2 * pi` is mathematically obvious.

**Impact**:
- **Files affected**: 1-2 files
- **Breaking changes**: None
- **Maintainability**: Low improvement (marginal)
- **Priority**: Lower than other Tier 1 items

---

### 1.10 Document torch.randn Usage Policy

**Current State**: Mix of `WeightInitializer` (production) and `torch.randn/torch.rand` (tests, examples, demos).

**Proposed Change**: Add policy to [docs/patterns/learning-strategies.md](docs/patterns/learning-strategies.md) or [CONTRIBUTING.md](CONTRIBUTING.md):
```markdown
## Weight Initialization Policy

**Production code**: Always use `WeightInitializer` registry
- Regions: WeightInitializer.sparse_random, WeightInitializer.gaussian
- Pathways: WeightInitializer.xavier, WeightInitializer.orthogonal

**Acceptable torch.randn/rand usage**:
- Test fixtures (controlled random data)
- Examples and demos (pedagogical clarity)
- Prototyping (temporary, mark with TODO)
- Noise generation (add_noise = torch.randn_like(x) * std)

**Never use for**: Synaptic weight initialization in regions or pathways
```

**Rationale**: Clarifies when direct PyTorch calls are acceptable vs. registry required.

**Impact**:
- **Files affected**: 1 documentation file
- **Breaking changes**: None
- **Maintainability**: Medium improvement (prevents future inconsistency)

---

### 1.11 Standardize Checkpoint Manager Naming ✅ COMPLETED

**Status**: Completed January 26, 2026

**Implementation**: Added checkpoint managers for all remaining regions (Cerebellum, LayeredCortex, ThalamicRelay).

**Files Created**:
- [src/thalia/regions/cerebellum/checkpoint_manager.py](src/thalia/regions/cerebellum/checkpoint_manager.py): `CerebellumCheckpointManager` (418 lines)
- [src/thalia/regions/cortex/checkpoint_manager.py](src/thalia/regions/cortex/checkpoint_manager.py): `LayeredCortexCheckpointManager` (480 lines)
- [src/thalia/regions/thalamus/checkpoint_manager.py](src/thalia/regions/thalamus/checkpoint_manager.py): `ThalamicCheckpointManager` (457 lines)

**Files Modified**:
- [src/thalia/regions/cerebellum/__init__.py](src/thalia/regions/cerebellum/__init__.py): Added export
- [src/thalia/regions/cerebellum/cerebellum.py](src/thalia/regions/cerebellum/cerebellum.py): Integrated manager
- [src/thalia/regions/cortex/__init__.py](src/thalia/regions/cortex/__init__.py): Added export
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py): Integrated manager
- [src/thalia/regions/thalamus/__init__.py](src/thalia/regions/thalamus/__init__.py): Added export
- [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py): Integrated manager

**Pattern Consistency**: All regions now follow the same checkpoint manager pattern:
1. Import checkpoint manager class
2. Initialize in `__init__`: `self.checkpoint_manager = RegionCheckpointManager(self)`
3. Delegate `get_full_state()` to `checkpoint_manager.collect_state()`
4. Delegate `load_full_state()` to `checkpoint_manager.restore_state(state)`

**Documentation**: [docs/reviews/checkpoint-managers-implementation.md](docs/reviews/checkpoint-managers-implementation.md)

**Current State**: Checkpoint managers consistently named and implemented across all regions:
- [src/thalia/regions/hippocampus/checkpoint_manager.py](src/thalia/regions/hippocampus/checkpoint_manager.py#L67): `HippocampusCheckpointManager`
- [src/thalia/regions/striatum/checkpoint_manager.py](src/thalia/regions/striatum/checkpoint_manager.py#L58): `StriatumCheckpointManager`
- [src/thalia/regions/prefrontal/checkpoint_manager.py](src/thalia/regions/prefrontal/checkpoint_manager.py#L69): `PrefrontalCheckpointManager`
- [src/thalia/regions/cerebellum/checkpoint_manager.py](src/thalia/regions/cerebellum/checkpoint_manager.py): `CerebellumCheckpointManager` ✨
- [src/thalia/regions/cortex/checkpoint_manager.py](src/thalia/regions/cortex/checkpoint_manager.py): `LayeredCortexCheckpointManager` ✨
- [src/thalia/regions/thalamus/checkpoint_manager.py](src/thalia/regions/thalamus/checkpoint_manager.py): `ThalamicCheckpointManager` ✨

**Impact**:
- **Files affected**: 9 files (3 new managers + 6 modified)
- **Breaking changes**: None
- **Maintainability**: High improvement - all regions now have uniform checkpoint interface
- **Testability**: Improved - checkpoint logic isolated and testable independently

---

### 1.12 Remove Unused TODO Comments

**Current State**: Only 3 TODO comments found (excellent!):
- [src/thalia/training/visualization/live_diagnostics.py](src/thalia/training/visualization/live_diagnostics.py#L543): `TODO(low-priority): Implement animated GIF creation`
- [src/thalia/training/curriculum/stage_manager.py](src/thalia/training/curriculum/stage_manager.py#L1736): `TODO(future): Reduce task complexity if applicable`
- [src/thalia/regions/hippocampus/checkpoint_manager.py](src/thalia/regions/hippocampus/checkpoint_manager.py#L460): `TODO: Check actual neurogenesis config when available`

**Proposed Change**:
1. Convert to GitHub issues for tracking
2. Remove comments or add issue links: `# See issue #123`

**Rationale**: TODOs should live in issue tracker, not code (except active development).

**Impact**:
- **Files affected**: 3 files
- **Breaking changes**: None
- **Maintainability**: Low improvement

---

### 1.13 Clarify Mixin Responsibilities

**Current State**: Excellent mixin architecture with clear separation:
- [src/thalia/mixins/growth_mixin.py](src/thalia/mixins/growth_mixin.py#L29): `GrowthMixin`
- [src/thalia/mixins/resettable_mixin.py](src/thalia/mixins/resettable_mixin.py#L16): `ResettableMixin`
- [src/thalia/mixins/diagnostics_mixin.py](src/thalia/mixins/diagnostics_mixin.py#L26): `DiagnosticsMixin`
- [src/thalia/mixins/state_loading_mixin.py](src/thalia/mixins/state_loading_mixin.py#L23): `StateLoadingMixin`
- [src/thalia/neuromodulation/mixin.py](src/thalia/neuromodulation/mixin.py#L173): `NeuromodulatorMixin`
- [src/thalia/learning/strategy_mixin.py](src/thalia/learning/strategy_mixin.py#L34): `LearningStrategyMixin`

**Observation**: All inherit from appropriate pattern. Good documentation in [docs/patterns/mixins.md](docs/patterns/mixins.md).

**Proposed Change**: Minor doc enhancement - add "Mixin Quick Reference" table to [docs/patterns/mixins.md](docs/patterns/mixins.md):
```markdown
| Mixin | Purpose | Key Methods | Used By |
|-------|---------|-------------|---------|
| GrowthMixin | Dynamic expansion | grow_output, grow_source | Regions, Pathways |
| ResettableMixin | State reset | reset_state | All components |
| DiagnosticsMixin | Health metrics | collect_diagnostics | Regions |
| ... | ... | ... | ... |
```

**Rationale**: Quick reference aids new contributors.

**Impact**:
- **Files affected**: 1 documentation file
- **Breaking changes**: None
- **Maintainability**: Low improvement

---

### 1.14 Verify Device Management Consistency

**Current State**: Mixed patterns for device management:
- **Pattern 1** (preferred): `device` parameter in constructor, stored as `self.device`
- **Pattern 2**: Rely on module's `.to(device)` call
- **Pattern 3**: Check tensor device dynamically

**Observation**: [src/thalia/mixins/device_mixin.py](src/thalia/mixins/device_mixin.py#L18) exists but not widely used.

**Proposed Change**:
1. Audit regions to ensure consistent device parameter in `__init__`
2. Document device management pattern in [docs/patterns/component-parity.md](docs/patterns/component-parity.md)

**Rationale**: Prevents device mismatch bugs in multi-GPU scenarios.

**Impact**:
- **Files affected**: Audit all regions (check), update 2-3 if needed
- **Breaking changes**: None (internal)
- **Maintainability**: Medium improvement (correctness)

---

### 1.15 Consolidate Exploration Constants

**Current State**: Exploration parameters in [src/thalia/constants/exploration.py](src/thalia/constants/exploration.py) (excellent organization).

**Observation**: File exists and is well-structured. Verify all exploration-related magic numbers use it.

**Proposed Change**: Audit [src/thalia/regions/striatum/exploration.py](src/thalia/regions/striatum/exploration.py) and [src/thalia/regions/striatum/action_selection.py](src/thalia/regions/striatum/action_selection.py) for hardcoded exploration thresholds.

**Example**:
- [src/thalia/regions/striatum/action_selection.py](src/thalia/regions/striatum/action_selection.py#L266): `torch.rand(1).item() < exploration_prob`

**Rationale**: Ensure constants module is actually used consistently.

**Impact**:
- **Files affected**: 2-3 files
- **Breaking changes**: None
- **Maintainability**: Low improvement (verification)

---

### 1.16 Learning Strategy Registry Validation

**Current State**: Excellent learning strategy pattern with registry in [src/thalia/learning/strategy_registry.py](src/thalia/learning/strategy_registry.py).

**Observation**: Pattern successfully adopted across codebase. [src/thalia/learning/rules/strategies.py](src/thalia/learning/rules/strategies.py) shows mature strategy implementation.

**Proposed Change**: Add registration validation in strategy registry:
```python
def validate_strategy(strategy_cls):
    """Ensure strategy implements required protocol."""
    required_methods = ['compute_update', 'reset']
    for method in required_methods:
        if not hasattr(strategy_cls, method):
            raise TypeError(f"Strategy {strategy_cls} missing {method}")
```

**Rationale**: Prevents registration of incomplete strategies.

**Impact**:
- **Files affected**: 1 file (registry)
- **Breaking changes**: None (validation only)
- **Maintainability**: Low improvement (quality gate)

---

### 1.17 Standardize Import Organization

**Current State**: Generally good import organization, but some variation in grouping.

**Proposed Change**: Document import order in [CONTRIBUTING.md](CONTRIBUTING.md):
```python
# Standard library
import math
from typing import Dict, Optional

# Third-party
import torch
import torch.nn as nn

# Thalia core
from thalia.core.neural_region import NeuralRegion
from thalia.core.region_state import BaseRegionState

# Thalia components
from thalia.components.neurons import create_pyramidal_neurons
from thalia.components.synapses import WeightInitializer

# Thalia utilities
from thalia.constants.neuron import TAU_MEM_STANDARD
from thalia.typing import SourceOutputs
```

**Rationale**: Consistent import organization improves readability and reduces merge conflicts.

**Impact**:
- **Files affected**: 1 documentation file + optional linter config
- **Breaking changes**: None
- **Maintainability**: Low improvement

---

### 1.18 Document Growth API Standardization

**Current State**: Excellent standardization of growth API:
- [src/thalia/core/neural_region.py](src/thalia/core/neural_region.py#L670): `NeuralRegion.grow_output`
- [src/thalia/core/neural_region.py](src/thalia/core/neural_region.py#L591): `NeuralRegion.grow_source`

All regions implement consistent signatures. See:
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L815)
- [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py#L1007)
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py#L1300)
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py#L1657)
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py#L822)
- [src/thalia/regions/cerebellum/cerebellum.py](src/thalia/regions/cerebellum/cerebellum.py#L528)

**Observation**: Pattern is already well-established and consistent!

**Proposed Change**: Add growth API documentation to [docs/patterns/component-parity.md](docs/patterns/component-parity.md) as a **success story** example:
```markdown
## Growth API (Fully Standardized) ✅

All `NeuralRegion` subclasses implement standardized growth methods:

### grow_output(n_new: int) -> None
Grows output dimension by adding neurons.

### grow_source(source_name: str, new_size: int) -> None
Grows input dimension for a specific source (MultiSourcePathway regions only).

**Adoption**: 100% across all regions (Hippocampus, Thalamus, Striatum, Cortex, Cerebellum, Prefrontal, Multisensory)
```

**Rationale**: Celebrates successful standardization, serves as template for other patterns.

**Impact**:
- **Files affected**: 1 documentation file
- **Breaking changes**: None
- **Maintainability**: Low improvement (documentation of success)

---

## Tier 2 - Moderate Refactoring

### 2.1 Extract Large Region Files into Components

**Current State**: Some region files exceed 1500+ lines, making navigation difficult:
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py): ~2600 lines
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py): ~2400 lines
- [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py): ~1100 lines
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py): ~1200 lines

**Observation**:
- Striatum already extracted some components: `LearningComponent`, `HomeostasisComponent`, `ExplorationComponent`, `CheckpointManager`
- Hippocampus has `LearningComponent` and `CheckpointManager`
- Cortex and Thalamus lack component extraction

**Proposed Change**: Extract following components:

**Hippocampus** (extract from 2400 to ~1200 lines):
- `HippocampusReplayComponent`: Spontaneous replay logic ([src/thalia/regions/hippocampus/spontaneous_replay.py](src/thalia/regions/hippocampus/spontaneous_replay.py) already exists, integrate)
- `HippocampusSynapticTaggingComponent`: Tagging logic ([src/thalia/regions/hippocampus/synaptic_tagging.py](src/thalia/regions/hippocampus/synaptic_tagging.py) exists, integrate)
- `HippocampusConsolidationComponent`: Memory consolidation

**Cortex** (extract from 1200 to ~600 lines):
- `CortexLearningComponent`: BCM + STDP composite strategy
- `CortexCheckpointManager`: Missing (add)

**Thalamus** (extract from 1100 to ~600 lines):
- `ThalamicGatingComponent`: Attention gating logic
- `ThalamicCheckpointManager`: Missing (add)

**Rationale**: Files >800 lines are harder to navigate. Component extraction follows established pattern (see [src/thalia/core/region_components.py](src/thalia/core/region_components.py)).

**Impact**:
- **Files affected**: 3 large region files → 6-8 new component files
- **Breaking changes**: Low (internal refactor, public API unchanged)
- **Maintainability**: High improvement
- **Effort**: 3-5 days

---

### 2.2 Consolidate Neuron Creation Patterns

**Current State**: Regions create neurons directly with `ConductanceLIF`:
```python
self.neurons = ConductanceLIF(n_neurons, neuron_config, device=device)
```

**Observation**: [src/thalia/components/neurons/neuron_factory.py](src/thalia/components/neurons/neuron_factory.py) exists with registry pattern but underutilized.

**Available factory types**:
```python
NeuronFactory.create("pyramidal", n_neurons=100, device=device)
NeuronFactory.create("relay", n_neurons=64, device=device)
NeuronFactory.create("cortical_layer", n_neurons=256, layer="L2/3", device=device)
```

**Proposed Change**: Migrate regions to use `NeuronFactory` for standardization:
```python
# Before
self.ca3_neurons = ConductanceLIF(self.config.ca3_size, neuron_config, device=device)

# After
self.ca3_neurons = NeuronFactory.create(
    "pyramidal",  # Explicit neuron type
    n_neurons=self.config.ca3_size,
    device=device,
    tau_mem=15.0,  # CA3-specific parameters
)
```

**Rationale**:
1. Makes neuron types explicit (improves biological accuracy documentation)
2. Enables future neuron type diversification
3. Consistent with other registry patterns (WeightInitializer, ComponentRegistry)

**Impact**:
- **Files affected**: ~8-10 region files
- **Breaking changes**: None (internal change)
- **Maintainability**: Medium improvement
- **Effort**: 2-3 days

---

### 2.3 Separate State Management from Region Logic

**Current State**: Region state classes ([src/thalia/regions/*/state.py](src/thalia/regions/)) follow good pattern with dataclasses.

**Antipattern detected**: Some state manipulation logic scattered in main region file instead of state class.

**Example pattern** (good):
```python
# State class handles state operations
@dataclass
class HippocampusState(BaseRegionState):
    ca3_voltage: torch.Tensor

    def reset(self) -> None:
        self.ca3_voltage.zero_()
```

**Proposed Change**: Audit regions for state manipulation in `forward()` and extract to state methods:
```python
# Before: Direct manipulation in forward()
def forward(self, inputs):
    self.state.ca3_voltage += current
    self.state.ca3_voltage.clamp_(-1.0, 1.0)

# After: State method
def forward(self, inputs):
    self.state.update_voltage(current)

# In state class
def update_voltage(self, current: torch.Tensor) -> None:
    self.ca3_voltage += current
    self.ca3_voltage.clamp_(self.v_reset, self.v_threshold)
```

**Rationale**: Encapsulates state logic, improves testability.

**Impact**:
- **Files affected**: 5-7 region files + state files
- **Breaking changes**: None (internal)
- **Maintainability**: Medium improvement
- **Effort**: 2-3 days

---

### 2.4 Unify Pathway Architectures

**Current State**: Multiple pathway implementations:
- [src/thalia/pathways/axonal_projection.py](src/thalia/pathways/axonal_projection.py#L83): `AxonalProjection` (pure routing, delays)
- [src/thalia/pathways/sensory_pathways.py](src/thalia/pathways/sensory_pathways.py): `VisualPathway`, `AuditoryPathway`, `LanguagePathway` (more complex)
- [src/thalia/regions/striatum/pathway_base.py](src/thalia/regions/striatum/pathway_base.py#L83): `StriatumPathway` (region-specific)

**Observation**:
- Axonal projection is the canonical "weights-at-dendrites" implementation
- Sensory pathways encode/transform sensory data
- Striatum pathway is region-specific base class (good pattern)

**Inconsistency**: Sensory pathways in `pathways/` vs striatum pathway in `regions/striatum/`.

**Proposed Change**:
1. **Option A** (minimal): Document pathway types in [docs/patterns/component-parity.md](docs/patterns/component-parity.md):
   - **Routing pathways**: Pure spike routing (AxonalProjection)
   - **Sensory pathways**: Input encoding/preprocessing
   - **Region-specific pathways**: Extracted complexity (StriatumPathway)

2. **Option B** (structural): Move region-specific pathways to `pathways/region_specific/`:
   - `pathways/region_specific/striatum_pathway.py`
   - Benefits: Clearer separation, better discoverability
   - Cost: Import path changes

**Rationale**: Clarifies pathway taxonomy, improves navigation.

**Impact** (Option A):
- **Files affected**: 1 documentation file
- **Breaking changes**: None
- **Maintainability**: Low improvement

**Impact** (Option B):
- **Files affected**: 1 moved file + import updates
- **Breaking changes**: Medium (import paths)
- **Maintainability**: Medium improvement
- **Effort**: 1-2 days

**Recommendation**: Start with Option A, consider Option B if more region-specific pathways emerge.

---

### 2.5 Consolidate Learning Trace Management

**Current State**: Learning traces managed in multiple places:
- [src/thalia/learning/eligibility/trace_manager.py](src/thalia/learning/eligibility/trace_manager.py): Eligibility traces
- [src/thalia/components/synapses/traces.py](src/thalia/components/synapses/traces.py): Generic trace utilities
- Individual learning strategies manage their own traces

**Duplication**: Trace decay logic repeated across strategies:
```python
# In STDP strategy
self.pre_trace = self.pre_trace * decay + pre_spikes

# In BCM strategy
self.post_trace = self.post_trace * decay + post_spikes

# In three-factor strategy
self.eligibility = self.eligibility * decay + correlation
```

**Proposed Change**: Consolidate trace management into `TraceManager` with strategy pattern:
```python
from thalia.learning.eligibility import TraceManager

class STDPStrategy:
    def __init__(self, config):
        self.trace_manager = TraceManager(
            trace_type="exponential",
            tau_ms=config.tau_stdp
        )

    def compute_update(self, ...):
        pre_trace = self.trace_manager.update_trace(pre_spikes)
```

**Rationale**: DRY principle, centralizes trace dynamics for consistency.

**Impact**:
- **Files affected**: ~6 learning strategy files
- **Breaking changes**: None (internal to strategies)
- **Maintainability**: Medium improvement
- **Effort**: 2-3 days

---

### 2.6 Standardize Diagnostics Collection Pattern

**Current State**: Regions implement `collect_diagnostics()` with varied return structures:
```python
# Some regions return flat dict
return {"firing_rate": rate, "sparsity": sparse}

# Others return nested dict
return {
    "ca3": {"firing_rate": ca3_rate},
    "ca1": {"firing_rate": ca1_rate}
}
```

**Proposed Change**: Standardize on nested structure with subregion support:
```python
from thalia.mixins.diagnostic_collector_mixin import DiagnosticCollectorMixin

def collect_diagnostics(self) -> Dict[str, Any]:
    diags = self._collect_basic_diagnostics()  # From mixin

    # Add region-specific metrics
    diags.update({
        "subregions": {
            "ca3": self._collect_ca3_diagnostics(),
            "ca1": self._collect_ca1_diagnostics(),
        }
    })
    return diags
```

**Rationale**: Consistent structure improves diagnostics aggregation and visualization.

**Impact**:
- **Files affected**: ~8 region files
- **Breaking changes**: Low (diagnostic consumers may need updates)
- **Maintainability**: Medium improvement
- **Effort**: 2 days

---

### 2.7 Extract Checkpoint Logic into Managers ✅ COMPLETED

**Status**: Completed January 26, 2026 (same implementation as Tier 1.11)

**Implementation Summary**: Added checkpoint managers for all remaining regions following the established pattern from Hippocampus, Striatum, and Prefrontal regions.

**Files Created**:
- [src/thalia/regions/cerebellum/checkpoint_manager.py](src/thalia/regions/cerebellum/checkpoint_manager.py): `CerebellumCheckpointManager` (418 lines)
  - Handles climbing fiber error signals, eligibility traces, enhanced microcircuit (granule/purkinje/DCN)
- [src/thalia/regions/cortex/checkpoint_manager.py](src/thalia/regions/cortex/checkpoint_manager.py): `LayeredCortexCheckpointManager` (480 lines)
  - Manages 6-layer state (L4/L2/3/L5/L6a/L6b), recurrent connections, gap junctions, attention gating
- [src/thalia/regions/thalamus/checkpoint_manager.py](src/thalia/regions/thalamus/checkpoint_manager.py): `ThalamicCheckpointManager` (457 lines)
  - Handles dual populations (relay + TRN), burst/tonic modes, alpha gating

**Pattern Followed**: All checkpoint managers inherit from [src/thalia/managers/base_checkpoint_manager.py](src/thalia/managers/base_checkpoint_manager.py):
```python
class RegionCheckpointManager(BaseCheckpointManager):
    def __init__(self, region): ...
    def collect_state(self) -> Dict[str, Any]: ...
    def restore_state(self, state: Dict[str, Any]) -> None: ...
    def get_neuromorphic_state(self) -> Dict[str, Any]: ...
```

**Region Integration**: Each region now delegates checkpoint operations:
```python
# In __init__:
self.checkpoint_manager = RegionCheckpointManager(self)

# In get_full_state:
return self.checkpoint_manager.collect_state()

# In load_full_state:
self.checkpoint_manager.restore_state(state)
```

**Documentation**: [docs/reviews/checkpoint-managers-implementation.md](docs/reviews/checkpoint-managers-implementation.md)

**Current State**: Complete checkpoint manager coverage:
- [src/thalia/regions/hippocampus/checkpoint_manager.py](src/thalia/regions/hippocampus/checkpoint_manager.py) ✅
- [src/thalia/regions/striatum/checkpoint_manager.py](src/thalia/regions/striatum/checkpoint_manager.py) ✅
- [src/thalia/regions/prefrontal/checkpoint_manager.py](src/thalia/regions/prefrontal/checkpoint_manager.py) ✅
- [src/thalia/regions/cerebellum/checkpoint_manager.py](src/thalia/regions/cerebellum/checkpoint_manager.py) ✅ NEW
- [src/thalia/regions/cortex/checkpoint_manager.py](src/thalia/regions/cortex/checkpoint_manager.py) ✅ NEW
- [src/thalia/regions/thalamus/checkpoint_manager.py](src/thalia/regions/thalamus/checkpoint_manager.py) ✅ NEW

**Impact**:
- **Files affected**: 9 files (3 new managers + 6 region integrations)
- **Breaking changes**: None (maintains existing interface)
- **Maintainability**: High improvement - uniform checkpoint interface
- **Testability**: High improvement - checkpoint logic isolated
- **Effort**: Completed in 1 day

---

### 2.8 Consolidate Configuration Management

**Current State**: Good configuration system with base classes:
- [src/thalia/config/base.py](src/thalia/config/base.py): `BaseConfig`
- [src/thalia/config/region_configs.py](src/thalia/config/region_configs.py): Region-specific configs
- [src/thalia/config/neuron_config.py](src/thalia/config/neuron_config.py): Neuron configs

**Antipattern**: Some regions embed config as dataclass, others use inheritance:
```python
# Pattern 1: Config as separate dataclass (good)
@dataclass
class StriatumConfig:
    n_d1: int
    n_d2: int

# Pattern 2: Config embedded in region (avoid)
class Cortex:
    def __init__(self, n_neurons: int, learning_rate: float):
        self.n_neurons = n_neurons  # Should be in config
```

**Proposed Change**: Audit all regions to ensure config separation:
1. All configurable parameters in dedicated `*Config` dataclass
2. Config inherits from `BaseConfig` (gets device, dtype, seed)
3. Region constructor takes config object, not individual parameters

**Rationale**: Consistent configuration management, easier serialization, clearer parameter documentation.

**Impact**:
- **Files affected**: 2-3 region files (minor adjustments)
- **Breaking changes**: Low (constructor signatures)
- **Maintainability**: Medium improvement
- **Effort**: 1-2 days

---

### 2.9 Improve Multi-Source Input Validation

**Current State**: Regions accept `Dict[str, torch.Tensor]` inputs but validation is inconsistent.

**Example issue**:
```python
def forward(self, inputs: SourceOutputs) -> torch.Tensor:
    # What if inputs is empty?
    # What if key doesn't exist?
    cortex_input = inputs["cortex"]  # KeyError if missing
```

**Proposed Change**: Add input validation mixin:
```python
from thalia.mixins import InputValidationMixin

class NeuralRegion(nn.Module, InputValidationMixin):
    def forward(self, inputs: SourceOutputs) -> torch.Tensor:
        self._validate_inputs(inputs, required_sources=["cortex", "thalamus"])
        # Proceed safely
```

Or use base method in NeuralRegion:
```python
def _validate_source_inputs(self, inputs: SourceOutputs) -> None:
    """Validate multi-source inputs against registered sources."""
    for source in self.synaptic_weights.keys():
        if source not in inputs:
            raise ValueError(f"Missing input for registered source: {source}")
```

**Rationale**: Prevents runtime errors from missing inputs, improves error messages.

**Impact**:
- **Files affected**: 1 base class + optional mixin
- **Breaking changes**: None (adds validation)
- **Maintainability**: Medium improvement
- **Effort**: 1-2 days

---

### 2.10 Consolidate Oscillator Integration

**Current State**: Oscillator integration in regions is manual:
```python
def forward(self, inputs, theta_phase, gamma_phase):
    theta_mod = compute_theta_modulation(theta_phase)
    # Use theta_mod for gating
```

**Observation**: [src/thalia/coordination/oscillator.py](src/thalia/coordination/oscillator.py) provides excellent oscillator system, but integration pattern varies across regions.

**Proposed Change**: Standardize oscillator integration via mixin or helper:
```python
from thalia.coordination import OscillatorIntegrationMixin

class Hippocampus(NeuralRegion, OscillatorIntegrationMixin):
    def forward(self, inputs):
        theta_mod = self.get_theta_modulation()  # From mixin
        gamma_mod = self.get_gamma_modulation()
```

**Rationale**: Reduces boilerplate, ensures consistent oscillator phase handling.

**Impact**:
- **Files affected**: 5-7 regions
- **Breaking changes**: None (internal)
- **Maintainability**: Medium improvement
- **Effort**: 2-3 days

---

### 2.11 Document Component Lifecycle

**Current State**: Component initialization order and lifecycle not formally documented.

**Proposed Change**: Add lifecycle documentation to [docs/patterns/component-parity.md](docs/patterns/component-parity.md):
```markdown
## Component Lifecycle

### Initialization Order
1. `__init__`: Create neurons, allocate state tensors
2. `register_source`: Add input sources (weights allocated)
3. `reset_state`: Initialize dynamic state to resting values
4. `to(device)`: Move to GPU (if needed)

### Forward Pass
1. `forward(inputs)`: Process multi-source inputs
2. Apply learning (if enabled): `learning_strategy.apply(...)`
3. Update internal state
4. Return output spikes

### Checkpointing
1. `get_state()`: Export state dict
2. `load_state(state_dict)`: Restore from checkpoint
3. `validate_state()`: Verify integrity
```

**Rationale**: Helps developers understand component contracts, prevents initialization bugs.

**Impact**:
- **Files affected**: 1-2 documentation files
- **Breaking changes**: None
- **Maintainability**: Medium improvement
- **Effort**: 1 day

---

### 2.12 Standardize Error Messages

**Current State**: Good use of [src/thalia/core/errors.py](src/thalia/core/errors.py) for custom exceptions.

**Observation**: Error messages vary in detail and format across regions.

**Proposed Change**: Add error message templates to `errors.py`:
```python
class ComponentError(Exception):
    """Base exception for component errors."""

    @classmethod
    def missing_input(cls, component_name: str, missing_source: str):
        return cls(
            f"{component_name} expected input from '{missing_source}' but it was not provided. "
            f"Registered sources: {registered_sources}"
        )
```

**Rationale**: Consistent, informative error messages improve debugging experience.

**Impact**:
- **Files affected**: 1 errors file + 5-10 regions
- **Breaking changes**: None (improves errors)
- **Maintainability**: Low-medium improvement
- **Effort**: 1-2 days

---

## Tier 3 - Major Restructuring

### 3.1 Unify Region and Pathway Hierarchies

**Current State**: Regions and pathways in separate directories:
- `src/thalia/regions/` - Brain regions
- `src/thalia/pathways/` - Connections

**Design question**: Should pathways be first-class alongside regions, or nested within?

**Option A** (current): Keep separate
```
src/thalia/
  regions/
    cortex/
    hippocampus/
  pathways/
    axonal_projection.py
    sensory_pathways.py
```

**Option B**: Nest pathways under regions
```
src/thalia/regions/
  cortex/
    layered_cortex.py
    pathways/
      cortical_feedback.py
  hippocampus/
    trisynaptic.py
    pathways/
      perforant_path.py
```

**Option C**: Unified components directory
```
src/thalia/components/
  brain/
    cortex/
    hippocampus/
  pathways/
    axonal/
    sensory/
```

**Analysis**:
- Current structure (A) is clean and discoverable
- Option B better reflects biological organization (pathways connect regions)
- Option C is most flexible but loses biological metaphor

**Recommendation**: **Keep Option A** (current structure). It's working well, component registry handles dynamic relationships.

**If changing** (not recommended for now):
- **Impact**: ~50+ files with import updates
- **Breaking changes**: High (all imports change)
- **Effort**: 1-2 weeks

---

### 3.2 Extract Common Layer Implementations

**Current State**: Cortex has laminar structure (L4, L2/3, L5, L6), but layer logic is embedded in [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py).

**Potential reuse**: Other regions (e.g., Thalamus) have layer-like structures.

**Proposed Change**: Extract generic layer abstraction:
```python
# src/thalia/core/neural_layer.py
class NeuralLayer(nn.Module):
    """Generic neural layer with neurons and local connectivity."""

    def __init__(self, n_neurons, config):
        self.neurons = NeuronFactory.create(config.neuron_type, n_neurons)
        self.lateral_weights = ...

    def forward(self, bottom_up, top_down=None):
        # Generic layer processing
```

Then cortex becomes:
```python
class LayeredCortex(NeuralRegion):
    def __init__(self, config):
        self.L4 = NeuralLayer(config.n_l4, layer_config)
        self.L23 = NeuralLayer(config.n_l23, layer_config)
```

**Rationale**: Reusable layer abstraction, reduces duplication if other regions adopt laminar structure.

**Concerns**:
- May over-abstract (each layer has unique properties)
- Current implementation is clear and biological
- Premature optimization?

**Recommendation**: **Defer** until clear reuse case emerges. Current structure is readable.

**If implementing**:
- **Impact**: 1 new base class + 2-3 regions refactored
- **Breaking changes**: None (internal)
- **Effort**: 1-2 weeks

---

### 3.3 Implement Event-Driven Simulation

**Current State**: Clock-driven simulation (fixed timestep) in [src/thalia/core/dynamic_brain.py](src/thalia/core/dynamic_brain.py).

**Biological limitation**: Real neurons spike asynchronously, not on fixed grid.

**Proposed Change**: Event-driven simulation mode where spikes trigger immediate processing:
```python
class EventDrivenBrain:
    def process_event(self, spike_event):
        source_region = spike_event.region
        target_regions = self.connectivity[source_region]
        for target in target_regions:
            target.receive_spike(spike_event)
```

**Benefits**:
- More biologically accurate
- Potentially faster for sparse networks
- Supports variable delays naturally

**Challenges**:
- Major architectural change
- Learning rule adaptation needed
- Numerical integration complexities

**Recommendation**: **Research project** - requires careful design, potentially separate module.

**If implementing**:
- **Impact**: New brain class, adapter pattern for regions
- **Breaking changes**: None (new mode, keep clock-driven)
- **Effort**: 4-6 weeks

---

### 3.4 Generalize Multi-Modal Integration

**Current State**: [src/thalia/regions/multisensory.py](src/thalia/regions/multisensory.py) handles visual, auditory, language integration.

**Observation**: Hardcoded for specific modalities. What about tactile, olfactory, gustatory?

**Proposed Change**: Generic multi-modal integration region:
```python
class MultiModalIntegration(NeuralRegion):
    def __init__(self, modalities: List[str]):
        for modality in modalities:
            self.register_modality(modality)

    def register_modality(self, name: str):
        # Dynamically create modality pathway
```

**Benefits**:
- Extensible to arbitrary modalities
- Cleaner architecture for multi-modal research

**Challenges**:
- Loses biological specificity (superior colliculus is visual/auditory specific)
- May over-generalize

**Recommendation**: **Evaluate need** - current implementation works for main modalities. Consider if olfactory/tactile integration becomes priority.

**If implementing**:
- **Impact**: 1 new generic region, migrate current multisensory
- **Breaking changes**: Medium
- **Effort**: 2-3 weeks

---

### 3.5 Distributed Multi-GPU Training

**Current State**: Single-device execution. Device management exists but no distributed training.

**Proposed Change**: Support for:
1. Model parallelism (regions on different GPUs)
2. Data parallelism (batch distribution)
3. Pipeline parallelism (sequential region execution)

**Example**:
```python
brain = DynamicBrain(config, devices=["cuda:0", "cuda:1"])
# Cortex on GPU 0, Hippocampus on GPU 1
```

**Benefits**:
- Scale to larger models
- Faster training

**Challenges**:
- Spike communication between GPUs (latency)
- Synchronization complexity
- Debugging difficulty

**Recommendation**: **Future enhancement** - requires careful profiling to identify bottlenecks first.

**If implementing**:
- **Impact**: Core brain architecture changes, new device management
- **Breaking changes**: None (opt-in feature)
- **Effort**: 4-8 weeks

---

## Risk/Impact Assessment and Sequencing

### Recommended Implementation Sequence

**Phase 1: Quick Wins (Week 1-2)**
Implement Tier 1 items 1.1-1.10:
- Consolidate magic numbers → constants
- Standardize weight initialization naming
- Document patterns (TODO cleanup, import style, etc.)
- **Risk**: Low, **Impact**: High visibility improvements

**Phase 2: Structural Improvements (Week 3-5)**
Implement Tier 2 items 2.1-2.7:
- Extract components from large files (Hippocampus, Cortex)
- Add missing checkpoint managers
- Consolidate learning trace management
- **Risk**: Medium (requires testing), **Impact**: High maintainability

**Phase 3: Pattern Consolidation (Week 6-7)**
Implement Tier 2 items 2.8-2.12:
- Configuration management audit
- Input validation improvements
- Oscillator integration standardization
- **Risk**: Low-medium, **Impact**: Medium quality improvements

**Phase 4: Strategic Evaluation (Week 8+)**
Evaluate Tier 3 items:
- Most can be **deferred** (working well as-is)
- Prioritize based on performance profiling results
- Consider as research projects, not immediate refactors

### Risk Mitigation Strategies

1. **Test Coverage**: Ensure comprehensive tests before refactoring large files
2. **Incremental Changes**: One Tier 1 item at a time, validate before next
3. **Backward Compatibility**: Preserve public APIs during internal refactors
4. **Documentation**: Update docs alongside code changes
5. **Review Checkpoints**: Team review after each phase

---

## Appendix A: Affected Files and Links

### Tier 1 Files (18 recommendations)

**Magic Number Consolidation (1.1)**:
- [src/thalia/constants/learning.py](src/thalia/constants/learning.py)
- [src/thalia/tasks/executive_function.py](src/thalia/tasks/executive_function.py)
- [src/thalia/training/curriculum/stage_monitoring.py](src/thalia/training/curriculum/stage_monitoring.py)
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py)
- [src/thalia/regions/prefrontal/goal_emergence.py](src/thalia/regions/prefrontal/goal_emergence.py)
- [src/thalia/regions/multisensory.py](src/thalia/regions/multisensory.py)
- [src/thalia/neuromodulation/mixin.py](src/thalia/neuromodulation/mixin.py)
- [src/thalia/learning/homeostasis/synaptic_homeostasis.py](src/thalia/learning/homeostasis/synaptic_homeostasis.py)

**Weight Initialization (1.2)**:
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py)
- [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py)
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py)
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py)
- [src/thalia/regions/cerebellum/cerebellum.py](src/thalia/regions/cerebellum/cerebellum.py)
- [src/thalia/core/neural_region.py](src/thalia/core/neural_region.py)

**State Reset Duplication (1.6)**:
- [src/thalia/core/region_state.py](src/thalia/core/region_state.py)
- [src/thalia/regions/striatum/state.py](src/thalia/regions/striatum/state.py)
- [src/thalia/regions/thalamus/state.py](src/thalia/regions/thalamus/state.py)
- [src/thalia/regions/prefrontal/state.py](src/thalia/regions/prefrontal/state.py)
- [src/thalia/regions/hippocampus/state.py](src/thalia/regions/hippocampus/state.py)
- [src/thalia/regions/cortex/state.py](src/thalia/regions/cortex/state.py)
- [src/thalia/regions/cerebellum/state.py](src/thalia/regions/cerebellum/state.py)

### Tier 2 Files (12 recommendations)

**Large File Extraction (2.1)**:
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py) (2600 lines)
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py) (2400 lines)
- [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py) (1100 lines)
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py) (1200 lines)

**Checkpoint Managers (2.7)**:
- NEW: `src/thalia/regions/cerebellum/checkpoint_manager.py`
- NEW: `src/thalia/regions/cortex/checkpoint_manager.py`
- NEW: `src/thalia/regions/thalamus/checkpoint_manager.py`

**Learning Trace Consolidation (2.5)**:
- [src/thalia/learning/eligibility/trace_manager.py](src/thalia/learning/eligibility/trace_manager.py)
- [src/thalia/components/synapses/traces.py](src/thalia/components/synapses/traces.py)
- [src/thalia/learning/rules/strategies.py](src/thalia/learning/rules/strategies.py)

### Tier 3 Files (5 recommendations)

**No immediate file changes recommended** - all Tier 3 items are long-term strategic considerations.

---

## Appendix B: Detected Code Duplications

### B.1 State Reset Pattern

**Duplication**: Identical neuron state reset logic in 7 state classes.

**Example code** (repeated):
```python
def reset(self) -> None:
    """Reset all state to initial values."""
    self.voltage.zero_()
    self.conductance_E.zero_()
    self.conductance_I.zero_()
    self.refractory_timer.zero_()
    self.adaptation.zero_()
```

**Locations**:
1. [src/thalia/regions/striatum/state.py](src/thalia/regions/striatum/state.py#L304)
2. [src/thalia/regions/thalamus/state.py](src/thalia/regions/thalamus/state.py#L123)
3. [src/thalia/regions/prefrontal/state.py](src/thalia/regions/prefrontal/state.py#L114)
4. [src/thalia/regions/hippocampus/state.py](src/thalia/regions/hippocampus/state.py#L189)
5. [src/thalia/regions/cortex/state.py](src/thalia/regions/cortex/state.py#L190)
6. [src/thalia/regions/cortex/state.py](src/thalia/regions/cortex/state.py#L373)
7. [src/thalia/regions/cerebellum/state.py](src/thalia/regions/cerebellum/state.py#L187)

**Consolidation**: Add `_reset_neuron_state()` helper in [src/thalia/core/region_state.py](src/thalia/core/region_state.py) `BaseRegionState` (see Tier 1 item 1.6).

**Lines saved**: ~35 lines (5 lines × 7 files)

---

### B.2 Weight Initialization Methods

**Duplication**: Similar `_initialize_weights` pattern across regions.

**Example pattern**:
```python
def _initialize_weights(self) -> torch.Tensor:
    """Initialize region-specific synaptic weights."""
    # Pattern varies but basic structure is:
    if self.source_type == "sparse":
        return WeightInitializer.sparse_random(n_out, n_in, sparsity, device)
    else:
        return WeightInitializer.gaussian(n_out, n_in, mean, std, device)
```

**Locations**:
1. [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L524-L690)
2. [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py#L364-L400)
3. [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py#L374-L580)
4. [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py#L2247-L2300)

**Consolidation**: Standardize method name and signature, add base implementation in `NeuralRegion` for common patterns (see Tier 1 item 1.2).

**Lines saved**: Minimal (mostly consistency improvement, not direct duplication)

---

### B.3 Activity Threshold Checks

**Duplication**: Repeated `> 0.01` or `< 0.01` threshold checks for activity detection.

**Example pattern**:
```python
if activity > 0.01:  # Active
    # Process

if firing_rate < 0.01:  # Silent
    # Warn or skip
```

**Locations**:
1. [src/thalia/training/curriculum/stage_monitoring.py](src/thalia/training/curriculum/stage_monitoring.py#L184): Silent region detection
2. [src/thalia/regions/multisensory.py](src/thalia/regions/multisensory.py#L1018): Multi-modal activity
3. [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py#L1676): Encoding modulation check
4. [src/thalia/learning/homeostasis/synaptic_homeostasis.py](src/thalia/learning/homeostasis/synaptic_homeostasis.py#L110): Minimum activity config

**Consolidation**: Use constant from [src/thalia/constants/learning.py](src/thalia/constants/learning.py) (see Tier 1 items 1.1 and 1.8).

**Lines saved**: Minimal, but fixes inconsistency bug (0.001 vs 0.01 mismatch).

---

### B.4 Diagnostic Collection Boilerplate

**Duplication**: Basic diagnostic collection repeated in each region's `collect_diagnostics()`:
```python
def collect_diagnostics(self) -> Dict[str, Any]:
    return {
        "firing_rate": self._compute_firing_rate(),
        "sparsity": self._compute_sparsity(),
        "mean_weight": self.weights.mean().item(),
    }
```

**Observation**: [src/thalia/mixins/diagnostic_collector_mixin.py](src/thalia/mixins/diagnostic_collector_mixin.py) exists with helpers but not consistently used.

**Consolidation**: Encourage use of `DiagnosticCollectorMixin` methods:
- `_compute_firing_rate_diagnostic(spikes)`
- `_compute_sparsity_diagnostic(spikes)`
- `_compute_weight_statistics(weights)`

**Lines saved**: 3-5 lines per region × 8 regions = 24-40 lines

---

### B.5 Trace Update Logic

**Duplication**: Exponential trace decay repeated across learning strategies:
```python
self.pre_trace = self.pre_trace * decay_factor + pre_spikes
self.post_trace = self.post_trace * decay_factor + post_spikes
```

**Locations**:
1. STDP strategy (pre/post traces)
2. BCM strategy (running average)
3. Three-factor strategy (eligibility trace)
4. Various region-specific learning components

**Consolidation**: Use [src/thalia/components/synapses/traces.py](src/thalia/components/synapses/traces.py) `update_trace()` utility or [src/thalia/learning/eligibility/trace_manager.py](src/thalia/learning/eligibility/trace_manager.py) (see Tier 2 item 2.5).

**Lines saved**: 2-3 lines per strategy × 6 strategies = 12-18 lines

---

## Summary of Duplication Impact

| Duplication Type | Instances | Lines Duplicated | Consolidation Effort |
|------------------|-----------|------------------|---------------------|
| State reset | 7 | ~35 | Low (1 day) |
| Weight initialization | 4 | ~100 | Low (1 day) |
| Activity thresholds | 8+ | ~15 | Very low (hours) |
| Diagnostic collection | 8 | 24-40 | Low (1 day) |
| Trace updates | 6+ | 12-18 | Medium (2 days) |
| **Total** | **33+** | **186-208** | **5-6 days** |

**Note**: Line counts are approximations. Actual duplication is in logic/pattern rather than exact code.

---

## Antipatterns Detected

### A.1 God Object Pattern - AVOIDED ✅

**Analysis**: Early detection prevented god objects. Regions that started growing large (Striatum, Hippocampus) successfully extracted components:
- `StriatumLearningComponent`
- `StriatumHomeostasisComponent`
- `StriatumExplorationComponent`
- `HippocampusLearningComponent`

**Status**: **Not a problem** - good component extraction pattern established.

---

### A.2 Magic Numbers - DETECTED ⚠️

**Severity**: Medium

**Description**: Numeric literals for learning rates, thresholds, and noise levels scattered throughout code without named constants.

**Examples**: See Appendix B.3 (Activity Threshold Checks)

**Fix**: Tier 1 items 1.1, 1.3, 1.8 (consolidate to constants modules)

---

### A.3 Inconsistent Abstraction - MINOR ⚠️

**Severity**: Low

**Description**: Some pathways are simple (AxonalProjection), others complex (VisualPathway with encoding). Not clear when to use which.

**Fix**: Tier 2 item 2.4 (document pathway taxonomy)

---

### A.4 Violation of Biological Plausibility - NOT DETECTED ✅

**Analysis**: Excellent adherence to biological constraints:
- ✅ Binary spikes (not firing rates) in processing
- ✅ Local learning rules (no backpropagation detected)
- ✅ Weights at dendrites (not in axons)
- ✅ Conductance-based neurons (voltage-dependent currents)

**Status**: **Not a problem** - biological plausibility maintained.

---

### A.5 Premature Optimization - NOT DETECTED ✅

**Analysis**: Code is clean and readable, not over-engineered. Good balance of abstraction (mixins, registries) without unnecessary complexity.

**Status**: **Not a problem**

---

### A.6 Tight Coupling - LOW ⚠️

**Severity**: Low

**Description**: Some regions directly instantiate other regions or components instead of using dependency injection.

**Example**: Regions create `ConductanceLIF` neurons directly instead of accepting neuron instance.

**Fix**: Tier 2 item 2.2 (migrate to NeuronFactory for flexibility)

**Note**: This is a minor issue; current approach works fine for single-neuron-type regions.

---

## Pattern Improvements

### P.1 Learning Strategy Pattern - EXCELLENT ✅

**Current Pattern**: Pluggable learning strategies with registry.

**Status**: Successfully adopted, mature implementation. No improvement needed.

**Example**:
```python
strategy = create_strategy("three_factor", learning_rate=0.001)
new_weights, metrics = strategy.compute_update(weights, pre, post, dopamine)
```

**Benefit**: Eliminates code duplication for learning rules, enables easy experimentation.

---

### P.2 Mixin Architecture - EXCELLENT ✅

**Current Pattern**: Multiple inheritance with focused mixins.

**Status**: Well-designed, good separation of concerns. No improvement needed.

**Example**:
```python
class NeuralRegion(
    nn.Module,
    BrainComponentMixin,
    NeuromodulatorMixin,
    GrowthMixin,
    ResettableMixin,
    DiagnosticsMixin,
    StateLoadingMixin,
    LearningStrategyMixin,
):
```

**Benefit**: Each mixin has single responsibility, easily testable.

---

### P.3 State Dataclass Pattern - GOOD ✅

**Current Pattern**: Separate state dataclasses for each region.

**Opportunity**: Add helper methods for common state operations (see Tier 1 item 1.6).

**Before**:
```python
def reset(self) -> None:
    self.voltage.zero_()
    self.conductance_E.zero_()
    # ... repeated 7 times
```

**After**:
```python
def reset(self) -> None:
    self._reset_neuron_state()  # Common helper
    self.region_specific_state.zero_()
```

**Benefit**: Reduces duplication, centralizes common state management.

---

### P.4 Component Registry Pattern - EXCELLENT ✅

**Current Pattern**: Unified registry for regions, pathways, modules.

**Status**: Mature, well-documented. No improvement needed.

**Example**:
```python
@register_region("cortex", config_class=LayeredCortexConfig)
class LayeredCortex(NeuralRegion):
    pass

cortex = ComponentRegistry.create("region", "cortex", config)
```

**Benefit**: Dynamic component creation, plugin architecture support.

---

### P.5 Weight Initializer Registry - GOOD ✅

**Current Pattern**: Registry for weight initialization methods.

**Opportunity**: Ensure 100% adoption across codebase (currently ~80-90%).

**Status**: Mostly adopted, minor cleanup needed (see Tier 1 item 1.2).

**Example**:
```python
weights = WeightInitializer.sparse_random(n_out, n_in, sparsity=0.2, device=device)
```

**Benefit**: Consistent initialization, easier testing, clearer biological assumptions.

---

### P.6 Checkpoint Manager Pattern - GOOD, INCOMPLETE ⚠️

**Current Pattern**: Base checkpoint manager with region-specific implementations.

**Opportunity**: Add checkpoint managers for remaining regions (Cerebellum, Cortex, Thalamus).

**Status**: Partially adopted (3/8 regions), pattern is excellent (see Tier 2 item 2.7).

**Benefit**: Consistent save/load behavior, easier to maintain.

---

### P.7 Component Extraction Pattern - GOOD, EXPANDING ✅

**Current Pattern**: Extract large regions into focused components (Learning, Homeostasis, etc.).

**Status**: Successfully applied to Striatum and Hippocampus, should expand to Cortex and Thalamus (see Tier 2 item 2.1).

**Before** (2600 lines):
```python
class Striatum(NeuralRegion):
    def __init__(self):
        # 2600 lines of learning, homeostasis, exploration, etc.
```

**After** (split):
```python
class Striatum(NeuralRegion):  # 800 lines
    def __init__(self):
        self.learning = StriatumLearningComponent()
        self.homeostasis = StriatumHomeostasisComponent()

# Separate files:
# striatum/learning_component.py
# striatum/homeostasis_component.py
```

**Benefit**: Improved maintainability, easier testing, clearer separation of concerns.

---

### P.8 Multi-Source Input Pattern - EXCELLENT ✅

**Current Pattern**: Regions accept `Dict[str, torch.Tensor]` for multi-source inputs.

**Status**: Well-designed, biologically accurate (weights at dendrites).

**Example**:
```python
def forward(self, inputs: SourceOutputs) -> torch.Tensor:
    # inputs = {"cortex": spikes, "thalamus": spikes}
    for source, spikes in inputs.items():
        weighted_input += torch.matmul(spikes, self.synaptic_weights[source])
```

**Opportunity**: Add input validation (see Tier 2 item 2.9) to catch missing sources early.

**Benefit**: Natural multi-source integration, biologically plausible.

---

## Conclusion

**Strengths**:
- Excellent architectural patterns (mixins, registries, component extraction)
- Strong biological plausibility adherence
- Good separation of concerns in most areas
- Mature learning strategy system

**Improvement Areas**:
- Magic number consolidation (high priority)
- Component extraction for large files (medium priority)
- Missing checkpoint managers (low priority)
- Minor pattern adoption gaps (weight initialization, trace management)

**Overall Assessment**: **Solid architecture with minor technical debt**. Most issues are "polish" level (naming consistency, constant extraction) rather than fundamental design problems. No major antipatterns detected. Recommended approach is incremental improvement via Tier 1/2 items, with Tier 3 deferred as long-term research.

---

**Document Date**: January 26, 2026
**Codebase Version**: Current HEAD
**Reviewer**: GitHub Copilot (Claude Sonnet 4.5)
**Next Review**: After Phase 2 completion (estimated ~6 weeks)
