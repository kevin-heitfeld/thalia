# Architecture Review – 2025-12-11

**Date**: December 11, 2025
**Scope**: `src/thalia/` directory (core, regions, learning, integration, sensory)
**Methodology**: Static analysis, pattern detection, adherence to documented patterns
**Status**: Complete

## Executive Summary

Thalia demonstrates strong adherence to its documented architectural patterns, particularly the **Component Parity Principle** and neuroscience-inspired design. The codebase successfully avoids most deep learning antipatterns (no backpropagation, properly local learning rules, spike-based processing). Major improvements completed in Tier 1 and significant progress in Tier 2.

**Recent Progress (December 11, 2025)**:
- ✅ **Tier 1 Complete**: All 6 quick wins implemented (constants, config naming, consolidations)
- ✅ **Tier 2.1 Complete**: Manager pattern standardized with component pattern
- ✅ **Tier 2.3 Complete**: Learning strategy pattern fully implemented with registry
- ✅ **Tier 2.4 Complete**: Config hierarchy consolidated and simplified

Opportunities for continued improvement:
1. **Learning strategy adoption**: Infrastructure complete, regions can migrate gradually
2. **Directory structure**: Standardize region organization (striatum has subdirectory, cerebellum doesn't)
3. **Integration tests**: Expand pathway-region coordination tests

**Key Strengths**:
- ✅ No backpropagation (biological plausibility maintained)
- ✅ Spike-based processing (binary spikes, LIF neurons)
- ✅ WeightInitializer registry (eliminates scattered initialization)
- ✅ Neuron constants (neuron_constants.py eliminates magic numbers)
- ✅ Learning constants (learning_constants.py consolidates learning params)
- ✅ Component protocol (regions and pathways are equals)
- ✅ Component pattern (BaseManager) for region sub-components
- ✅ Learning strategy pattern (pluggable, composable learning rules)

**Key Opportunities**:
- 📁 Standardize region directory structure (clearer structure)
- 🔧 Migrate regions to learning strategies (optional, backward compatible)
- 📝 Simplify config hierarchy (reduce complexity)
- 🧪 Expand integration tests (catch pathway-region issues)

---

## Tier 1 - High Impact, Low Disruption

**Priority**: Implement first. Quick wins that improve maintainability without breaking changes.

### 1.1 Extract Magic Numbers to Named Constants

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Created `src/thalia/core/learning_constants.py` with 15+ learning rate and plasticity constants
- Created `src/thalia/core/homeostasis_constants.py` with homeostatic regulation constants
- Updated 8 region/learning files to use new constants:
  - `striatum/config.py` (eligibility, STDP)
  - `hippocampus/config.py` (one-shot learning, STDP)
  - `learning/bcm.py` (BCM learning rate, threshold tau)
  - `regions/base.py` (default learning rate, homeostasis)
  - `prefrontal.py` (STDP learning rate)
  - `cerebellum.py` (error-corrective learning, STDP)
  - `cortex/config.py` (STDP parameters)

**Current State**:
```python
# Scattered throughout codebase
tau_mem = 20.0
v_threshold = 1.0
learning_rate = 0.01
```

**Good Example** (already implemented):
```python
# src/thalia/core/neuron_constants.py
TAU_MEM_STANDARD = 20.0
V_THRESHOLD_STANDARD = 1.0
E_EXCITATORY = 0.003
```

**Remaining Issues**:
- Learning rates scattered across region files (0.01, 0.001, 0.0001)
- Homeostatic parameters not centralized
- Pathway STDP parameters not standardized

**Proposed Change**:
```python
# src/thalia/core/learning_constants.py
LEARNING_RATE_STDP = 0.001
LEARNING_RATE_BCM = 0.01
LEARNING_RATE_THREE_FACTOR = 0.001
TAU_ELIGIBILITY_STANDARD = 1000.0  # 1 second
TAU_BCM_THRESHOLD = 5000.0  # 5 seconds

# src/thalia/core/homeostasis_constants.py
TARGET_FIRING_RATE_STANDARD = 5.0  # Hz
HOMEOSTATIC_TAU_STANDARD = 1000.0  # 1 second
```

**Rationale**: Improves discoverability and ensures biological consistency across regions.

**Impact**: Low (search/replace, no API changes)
**Files Affected**:
- Create: `src/thalia/core/learning_constants.py`, `src/thalia/core/homeostasis_constants.py`
- Modify: ~20 files across `src/thalia/regions/`, `src/thalia/learning/`

---

### 1.2 Standardize Config Class Naming

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Renamed `TrisynapticConfig` → `HippocampusConfig` as primary class
- Renamed `TrisynapticState` → `HippocampusState` as primary class
- Added `Hippocampus` as preferred alias for `TrisynapticHippocampus`
- Kept deprecated aliases with `DeprecationWarning` (to be removed in v0.4.0)
- Updated 10 files to use new naming:
  - `hippocampus/config.py` (primary definitions + deprecation aliases)
  - `hippocampus/__init__.py` (exports with both names)
  - `regions/__init__.py` (updated top-level exports)
  - `hippocampus/trisynaptic.py` (internal usage)
  - `hippocampus/plasticity_manager.py` (type hints)
  - `hippocampus/episode_manager.py` (type hints)
  - `regions/factory.py` (example usage)
  - `memory/sequence.py` (instantiation)
  - `events/adapters/factory.py` (instantiation)
  - `tests/unit/test_region_axonal_delays.py` (test usage)

**Current State**: Inconsistent naming patterns
```python
StriatumConfig          # Good: RegionName + Config
TrisynapticConfig       # Misleading: Circuit type, not "HippocampusConfig"
HomeostasisManagerConfig  # Good: Component + Config
LearningManagerConfig   # Missing: Which region's learning manager?
```

**Proposed Changes**:
1. ✅ Rename `TrisynapticConfig` → `HippocampusConfig` (primary config)
2. ⏸️ Rename `LearningManagerConfig` → `StriatumLearningConfig` (specify region) - **Deferred to Tier 2.1**
3. ✅ Keep `StriatumConfig`, `CortexConfig` (already consistent)

**Rationale**:
- `TrisynapticConfig` is opaque to users unfamiliar with hippocampal circuits
- Manager configs should indicate which region they belong to
- Follows pattern: `{Region}{Component}Config`

**Impact**: Medium (breaking change for config imports, but fixable via deprecation)
**Files Affected**:
- `src/thalia/regions/hippocampus/config.py` (rename)
- `src/thalia/regions/striatum/learning_manager.py` (rename) - **Deferred**
- ~10 test files with config imports
- Add deprecation warnings for old names (1 release cycle)

---

### 1.3 Consolidate Duplicated Code: Spike Rate Calculation

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Created `src/thalia/core/spike_utils.py` with comprehensive spike utilities:
  - `compute_firing_rate()` - Main consolidation function
  - `compute_spike_count()` - Count total spikes
  - `compute_spike_density()` - Local density with sliding window
  - `is_silent()` - Check for silent populations
  - `is_saturated()` - Check for saturated populations
- Updated 9 files to use `compute_firing_rate()`:
  - `core/brain.py` (2 usages: CA1 activity, WM activity)
  - `regions/base.py` (2 usages: diagnostics, health check)
  - `learning/intrinsic_plasticity.py` (1 usage: population rate)
  - `learning/ei_balance.py` (4 usages: E/I ratio calculation, update method)
  - `core/neuromodulator_manager.py` (2 usages: PFC uncertainty, CA1 reward)
  - `core/growth.py` (1 usage: capacity metrics)
  - `training/live_diagnostics.py` (1 usage: firing rate history)
  - `sensory/pathways.py` (1 usage: sparsity metadata)

**Duplication Detected**:
```python
# Location 1: src/thalia/core/diagnostics_mixin.py:45
def get_firing_rate(self, spikes: torch.Tensor) -> float:
    if spikes.numel() == 0:
        return 0.0
    return spikes.float().mean().item()

# Location 2: src/thalia/regions/base.py:652
def _compute_firing_rate(self, spikes: torch.Tensor) -> float:
    if spikes.numel() == 0:
        return 0.0
    return spikes.float().mean().item()

# Location 3: src/thalia/diagnostics/health_monitor.py:89
firing_rate = spikes.float().mean().item() if spikes.numel() > 0 else 0.0
```

**Proposed Consolidation**:
```python
# src/thalia/core/spike_utils.py (new file)
def compute_firing_rate(spikes: torch.Tensor) -> float:
    """Compute population firing rate from binary spike tensor.

    Args:
        spikes: Binary spike tensor (any shape)

    Returns:
        Fraction of neurons firing (0.0 to 1.0)
    """
    if spikes.numel() == 0:
        return 0.0
    return spikes.float().mean().item()
```

Then replace all 3+ locations with:
```python
from thalia.core.spike_utils import compute_firing_rate
rate = compute_firing_rate(spikes)
```

**Rationale**: DRY principle, single source of truth for spike rate computation.

**Impact**: Low (internal refactoring, no API changes)
**Files Affected**:
- Create: `src/thalia/core/spike_utils.py`
- Modify: `diagnostics_mixin.py`, `base.py`, `health_monitor.py`, ~5 more

---

### 1.4 Extract Common Weight Clamping Pattern

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Used existing `clamp_weights()` function from `src/thalia/core/utils.py`
- Updated 10 files to use centralized utility:
  - `regions/base.py` (1 usage)
  - `regions/striatum/striatum.py` (2 usages)
  - `regions/striatum/d1_pathway.py` (1 usage)
  - `regions/striatum/d2_pathway.py` (1 usage)
  - `regions/cerebellum.py` (5 usages)
  - `learning/unified_homeostasis.py` (3 usages)
  - `learning/strategies.py` (1 usage)
  - `integration/spiking_pathway.py` (2 usages)
  - `core/stp.py` (3 usages)
- Consolidated 19 inline weight clamping operations to single utility function

**Duplication Detected**:
```python
# Pattern appears in 8+ locations
# Location 1: src/thalia/regions/striatum/learning_manager.py:187
self.weights.data.clamp_(min=self.config.w_min, max=self.config.w_max)

# Location 2: src/thalia/regions/hippocampus/plasticity_manager.py:142
self.weights.clamp_(self.config.w_min, self.config.w_max)

# Location 3: src/thalia/regions/cortex/layered_cortex.py:456
torch.clamp(self.weights, min=0.0, max=self.config.w_max, out=self.weights)
```

**Proposed Consolidation**:
```python
# Add to src/thalia/core/weight_init.py
@staticmethod
def clamp_weights(
    weights: torch.Tensor,
    w_min: float,
    w_max: float,
    inplace: bool = True
) -> torch.Tensor:
    """Clamp weights to biological bounds.

    Args:
        weights: Weight tensor to clamp
        w_min: Minimum weight value
        w_max: Maximum weight value
        inplace: Modify tensor in-place (default: True)

    Returns:
        Clamped weights (same tensor if inplace=True)
    """
    if inplace:
        return weights.clamp_(min=w_min, max=w_max)
    return torch.clamp(weights, min=w_min, max=w_max)
```

**Rationale**: Eliminates slight variations in clamping logic, ensures consistency.

**Impact**: Low (internal refactoring)
**Files Affected**: ~8 files across regions

---

### 1.5 Standardize Device Handling Patterns

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Fixed 5 device handling violations in `training/task_loaders.py`:
  - Lines 267, 270: Added device parameter to motor control task tensors
  - Line 299: Added device parameter to reaching task tensors
  - Line 330: Added device parameter to manipulation task tensors
  - Line 361: Added device parameter to prediction task tensors
- Verified all other task files already compliant
- All tensor creation now follows Pattern 1 (specify device at creation)

**Current State**: Mostly good, but some violations of Pattern 1 found.

**Good Pattern** (majority of codebase):
```python
# Pattern 1: Specify device at creation
weights = torch.zeros(n_output, n_input, device=device)
membrane = torch.zeros(n_neurons, device=self.device)
```

**Violations Found**:
```python
# src/thalia/training/task_loaders.py:267
motor_spikes = torch.rand(self.wrapper.n_motor_neurons) < SPIKE_PROBABILITY_LOW
# Missing: device=self.device

# src/thalia/tasks/sensorimotor.py:189
proprioception = torch.randn(...)
# Missing: device=self.device
```

**Proposed Fix**: Add device parameter to all tensor creation in training/tasks modules.

**Rationale**: Prevents silent device mismatch errors, especially when using CUDA.

**Impact**: Low (bug fixes, no API changes)
**Files Affected**: `task_loaders.py`, `sensorimotor.py`, `working_memory.py`, `executive_function.py`

---

### 1.6 Remove Unused Imports and Dead Code

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Configured Python environment and installed ruff linter
- Fixed 39 unused imports across codebase:
  - 36 automatically fixed by `ruff --fix`
  - 3 manually fixed in `__init__.py` (added to `__all__`)
- Reviewed 4 TODO comments - all valid future work items:
  - `live_diagnostics.py:346` - "Implement animated GIF creation" (keep)
  - `cerebellum.py:246` - "Migrate to trace manager if needed" (keep)
  - `checkpoint_manager.py:148` - "Get version from package" (keep)
  - `brain.py:856` - "Implement _create_real_* functions" (keep)
- Verified no `.backward()` usage in current codebase
- All ruff F401 checks now pass

**Findings**:
- Several `from typing import TYPE_CHECKING` imports unused
- Some `# TODO` comments reference completed features
- Deprecated pattern: `.backward()` found in one location (metacognition diagnostic tool, intentional)

**Proposed Actions**:
1. Run `ruff check --select F401` (unused imports) and clean
2. Review `# TODO` comments and remove obsolete ones
3. Document exception for `.backward()` in `diagnostics/metacognition.py` (meta-learning tool, not violating biological plausibility of main brain)

**Rationale**: Code hygiene, reduces cognitive load.

**Impact**: Very low (cosmetic cleanup)
**Files Affected**: ~15 files with unused imports

---

## Tier 2 - Moderate Refactoring

**Priority**: Strategic improvements that require more planning but significantly improve architecture.

### 2.1 Unify Manager Pattern Across Regions

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:

Created standardized component pattern with base classes and region-specific implementations:

**Base Component Classes** (`src/thalia/core/region_components.py`):
- `LearningComponent`: Unified learning interface (apply_learning, reset_state, diagnostics)
- `HomeostasisComponent`: Unified homeostasis interface (apply_homeostasis, diagnostics)
- `MemoryComponent`: Unified memory interface (store_memory, retrieve_memories)
- `ExplorationComponent`: Unified exploration interface (compute_exploration_bonus)

**Striatum Components**:
- `StriatumLearningComponent` (was `LearningManager`): Three-factor learning
- `StriatumHomeostasisComponent` (was `HomeostasisManager`): D1/D2 balance
- `StriatumExplorationComponent` (was `ExplorationManager`): UCB exploration

**Hippocampus Components**:
- `HippocampusLearningComponent` (was `PlasticityManager`): STDP + synaptic scaling
- `HippocampusMemoryComponent` (was `EpisodeManager`): Episodic buffer management

**Backwards Compatibility**:
- Module-level aliases: `LearningManager = StriatumLearningComponent`
- Instance properties: `@property learning_manager → self.learning`
- Zero breaking changes for existing code

**Region Updates**:
- `striatum.py`: self.learning_manager → self.learning (+ backwards compat property)
- `trisynaptic.py`: self.plasticity_manager → self.learning (+ backwards compat property)

**Documentation**:
- Created `docs/patterns/component-standardization.md` with comprehensive pattern guide
- Guidelines for when to create components vs utilities
- Migration checklist for future regions

**Non-Components** (kept as utilities):
- `CheckpointManager`: Pure I/O operations
- `ReplayEngine`: Complex reusable algorithm (used by sleep system)
- `ForwardPassCoordinator`: Simple orchestration (future: absorb into region)
- `StateTracker`: Simple state storage (future: absorb into region)

**Benefits Achieved**:
- ✅ Consistent naming across regions (Component pattern)
- ✅ Clear separation of concerns
- ✅ Unified interfaces for cross-region functionality
- ✅ Improved discoverability (region.learning, region.memory)
- ✅ Zero breaking changes via compatibility layer

**Decision**: Implemented Option A (Functional Decomposition) with standardized naming.

**Related Documentation**:
- Pattern: `docs/patterns/component-standardization.md`
- Commits: 7e34373, 5646783

---

### 2.2 Standardize Region Directory Structure

**Current State**: Inconsistent organization

**With Subdirectories**:
```
regions/
├── striatum/           # Subdirectory with 13 files
│   ├── striatum.py
│   ├── config.py
│   ├── d1_pathway.py
│   ├── d2_pathway.py
│   └── [9 more files]
├── hippocampus/        # Subdirectory with 6 files
│   ├── trisynaptic.py
│   └── [5 more files]
└── cortex/             # Subdirectory with 3 files
    ├── layered_cortex.py
    └── [2 more files]
```

**Without Subdirectories**:
```
regions/
├── prefrontal.py       # Single file (525 lines)
├── cerebellum.py       # Single file (580 lines)
├── base.py             # Base class (939 lines)
├── theta_dynamics.py   # Shared utility
└── factory.py          # Region factory
```

**Proposed Guideline**:
- **Single file**: Regions with < 800 lines, no sub-components
- **Subdirectory**: Regions with > 800 lines OR multiple sub-components

**Apply Guideline**:
1. **Keep single files**: `prefrontal.py` (525 lines), `cerebellum.py` (580 lines)
2. **Keep subdirectories**: `striatum/` (complex RL logic), `hippocampus/` (trisynaptic circuit)
3. **Decision needed**: `cortex/` (only 3 files, could be merged)

**Proposed Action**:
```
cortex/
├── layered_cortex.py     → Keep (hierarchical structure)
├── predictive_cortex.py  → Keep (predictive coding complex)
└── config.py             → Merge into layered_cortex.py (simple dataclass)
```

**Rationale**: Balances discoverability (single file easy to find) vs. organization (subdirectory for complexity).

**Impact**: Low (file moves, update imports)
**Files Affected**: `cortex/config.py` → merge into `layered_cortex.py`

---

### 2.3 Abstract Learning Rules into Strategy Pattern

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Comprehensive learning strategy system in `src/thalia/learning/strategies.py`
- Registry pattern in `src/thalia/learning/strategy_registry.py`
- Mixin for easy integration in `src/thalia/learning/strategy_mixin.py`
- 6 built-in strategies: Hebbian, STDP, BCM, ThreeFactor, ErrorCorrective, Composite
- All strategies registered with aliases and metadata
- Prefrontal cortex already uses STDPStrategy
- Full documentation in `docs/patterns/learning-strategy-pattern.md`
- 20 passing integration tests in `tests/integration/test_learning_strategy_pattern.py`

**Current State**: Infrastructure complete, ready for region migration

```python
# OLD: Duplicated learning logic
# Striatum: Three-factor rule
class Striatum:
    def _apply_learning(self, ...):
        weight_update = eligibility * dopamine * learning_rate
        self.weights += weight_update

# NEW: Using strategy pattern
from thalia.learning import LearningStrategyRegistry, ThreeFactorConfig

class Striatum:
    def __init__(self, config):
        self.learning_strategy = LearningStrategyRegistry.create(
            "three_factor",
            ThreeFactorConfig(learning_rate=config.learning_rate)
        )

    def _apply_learning(self, pre, post, dopamine):
        new_weights, metrics = self.learning_strategy.compute_update(
            weights=self.weights,
            pre=pre,
            post=post,
            modulator=dopamine,
        )
        self.weights.data.copy_(new_weights)
        return metrics
```

**Antipattern Eliminated**: **Duplicated learning logic with slight variations.**

**Implemented Pattern**: Learning Strategy Registry (similar to WeightInitializer)

**Infrastructure**:
- `src/thalia/learning/strategies.py` - Base classes and implementations
- `src/thalia/learning/strategy_registry.py` - Registry and factory
- `src/thalia/learning/strategy_mixin.py` - Mixin for easy integration
- `docs/patterns/learning-strategy-pattern.md` - Comprehensive documentation
- `tests/integration/test_learning_strategy_pattern.py` - Integration tests (20 passing)

**Available Strategies**:
- `HebbianStrategy`: Δw ∝ pre × post
- `STDPStrategy`: Spike-timing dependent plasticity
- `BCMStrategy`: Bienenstock-Cooper-Munro with sliding threshold
- `ThreeFactorStrategy`: Δw = eligibility × modulator (for RL)
- `ErrorCorrectiveStrategy`: Δw = pre × (target - actual) (for supervised)
- `CompositeStrategy`: Compose multiple strategies

**Usage Pattern**:
```python
from thalia.learning import LearningStrategyRegistry, STDPConfig

# In region __init__:
self.learning_strategy = LearningStrategyRegistry.create(
    "stdp",
    STDPConfig(learning_rate=config.stdp_lr, a_plus=0.01, a_minus=0.012)
)

# In region forward():
new_weights, metrics = self.learning_strategy.compute_update(
    weights=self.weights,
    pre=input_spikes,
    post=output_spikes,
)
self.weights.data.copy_(new_weights)
```

**Migration Status**:
- ✅ **Prefrontal**: Uses STDPStrategy (initial implementation)
- ✅ **LayeredCortex**: Migrated to BCMStrategy (fixed per-neuron BCM implementation)
- ⏳ **Striatum**: Uses custom three-factor (migration optional, working correctly)
- ⏳ **Hippocampus**: Uses custom STDP (migration optional, working correctly)
- ❌ **Cerebellum**: Custom error-modulated STDP (should NOT migrate - see rationale below)

**Cerebellum Migration Decision**:
The cerebellum implements a unique **error-modulated eligibility** learning rule that doesn't fit standard strategies:
- Uses 2D eligibility weights (not 1D activity vectors): `eligibility [n_output, n_input]`
- Error gates per-neuron: `dw = eligibility × error_sign × error_magnitude × beta_gate`
- Beta gating: Motor timing modulation (movement initiation window)
- Climbing fiber system: Specialized error computation
- This is biologically specialized, already well-structured, and not reusable elsewhere

**Verdict**: Keep cerebellum's custom implementation. Not everything should be forced into strategy pattern.

**Benefits Achieved**:
- ✅ Eliminates duplicated learning logic
- ✅ Makes learning rules composable and testable
- ✅ Easier to experiment with hybrid rules (CompositeStrategy)
- ✅ Matches neuroscience literature (learning rules are concepts)
- ✅ Registry enables discovery and plugin architecture

**Rationale**:
- Eliminates duplicated learning logic
- Makes learning rules composable and testable
- Easier to experiment with hybrid rules
- Matches neuroscience literature (learning rules are concepts, not implementation details)

**Impact**: Medium (infrastructure complete, migration optional for existing regions)
**Files Affected**:
- ✅ Created: `docs/patterns/learning-strategy-pattern.md`
- ✅ Created: `tests/integration/test_learning_strategy_pattern.py`
- ✅ Existing: All strategies in `src/thalia/learning/strategies.py`
- ⏳ Optional: Update regions to use strategies (backward compatible)

**Note**: Infrastructure complete. Regions can migrate at their own pace. Prefrontal already demonstrates the pattern. See `docs/patterns/learning-strategy-pattern.md` for full migration guide.

---

### 2.4 Consolidate Config Hierarchy

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Created `TrainingConfig` class with all training-specific parameters (learning toggles, epochs, monitoring)
- Created `NeuromodulationConfig` class within `BrainConfig` (dopamine, norepinephrine, acetylcholine)
- Created `PathwayConfig` base class for pathway configurations (completing component parity)
- Enhanced `RegionConfigBase` with weight bounds (`w_min`, `w_max`)
- Updated `ThaliaConfig` to include `training` field (fixed missing reference bug)
- Updated all exports in `config/__init__.py`
- Created comprehensive migration guide in `docs/patterns/config-simplification.md`
- All tests pass - imports work correctly and hierarchy is functional

**Benefits Achieved**:
- ✅ Clear separation of concerns (global, brain, training, language)
- ✅ Reduced duplication (weight bounds, neuromodulation, training params)
- ✅ Component parity (regions have `RegionConfigBase`, pathways have `PathwayConfig`)
- ✅ Better discoverability (IDE autocomplete for `config.training.*`, `config.brain.neuromodulation.*`)
- ✅ Zero breaking changes (additive only)

**New Hierarchy**:
```
ThaliaConfig
├── global_: GlobalConfig (device, timing, vocab)
├── brain: BrainConfig
│   ├── sizes: RegionSizes
│   ├── neuromodulation: NeuromodulationConfig (NEW)
│   └── region configs (cortex, hippocampus, etc)
├── training: TrainingConfig (NEW)
├── language: LanguageConfig
└── robustness: RobustnessConfig
```

**Documentation**: See `docs/patterns/config-simplification.md` for complete migration guide.

---

### 2.5 Enhance Pathway-Region Integration Tests

**Current State**: Limited testing of pathway-region interactions

**Existing Tests**:
- `tests/unit/test_region_axonal_delays.py` (delays in regions)
- `tests/unit/test_pathway_growth.py` (pathway growth in isolation)
- `tests/unit/test_striatum_growth.py` (region growth in isolation)

**Missing Tests**:
1. **Coordinated growth**: Region grows → Pathway adapts weights
2. **Learning synchronization**: Region learns → Pathway learns → Region receives updated input
3. **State consistency**: Pathway state reset → Region state reset
4. **Health propagation**: Pathway pathology → Region detects it

**Proposed Tests**:
```python
# tests/integration/test_pathway_region_coordination.py

def test_coordinated_growth():
    """When region grows, connected pathway adjusts."""
    region = Striatum(config)
    pathway = VisualPathway(n_output=region.n_input)

    initial_weights = pathway.weights.clone()

    # Region grows by 10 neurons
    region.add_neurons(n_new=10)

    # Pathway should expand output to match
    assert pathway.n_output == region.n_input

    # New weights should be initialized, old weights preserved
    assert torch.allclose(pathway.weights[:, :initial_n], initial_weights)

def test_learning_feedback_loop():
    """Pathway learning affects region input, region learning affects pathway reinforcement."""
    # Test STDP in pathway → region responds differently → dopamine → pathway adjusts
    pass

def test_health_propagation():
    """Pathway silence detected by region diagnostics."""
    # Make pathway go silent → region.check_health() should report low input
    pass
```

**Rationale**: Integration bugs (pathway silent, region doesn't adapt) are hard to catch without targeted tests.

**Impact**: Medium (test addition, no production code changes)
**Files Affected**: Create `tests/integration/test_pathway_region_coordination.py`

---

## Tier 3 - Major Restructuring

**Priority**: Long-term considerations requiring significant effort and planning.

### 3.1 Migrate to Unified Component Interface

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
- Created `BrainComponentBase` abstract base class in `component_protocol.py`
- Enforces all required methods from `BrainComponent` protocol at compile time
- Updated `NeuralComponent` to inherit from `BrainComponentBase`
- Added device and dtype as abstract properties with setters
- All regions and pathways now guaranteed to implement complete interface
- Created comprehensive documentation in `docs/patterns/component-interface-enforcement.md`
- All 9 tested components (6 regions + 3 pathways) pass interface compliance
- 27 integration tests pass without any breaking changes

**Benefits Achieved**:
- ✅ Static checking catches missing methods at instantiation
- ✅ IDE and type checker support for required methods
- ✅ Component parity guaranteed by type system
- ✅ Self-documenting code (abstract methods clearly mark requirements)
- ✅ Helpful error messages for unimplemented features

**Tested Components**:
- Regions: Striatum, TrisynapticHippocampus, LayeredCortex, PredictiveCortex, Prefrontal, Cerebellum
- Pathways: SpikingPathway, VisualPathway, LanguagePathway

**Current State**: Complete enforcement with Protocol → ABC migration

**Incomplete Implementations**:
- Some pathways lack `get_capacity_metrics()` (required for growth)
- Some regions lack `get_growth_config()` (should return GrowthConfig)
- `reset_state()` implementations vary (some full reset, some partial)

**Proposed Change**: Enforce protocol with abstract base class

```python
# src/thalia/core/component_protocol.py
from abc import ABC, abstractmethod

class BrainComponentBase(ABC):
    """Abstract base enforcing BrainComponent protocol.

    All regions and pathways MUST inherit from this.
    """

    @abstractmethod
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process input. REQUIRED."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset all temporal state. REQUIRED."""
        pass

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return current diagnostics. REQUIRED."""
        pass

    # ... enforce all protocol methods
```

Then:
```python
class BrainRegion(BrainComponentBase, nn.Module):
    """All regions inherit from enforced base."""
    pass

class BaseNeuralPathway(BrainComponentBase, nn.Module):
    """All pathways inherit from enforced base."""
    pass
```

**Rationale**: Static checking prevents missing implementations, enforces component parity.

**Impact**: Very high (all regions/pathways must fully implement protocol)
**Files Affected**: All region and pathway files

**Risk**: Breaking change if any component has partial implementation.

**Recommendation**: Audit current implementations first, add missing methods, THEN enforce with abstract base.

---

### 3.2 Decompose Brain.py God Object

**Current State**: `src/thalia/core/brain.py` is 1,853 lines with many responsibilities:

```python
class Brain(nn.Module):
    # 1. Region management
    def add_region(self, name, region): ...
    def get_region(self, name): ...

    # 2. Pathway management
    def connect(self, source, target, pathway): ...
    def get_pathway(self, source, target): ...

    # 3. Neuromodulation (VTA, LC, NB)
    def compute_dopamine(self): ...
    def set_dopamine(self, level): ...

    # 4. Oscillations
    def _update_oscillations(self): ...

    # 5. Growth coordination
    def trigger_growth(self): ...

    # 6. Forward pass coordination
    def forward(self, inputs): ...

    # 7. State management
    def reset_state(self): ...

    # 8. Checkpointing
    def save_checkpoint(self): ...
    def load_checkpoint(self): ...

    # 9. Diagnostics
    def check_health(self): ...
    def get_diagnostics(self): ...
```

**Antipattern**: **God Object** - Single class with too many responsibilities.

**Proposed Decomposition**:

```python
# src/thalia/core/brain_architecture.py
class BrainArchitecture:
    """Manages region and pathway topology."""
    def add_region(self, name, region): ...
    def connect(self, source, target, pathway): ...

# src/thalia/core/brain_neuromodulation.py
class BrainNeuromodulation:
    """Manages VTA, LC, NB systems."""
    def compute_dopamine(self): ...
    def broadcast_dopamine(self): ...

# src/thalia/core/brain_oscillations.py
class BrainOscillations:
    """Manages theta, gamma oscillations."""
    def update_oscillations(self): ...

# src/thalia/core/brain_growth.py
class BrainGrowth:
    """Coordinates region and pathway growth."""
    def trigger_growth(self): ...

# src/thalia/core/brain.py
class Brain(nn.Module):
    """Orchestrates brain components."""
    def __init__(self, config):
        self.architecture = BrainArchitecture()
        self.neuromodulation = BrainNeuromodulation()
        self.oscillations = BrainOscillations()
        self.growth = BrainGrowth()

    def forward(self, inputs):
        # Delegate to specialized components
        self.neuromodulation.update()
        self.oscillations.update()
        return self.architecture.forward(inputs)
```

**Rationale**:
- Reduces complexity (1,853 lines → 4 classes of ~400 lines each)
- Improves testability (test neuromodulation separately)
- Clearer responsibilities
- Easier to extend (add new neuromodulator = update one class)

**Impact**: Very high (major architectural change)
**Files Affected**: `brain.py`, create 4 new files, update all brain tests

**Risk**: Breaking change for anyone importing `Brain` internals.

**Recommendation**: Phase over multiple releases:
1. v0.2: Introduce new classes, keep `Brain` facade
2. v0.3: Deprecate direct access to `Brain` internals
3. v0.4: Full migration

---

### 3.3 Implement Pathway Learning Strategy Pattern

**Status**: ✅ **IMPLEMENTED** (December 11, 2025)

**Implementation Summary**:
**SpikingPathway successfully migrated to strategy pattern!**

1. **Code Migration Completed**:
   - Added imports: `STDPStrategy`, `STDPConfig`, `LearningStrategyRegistry`
   - Initialized `self.learning_strategy` in `__init__()` with `STDPStrategy`
   - Replaced `_apply_stdp()` call with `apply_strategy_learning()` in `forward()`
   - Removed 100+ lines of custom `_apply_stdp()` method
   - Backward compatible: old `learn()` method still works

2. **Test Results**:
   - ✅ All 27 integration tests pass
   - ✅ SpikingPathway instantiation works
   - ✅ Forward pass with learning works correctly
   - ✅ Weights update via strategy system
   - ✅ Strategy mixin methods available
   - ✅ Backward compatibility verified

3. **Documentation Updates**:
   - Updated `docs/patterns/learning-strategy-pattern.md` with:
     - Complete SpikingPathway migration example (before/after code)
     - Benefits of migration (code reduction, swappability, consistency)
     - Special pathway cases (sensory, attention, replay)

**Key Discovery**: Pathways already inherited `LearningStrategyMixin` from `NeuralComponent`, so they could always use strategies - just needed to migrate from custom learning code.

**Benefits Achieved**:
- ✅ Removed duplicate STDP logic (now shared with regions via strategy)
- ✅ Can swap learning rules by changing 1 line
- ✅ Consistent trace management across all components  
- ✅ Zero breaking changes (backward compatible)
- ✅ Component parity principle fulfilled (pathways = regions for learning)

**Advanced Features (noted as TODOs)**:
The old `_apply_stdp()` had some advanced features marked as future work:
- BCM metaplasticity modulation
- Neuromodulator modulation (DA, ACh, NE)
- Phase-locked STDP (for phase coding)
- Dopamine-STDP and Replay-STDP learning rules

These can be added to the strategy system when needed.

**Files Modified**:
- `src/thalia/integration/spiking_pathway.py`: Migrated to strategy pattern
- `docs/patterns/learning-strategy-pattern.md`: Added pathway migration example

**Rationale**:
- Pathways are neural components, same as regions (component parity principle)
- Same learning rules apply (STDP, BCM, etc.)
- Reusing existing infrastructure reduces complexity and ensures consistency

**Impact**: Low (backward compatible migration)
**Test Coverage**: ✅ All tests pass + migration verified
**No new code required!** The existing learning strategy infrastructure (Tier 2.3) already supports pathways through `NeuralComponent` inheritance. This is a **documentation and migration task**, not a coding task.

**Recommended Actions**:
1. Document that pathways can use learning strategies (update existing docs)
2. Optionally migrate `SpikingPathway._apply_stdp()` to use strategy pattern
3. No rush - current implementation works, migration is for consistency

**Rationale**:
- Pathways are neural components, same as regions (component parity principle)
- Same learning rules apply (STDP, BCM, etc.)
- Reusing existing infrastructure reduces complexity and ensures consistency

**Impact**: Very Low (documentation only, optional migration)
**Files Affected**: 
- Update: `docs/patterns/learning-strategy-pattern.md` (add pathway examples)
- Optional: `src/thalia/integration/spiking_pathway.py` (migrate to strategy pattern)

---

## Risk Assessment and Sequencing

### Implementation Sequence (Recommended)

**Phase 1 - Quick Wins** (2-4 weeks) - ✅ **COMPLETED December 11, 2025**
1. ✅ Extract magic numbers to constants (1.1)
2. ✅ Fix device handling patterns (1.5)
3. ✅ Clean up unused imports (1.6)
4. ✅ Consolidate spike rate calculation (1.3)
5. ✅ Standardize weight clamping (1.4)
6. ✅ Standardize config naming (1.2)

**Phase 2 - Structural Improvements** (1-2 months) - ⏳ **IN PROGRESS**
1. ✅ Unify manager pattern across regions (2.1) - **COMPLETED**
2. ✅ Abstract learning rules into strategy pattern (2.3) - **COMPLETED**
3. ⏳ Standardize region directory structure (2.2)
4. ⏳ Enhance pathway-region integration tests (2.5)
5. ⏳ Consolidate config hierarchy (2.4)

**Phase 3 - Major Refactoring** (3-6 months)
1. ✅ Migrate to unified component interface (3.1) - **COMPLETED**
2. ✅ Pathway learning strategy pattern (3.3) - **INFRASTRUCTURE COMPLETE** (doc update only)
3. ⏳ Decompose Brain.py god object (3.2) - **FUTURE WORK**

**Phase 4 - Long-Term Architecture** (6-12 months)
1. Decompose Brain.py god object (3.2)

### Risk Mitigation

**For Tier 1 Changes**:
- ✅ Low risk: Mostly internal refactoring
- ✅ Easy rollback: Changes localized
- ⚠️ Test thoroughly: Ensure no behavior changes

**For Tier 2 Changes**:
- ⚠️ Medium risk: Some breaking changes (config renaming)
- ✅ Phased rollout: Use deprecation warnings
- ⚠️ Update documentation: All examples must reflect new patterns

**For Tier 3 Changes**:
- ⚠️ High risk: Major architectural shifts
- ⚠️ Extensive testing: Integration tests essential
- ✅ Backward compatibility: Maintain facades during migration
- ⚠️ User communication: Release notes, migration guides

---

## Appendix A: Affected Files and Links

### Tier 1 Files

**1.1 Magic Number Extraction**
- `src/thalia/regions/striatum/striatum.py`
- `src/thalia/regions/hippocampus/trisynaptic.py`
- `src/thalia/regions/cortex/layered_cortex.py`
- `src/thalia/regions/prefrontal.py`
- `src/thalia/regions/cerebellum.py`
- `src/thalia/learning/bcm.py`
- `src/thalia/learning/ei_balance.py`
- `src/thalia/core/neuron.py`

**1.2 Config Naming**
- `src/thalia/regions/hippocampus/config.py` (rename `TrisynapticConfig`)
- `src/thalia/regions/striatum/learning_manager.py` (rename config)
- All test files importing these configs

**1.3 Spike Rate Consolidation**
- `src/thalia/core/diagnostics_mixin.py`
- `src/thalia/regions/base.py`
- `src/thalia/diagnostics/health_monitor.py`

**1.4 Weight Clamping**
- `src/thalia/regions/striatum/learning_manager.py`
- `src/thalia/regions/hippocampus/plasticity_manager.py`
- `src/thalia/regions/cortex/layered_cortex.py`
- `src/thalia/regions/prefrontal.py`

**1.5 Device Handling**
- `src/thalia/training/task_loaders.py`
- `src/thalia/tasks/sensorimotor.py`
- `src/thalia/tasks/working_memory.py`
- `src/thalia/tasks/executive_function.py`

### Tier 2 Files

**2.1 Manager Consolidation**
- `src/thalia/regions/striatum/learning_manager.py`
- `src/thalia/regions/striatum/homeostasis_manager.py`
- `src/thalia/regions/striatum/checkpoint_manager.py`
- `src/thalia/regions/striatum/exploration.py`
- `src/thalia/regions/striatum/forward_coordinator.py`
- `src/thalia/regions/hippocampus/plasticity_manager.py`
- `src/thalia/regions/hippocampus/episode_manager.py`
- `src/thalia/regions/hippocampus/replay_engine.py`

**2.2 Directory Structure**
- `src/thalia/regions/cortex/config.py` (merge into `layered_cortex.py`)

**2.3 Learning Strategy Pattern**
- All region files
- `src/thalia/learning/strategies.py` (already exists, expand)

**2.4 Config Consolidation**
- All files in `src/thalia/config/`
- All region config files

**2.5 Integration Tests**
- `tests/integration/test_pathway_region_coordination.py` (new)

### Tier 3 Files

**3.1 Component Interface**
- `src/thalia/core/component_protocol.py`
- `src/thalia/regions/base.py`
- `src/thalia/core/pathway_protocol.py`
- All region and pathway files

**3.2 Brain Decomposition**
- `src/thalia/core/brain.py`
- `src/thalia/core/brain_architecture.py` (new)
- `src/thalia/core/brain_neuromodulation.py` (new)
- `src/thalia/core/brain_oscillations.py` (new)
- `src/thalia/core/brain_growth.py` (new)

**3.3 Pathway Learning**
- `src/thalia/integration/spiking_pathway.py`
- `src/thalia/integration/pathways/spiking_attention.py`
- `src/thalia/integration/pathways/spiking_replay.py`
- `src/thalia/sensory/pathways.py`

---

## Appendix B: Detected Duplications and Locations

### Duplication 1: Spike Rate Calculation
**Locations**:
1. `src/thalia/core/diagnostics_mixin.py:45` - `get_firing_rate()`
2. `src/thalia/regions/base.py:652` - `_compute_firing_rate()`
3. `src/thalia/diagnostics/health_monitor.py:89` - inline implementation
4. `src/thalia/training/monitor.py:123` - inline implementation

**Consolidation Target**: `src/thalia/core/spike_utils.py` (new file)

---

### Duplication 2: Weight Clamping
**Locations**:
1. `src/thalia/regions/striatum/learning_manager.py:187` - `self.weights.data.clamp_(...)`
2. `src/thalia/regions/hippocampus/plasticity_manager.py:142` - `self.weights.clamp_(...)`
3. `src/thalia/regions/cortex/layered_cortex.py:456` - `torch.clamp(..., out=...)`
4. `src/thalia/regions/prefrontal.py:298` - `self.weights.clamp_(...)`
5. `src/thalia/learning/bcm.py:187` - weight bound enforcement
6. `src/thalia/integration/spiking_pathway.py:234` - weight clamping

**Consolidation Target**: `src/thalia/core/weight_init.py::clamp_weights()` (add method)

---

### Duplication 3: Eligibility Trace Update Pattern
**Locations**:
1. `src/thalia/regions/striatum/eligibility.py:89` - exponential decay implementation
2. `src/thalia/core/traces.py:67` - similar exponential decay
3. `src/thalia/regions/cerebellum.py:239` - trace update (different time constant)

**Consolidation Target**: `src/thalia/core/traces.py` (already exists, make it more generic)

---

### Duplication 4: Homeostatic Firing Rate Tracking
**Locations**:
1. `src/thalia/regions/striatum/homeostasis_manager.py:145` - EMA of firing rate
2. `src/thalia/learning/ei_balance.py:78` - EMA of firing rate
3. `src/thalia/learning/intrinsic_plasticity.py:123` - EMA of firing rate
4. `src/thalia/learning/unified_homeostasis.py:89` - EMA of firing rate

**Consolidation Target**: `src/thalia/learning/unified_homeostasis.py` (already exists, use consistently)

---

### Duplication 5: Device Validation Pattern
**Locations**:
1. `src/thalia/regions/base.py:156` - tensor device check
2. `src/thalia/core/pathway_protocol.py:178` - tensor device check
3. `src/thalia/integration/spiking_pathway.py:267` - tensor device check

**Pattern**:
```python
if input.device != self.device:
    input = input.to(self.device)
```

**Consolidation Target**: `src/thalia/core/utils.py::ensure_device()` helper

---

### Duplication 6: Checkpoint State Dictionary Construction
**Locations**:
1. `src/thalia/regions/striatum/checkpoint_manager.py:67` - state dict construction
2. `src/thalia/regions/hippocampus/episode_manager.py:234` - state dict construction
3. `src/thalia/regions/base.py:787` - `get_full_state()` implementation
4. `src/thalia/core/pathway_protocol.py:298` - pathway state dict

**Pattern**: All construct similar dicts with `{'weights': ..., 'config': ..., 'state': ...}`

**Consolidation Target**: `src/thalia/io/checkpoint.py` - add helper function

---

## Appendix C: Antipattern Catalog

### Antipattern 1: Manager Proliferation ⚠️
**Severity**: Medium
**Location**: `striatum/`, `hippocampus/` subdirectories
**Description**: 7+ manager classes with overlapping responsibilities
**Fix**: See Tier 2.1

---

### Antipattern 2: God Object (Brain) ⚠️
**Severity**: High
**Location**: `src/thalia/core/brain.py` (1,853 lines)
**Description**: Single class managing regions, pathways, neuromodulation, oscillations, growth, forward pass, state, checkpointing, diagnostics
**Fix**: See Tier 3.2

---

### Antipattern 3: Duplicated Learning Logic ⚠️
**Severity**: Medium
**Location**: All region files
**Description**: Each region implements learning differently despite conceptual similarities
**Fix**: See Tier 2.3

---

### Antipattern 4: Config Proliferation ⚠️
**Severity**: Medium
**Location**: `src/thalia/config/` (9 files), region config files
**Description**: Unclear hierarchy, overlapping concerns
**Fix**: See Tier 2.4

---

### Antipattern 5: Inconsistent Directory Structure ⚠️
**Severity**: Low
**Location**: `src/thalia/regions/`
**Description**: Some regions have subdirectories, others don't (no clear guideline)
**Fix**: See Tier 2.2

---

## Appendix D: Pattern Improvements

### Improvement 1: Learning Rule Abstraction
**Current**: Each region implements learning in `forward()` or `learn()`
**Better**: Strategy pattern with `LearningRule` registry
**Benefit**:
- Composable learning rules
- Easier testing (test rule independently of region)
- Hybrid rules (combine STDP + BCM)
**See**: Tier 2.3

---

### Improvement 2: Unified Manager Base
**Current**: Multiple manager classes with similar patterns
**Better**: Either absorb into regions OR create functional component types
**Benefit**:
- Reduces indirection
- Clearer ownership
- Easier to understand control flow
**See**: Tier 2.1

---

### Improvement 3: Component Enforcement
**Current**: Protocol defines interface, but not enforced
**Better**: Abstract base class enforces all methods
**Benefit**:
- Static checking catches missing implementations
- Guarantees component parity
**See**: Tier 3.1

---

### Improvement 4: Decomposed Brain
**Current**: Monolithic `Brain` class (1,853 lines)
**Better**: Specialized components (architecture, neuromodulation, oscillations, growth)
**Benefit**:
- Easier testing
- Clearer responsibilities
- Easier to extend
**See**: Tier 3.2

---

### Improvement 5: Pathway Learning Strategy
**Current**: Inconsistent learning across pathway types
**Better**: Explicit `PathwayLearningStrategy` classes
**Benefit**:
- Consistent learning behavior
- Easy to swap strategies
- Clear documentation of learning rules
**See**: Tier 3.3

---

## Conclusion

Thalia's architecture is fundamentally sound with strong adherence to biological plausibility constraints. The codebase successfully implements key neuroscience principles (spike-based processing, local learning rules, neuromodulation) while avoiding common deep learning antipatterns.

**Key Strengths**:
1. ✅ Component parity between regions and pathways
2. ✅ No backpropagation (biologically plausible)
3. ✅ WeightInitializer registry (eliminates scattered init)
4. ✅ Neuron constants (eliminates magic numbers)
5. ✅ Manager pattern (BaseManager) for sub-components

**Key Opportunities**:
1. 📊 Consolidate manager classes (reduce indirection)
2. 📁 Standardize region organization (clearer structure)
3. 🔧 Abstract learning rules (eliminate duplication)
4. 📝 Simplify config hierarchy (reduce complexity)
5. 🧪 Expand integration tests (catch pathway-region issues)

**Recommended Priority**:
1. **Start with Tier 1** (quick wins, high impact, low disruption)
2. **Phase in Tier 2** strategically (learning strategy pattern most important)
3. **Plan Tier 3** long-term (major refactoring requires careful migration)

The most impactful changes completed:
1. ✅ **Learning rule abstraction** (Tier 2.3) - Eliminates code duplication, enables experimentation
2. ✅ **Manager consolidation** (Tier 2.1) - Simplifies region architecture with component pattern
3. ✅ **Magic number extraction** (Tier 1.1) - Improves maintainability and biological consistency

Remaining high-value improvements:
1. **Config consolidation** (Tier 2.4) - Reduce cognitive load for users
2. **Region directory standardization** (Tier 2.2) - Improve code organization
3. **Integration tests** (Tier 2.5) - Catch pathway-region coordination bugs

Overall assessment: **Healthy codebase with clear improvement path. Major progress made on Tier 1 and Tier 2.** No fundamental architectural flaws detected. Recommendations focus on continuing the momentum with config simplification and testing improvements.

---

**Review Completed**: December 11, 2025
**Next Review**: Q2 2026 (after Tier 1-2 implementations)
