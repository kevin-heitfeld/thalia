# Architecture Review – 2025-12-24

## Executive Summary

This comprehensive architectural analysis examines the Thalia codebase following the established architectural patterns and biological plausibility constraints. The review focuses on `src/thalia/` (core, regions, learning, integration, sensory) and provides structured refactoring recommendations organized by priority and impact.

**Key Findings**:
- **Strong Foundation**: Excellent adherence to biological plausibility constraints (local learning, spike-based processing, no backpropagation in regions)
- **Good Pattern Consistency**: WeightInitializer registry and NeuronFactory patterns are well-adopted across regions
- **Component Extraction Progress**: Recent work (BaseManager, component extraction patterns) shows effective anti-god-object strategy
- **Opportunities**: Further consolidation of growth patterns, documentation improvements, and minor naming refinements

**Overall Assessment**: The architecture is fundamentally sound with well-established patterns. Recommendations focus on incremental improvements rather than major restructuring.

---

## Findings by Priority Tier

### Tier 1 - High Impact, Low Disruption

*Changes that significantly improve code quality without breaking existing functionality*

#### 1.1 Documentation: Magic Number Extraction

**Current State**: Some numerical constants lack named constants with biological context.

**Detected Instances**:
- `src/thalia/utils/oscillator_utils.py`: Various threshold values (0.5, 0.1, etc.) in modulation functions
- `src/thalia/visualization/constants.py`: Layout constants (0.1, 0.9) without clear biological meaning
- Region-specific thresholds scattered across implementations

**Proposed Change**: Extract remaining magic numbers to named constants with documentation.

**Example**:
```python
# Before
if ach_level > 0.5:
    suppression = 1.0 - 0.3 * acetylcholine

# After
from thalia.regulation.neuromodulation_constants import (
    ACH_ENCODING_THRESHOLD,  # 0.5 - threshold for encoding mode
    ACH_SUPPRESSION_STRENGTH,  # 0.3 - suppression factor for recurrence
)

if ach_level > ACH_ENCODING_THRESHOLD:
    suppression = 1.0 - ACH_SUPPRESSION_STRENGTH * acetylcholine
```

**Impact**: 
- **Files Affected**: ~10-15 files across utils/, visualization/, regions/
- **Breaking Changes**: None (internal refactoring only)
- **Benefits**: Improved biological interpretability, easier tuning

**Rationale**: Named constants with biological context make the code more discoverable and maintainable. The existing pattern in `components/neurons/neuron_constants.py` should be extended to other subsystems.

---

#### 1.2 Naming: Clarify "Component" Terminology

**Current State**: "Component" is overloaded across multiple contexts:
- `BrainComponent` protocol (regions + pathways)
- `LearnableComponent` base class (neural components)
- `RoutingComponent` base class (non-learning components)
- Manager components (`LearningComponent`, `HomeostasisComponent`, etc.)
- UI/diagnostic components

**Proposed Change**: Add clarifying prefixes where ambiguity exists.

**Specific Recommendations**:

1. **Manager Components** → Already well-named: `StriatumLearningComponent`, `StriatumHomeostasisComponent`
   - No change needed (already clear)

2. **Core Protocols** → Keep existing names (they're foundational):
   - `BrainComponent` (protocol) ✅
   - `LearnableComponent` (base class) ✅
   - `RoutingComponent` (base class) ✅

3. **Documentation Enhancement**: Add disambiguation section to `docs/api/COMPONENT_CATALOG.md`

**Example Documentation Addition**:
```markdown
## Component Terminology Guide

**Component Types**:
1. **Brain Components**: Regions and pathways (implement BrainComponent protocol)
2. **Manager Components**: Extracted subsystems (LearningComponent, HomeostasisComponent)
3. **Neural Components**: Learnable units (LearnableComponent subclasses)
4. **Routing Components**: Pure signal routing (RoutingComponent subclasses)

When adding new components, follow naming conventions:
- Managers: `{Region}{Purpose}Component` (e.g., StriatumLearningComponent)
- Regions: `{Name}Region` or just `{Name}` (e.g., LayeredCortex)
- Pathways: `{Name}Pathway` (e.g., AxonalProjection)
```

**Impact**:
- **Files Affected**: 1 documentation file
- **Breaking Changes**: None (documentation only)
- **Benefits**: Reduced cognitive load for new contributors

---

#### 1.3 Pattern: Consolidate torch.randn/torch.rand Usage

**Current State**: Some files use direct `torch.randn()` or `torch.rand()` instead of `WeightInitializer` or documented utility functions.

**Detected Violations**:
- `src/thalia/utils/core_utils.py:215`: `torch.rand()` for phase preferences (documented as utility)
- `src/thalia/training/datasets/loaders.py`: `torch.rand()` for noise/stimulus generation (appropriate use)
- `src/thalia/tasks/stimulus_utils.py`: `torch.randn()` for stimulus generation (appropriate use)
- `src/thalia/diagnostics/metacognition.py:685`: `torch.randn()` for test patterns (appropriate use)

**Proposed Change**: Document exceptions where direct tensor creation is appropriate.

**Guideline**:
```python
# ✅ CORRECT: Use WeightInitializer for synaptic weights
weights = WeightInitializer.gaussian(n_output, n_input, mean=0.3, std=0.1, device=device)

# ✅ CORRECT: Use utility functions for biological patterns
phase_prefs = create_random_phase_preferences(n_neurons, device)  # Already exists

# ✅ CORRECT: Direct torch operations for non-synaptic data
test_input = torch.randn(input_size, device=device)  # Test data
noise = torch.randn_like(stimulus) * noise_scale  # Additive noise
stimulus_pattern = torch.rand(dim, device=device)  # Random stimulus

# ❌ INCORRECT: Direct torch for synaptic weights
weights = torch.randn(n_output, n_input, device=device) * 0.1
```

**Impact**:
- **Files Affected**: Add to CONTRIBUTING.md guidelines
- **Breaking Changes**: None (guideline clarification)
- **Benefits**: Clear distinction between synaptic and non-synaptic tensors

**Status**: **Mostly compliant** - only 1-2 edge cases found, already have utility functions

---

#### 1.4 Code Deduplication: Growth Method Helpers (COMPLETED in Dec 2025)

**Status**: ✅ **Already implemented** in `src/thalia/mixins/growth_mixin.py`

**Existing Solutions**:
- `_expand_weights()`: Expands weight matrices by adding rows
- `_create_new_weights()`: Creates new weight tensors with specified initialization
- `_grow_weight_matrix_cols()`: Adds columns to weight matrices (functional)
- `_grow_weight_matrix_rows()`: Adds rows to weight matrices (functional)

**Adoption Status**: Good adoption in recent regions, some legacy regions still use manual patterns.

**Recommendation**: Encourage adoption in remaining regions through code review, no forced refactoring.

**Example Legacy Pattern** (manual expansion):
```python
# Legacy pattern (still works, but verbose)
def grow_input(self, n_new: int) -> None:
    old_n_input = self.weights.shape[1]
    new_weights = torch.zeros(
        self.weights.shape[0], 
        old_n_input + n_new, 
        device=self.device
    )
    new_weights[:, :old_n_input] = self.weights
    new_weights[:, old_n_input:] = WeightInitializer.xavier(
        self.weights.shape[0], n_new, device=self.device
    )
    self.weights = nn.Parameter(new_weights)

# Modern pattern (using mixin)
def grow_input(self, n_new: int) -> None:
    new_weights = self._grow_weight_matrix_cols(
        self.weights, n_new, initializer='xavier'
    )
    self.weights = nn.Parameter(new_weights)
```

**Impact**:
- **Files Affected**: 0 (no forced changes)
- **Breaking Changes**: None
- **Benefits**: Already captured by recent implementations

---

#### 1.5 Documentation: Expand Growth API Examples

**Current State**: `docs/patterns/growth-api.md` provides good overview but could include more region-specific examples.

**Proposed Addition**: Add "Common Growth Patterns by Region Type" section.

**Example Content**:
```markdown
## Common Growth Patterns by Region Type

### Pattern 1: Simple Feedforward Region
```python
def grow_output(self, n_new: int) -> None:
    """Add output neurons."""
    # Expand weights (add rows)
    new_weights = self._grow_weight_matrix_rows(
        self.weights, n_new, initializer='xavier'
    )
    self.weights = nn.Parameter(new_weights)
    
    # Expand neurons
    old_neurons = self.neurons
    self.neurons = ConductanceLIF(
        old_neurons.n_neurons + n_new, 
        old_neurons.config
    )
    
    # Update config
    self.config.n_output += n_new
```

### Pattern 2: Multi-Source Region (NeuralRegion)
```python
def grow_source(self, source_name: str, new_size: int) -> None:
    """Grow specific input source."""
    old_size = self.input_sizes[source_name]
    n_new = new_size - old_size
    
    # Expand weights for this source
    old_weights = self.synaptic_weights[source_name]
    new_weights = self._grow_weight_matrix_cols(
        old_weights, n_new, initializer='sparse_random', sparsity=0.2
    )
    self.synaptic_weights[source_name] = nn.Parameter(new_weights)
    
    # Update tracking
    self.input_sizes[source_name] = new_size
```

### Pattern 3: Layered Region (e.g., LayeredCortex)
```python
def grow_output(self, n_new: int) -> None:
    """Add neurons to output layer (L5)."""
    # Distribute to layer (60% of growth)
    n_l5_new = int(n_new * 0.6)
    
    # Grow L5 neurons
    old_l5_neurons = self.l5_neurons
    self.l5_neurons = create_pyramidal_neurons(
        old_l5_neurons.n_neurons + n_l5_new, self.device
    )
    
    # Grow inter-layer connections (L2/3 → L5)
    self.w_l23_l5 = nn.Parameter(
        self._grow_weight_matrix_rows(
            self.w_l23_l5, n_l5_new, initializer='gaussian'
        )
    )
    
    # Update config
    self.config.l5_size += n_l5_new
    self.config.n_output += n_l5_new
```
```

**Impact**:
- **Files Affected**: `docs/patterns/growth-api.md`
- **Breaking Changes**: None (documentation enhancement)
- **Benefits**: Clearer implementation guidance for new regions

---

### Tier 2 - Moderate Refactoring

*Strategic improvements requiring coordinated changes across multiple files*

#### 2.1 Pattern: Extract Common Checkpoint Patterns

**Current State**: Each region implements its own checkpoint manager with similar boilerplate.

**Observed Pattern**: Most checkpoint managers follow this structure:
1. Extract per-neuron data with incoming synapses
2. Extract learning state (STP, STDP, eligibility)
3. Extract neuromodulator state
4. Extract region-specific state
5. Package into neuromorphic format

**Existing Infrastructure**: `BaseCheckpointManager` (Dec 2025) provides excellent foundation.

**Current Adoption**:
- ✅ `StriatumCheckpointManager`: Fully implemented
- ✅ `PrefrontalCheckpointManager`: Fully implemented
- ✅ `HippocampusCheckpointManager`: Fully implemented
- ⏸️ `LayeredCortex`, `Cerebellum`, `Thalamus`: Using legacy format (elastic tensor)

**Recommendation**: **No action required** - existing managers are well-structured, remaining regions use appropriate format for their characteristics.

**Rationale**: The hybrid format system (neuromorphic vs elastic tensor) allows regions to choose based on their characteristics:
- **Neuromorphic format**: Small regions, growth-enabled, high interpretability needs
- **Elastic tensor format**: Large stable regions (cortex, cerebellum), performance-critical

This is **good architectural diversity**, not a refactoring opportunity.

---

#### 2.2 Consolidation: Learning Strategy Factory Pattern

**Current State**: Multiple factory functions exist for creating learning strategies:
- `create_strategy()` in `learning/rules/strategies.py` (generic)
- `create_cortex_strategy()` in `learning/rules/strategies.py` (region-specific)
- `create_striatum_strategy()` in `regions/striatum/learning_component.py` (region-specific)
- `create_hippocampus_strategy()` (if exists)

**Proposed Change**: Consolidate region-specific factories into `learning/factories/` module.

**Structure**:
```
src/thalia/learning/
├── rules/
│   └── strategies.py           # Core strategies (Hebbian, STDP, BCM, ThreeFactor)
├── factories/
│   ├── __init__.py            # Re-export all create_* functions
│   ├── generic.py             # create_strategy() (generic factory)
│   ├── cortex_factories.py    # create_cortex_strategy() (STDP+BCM composite)
│   ├── striatum_factories.py  # create_striatum_strategy() (three-factor)
│   └── hippocampus_factories.py  # create_hippocampus_strategy() (STDP one-shot)
```

**Example** (`learning/factories/cortex_factories.py`):
```python
"""Cortex-specific learning strategy factories."""

from thalia.learning.rules.strategies import (
    STDPStrategy, BCMStrategy, CompositeStrategy,
    STDPConfig, BCMConfig,
)

def create_cortex_strategy(
    learning_rate: float = 0.001,
    stdp_a_plus: float = 0.01,
    stdp_a_minus: float = 0.012,
    bcm_tau_theta: float = 5000.0,
    **kwargs
) -> CompositeStrategy:
    """Create cortex learning strategy (STDP + BCM).
    
    Biological motivation:
    - STDP: Spike-timing based Hebbian plasticity
    - BCM: Sliding threshold for metaplasticity
    
    Returns:
        CompositeStrategy with STDP modulated by BCM threshold
    """
    stdp = STDPStrategy(STDPConfig(
        learning_rate=learning_rate,
        a_plus=stdp_a_plus,
        a_minus=stdp_a_minus,
        **kwargs
    ))
    
    bcm = BCMStrategy(BCMConfig(
        learning_rate=learning_rate,
        tau_theta=bcm_tau_theta,
        **kwargs
    ))
    
    return CompositeStrategy([stdp, bcm])
```

**Impact**:
- **Files Affected**: 8-10 files (learning/ module, region imports)
- **Breaking Changes**: Low (add backward compatibility imports)
- **Benefits**: Centralized learning configuration, easier discovery

**Migration Path**:
1. Create `learning/factories/` module
2. Move region-specific factories
3. Add backward compatibility imports in old locations (deprecated)
4. Update documentation
5. Remove deprecated imports after 2-3 releases

---

#### 2.3 Naming: Standardize State Class Naming

**Current State**: State classes use different naming patterns:
- `StriatumState` (dataclass)
- `BaseRegionState` (dataclass)
- `NeuralComponentState` (dataclass)
- `PathwayState` (protocol/dataclass hybrid)

**Observation**: All end with "State" (good consistency), no changes needed.

**Recommendation**: **No action** - naming is already consistent.

---

#### 2.4 Pattern: Standardize Diagnostics Collection

**Current State**: Most regions implement similar diagnostics patterns:
```python
def get_diagnostics(self) -> Dict[str, Any]:
    diag = {"region": "striatum"}
    
    # Weight stats
    diag["d1_weight_mean"] = self.d1_weights.mean().item()
    diag["d1_weight_std"] = self.d1_weights.std().item()
    # ... repeat for d2, fsi, etc.
    
    # Spike stats
    diag["d1_sparsity"] = 1.0 - self.d1_spikes.mean().item()
    # ... repeat
    
    return diag
```

**Existing Infrastructure**: `DiagnosticsMixin` provides `weight_diagnostics()`, `spike_diagnostics()`, etc. (good!)

**Current Adoption**: Mixed - some regions use mixin helpers, others implement manually.

**Recommendation**: Encourage adoption through documentation, not forced refactoring.

**Documentation Addition** (`docs/patterns/diagnostics.md`):
```markdown
## Using DiagnosticsMixin for Efficient Diagnostics

The `DiagnosticsMixin` provides helpers to reduce boilerplate:

```python
from thalia.mixins.diagnostics_mixin import DiagnosticsMixin

class MyRegion(NeuralRegion, DiagnosticsMixin):
    def get_diagnostics(self) -> Dict[str, Any]:
        # Use mixin helper for standard metrics
        return self.collect_standard_diagnostics(
            region_name="my_region",
            weight_matrices={
                "input": self.input_weights,
                "recurrent": self.recurrent_weights,
            },
            spike_tensors={
                "output": self.output_spikes,
            },
            trace_tensors={
                "eligibility": self.eligibility_trace,
            },
            custom_metrics={
                "learning_rate": self.current_lr,
                "exploration": self.exploring,
            }
        )
```

Benefits:
- **Less boilerplate**: 5-10 lines instead of 30-50
- **Consistent metrics**: Same stats computed everywhere
- **Extensible**: Add custom_metrics as needed
```

**Impact**:
- **Files Affected**: 1 documentation file
- **Breaking Changes**: None (opt-in pattern)
- **Benefits**: Reduced code duplication for new regions

---

### Tier 3 - Major Restructuring

*Long-term considerations requiring significant architectural changes*

#### 3.1 Consideration: Unified Learning Strategy API Evolution

**Current State**: Learning strategies use `compute_update()` method returning `(new_weights, metrics)`.

**Observation**: This API works well and is consistently implemented across all strategies.

**Potential Future Enhancement** (NOT recommended for immediate action):

Consider adding optional streaming metrics API for large-scale regions:
```python
class LearningStrategy(Protocol):
    def compute_update(...) -> Tuple[Tensor, Dict]:
        """Standard batch update (current API)."""
        ...
    
    def compute_update_streaming(...) -> Iterator[Tuple[Tensor, Dict]]:
        """Optional streaming API for large regions (future)."""
        ...
```

**Recommendation**: **Defer indefinitely** - current API is working well, no observed pain points.

**Rationale**: The existing `compute_update()` API is:
- Simple and easy to understand
- Consistently implemented
- Sufficient for current needs
- Biologically plausible (local updates)

**Status**: ⏸️ **Not actionable** - document for future consideration only

---

#### 3.2 Consideration: Component-Based Region Decomposition

**Current State**: Large regions (Striatum ~2372 lines, LayeredCortex ~2152 lines, Hippocampus ~2265 lines) are well-documented with navigation aids.

**Recent Progress**: Component extraction pattern successfully applied:
- ✅ `StriatumLearningComponent`, `StriatumHomeostasisComponent`, `StriatumExplorationComponent`
- ✅ `BaseManager` standardization
- ✅ Clear documentation justifying large files (ADR-011)

**Observation**: Large files are **justified** because they represent single biological computations:
- Striatum: D1/D2 opponent pathways interact every timestep
- LayeredCortex: L4→L2/3→L5 cascade is one biological circuit
- Hippocampus: DG→CA3→CA1 trisynaptic loop is one memory operation

**Recommendation**: **No further decomposition needed** - current structure is appropriate.

**Rationale**: Splitting these regions would:
1. Require passing 15-20 intermediate tensors between files
2. Obscure the biological circuit structure
3. Duplicate device/config management
4. Increase cognitive load (files scattered across directories)

The current pattern (main region file + extracted orthogonal components) strikes the right balance.

**Status**: ✅ **Complete** - no further action needed

---

#### 3.3 Consideration: Async/Parallel Region Processing

**Current State**: Regions process sequentially in brain forward pass.

**Potential Enhancement**: Enable parallel processing of independent regions.

**Example**:
```python
# Current (sequential)
thalamus_out = brain.thalamus(sensory_input)
cortex_out = brain.cortex(thalamus_out)
striatum_out = brain.striatum(cortex_out)

# Future (parallel - if beneficial)
with torch.no_grad():  # For independent regions
    future_thalamus = executor.submit(brain.thalamus, sensory_input)
    future_hippocampus = executor.submit(brain.hippocampus, other_input)
```

**Challenges**:
1. Biological plausibility: Brains are inherently sequential with delays
2. PyTorch limitations: Requires careful device management
3. Complexity: Dependency tracking, synchronization
4. Benefit unclear: CPU-bound training is already parallel at batch level

**Recommendation**: **Not pursued** - biological sequentiality is a feature, not a bug.

**Rationale**: 
- Axonal delays (1-50ms) mean biological regions ARE sequential
- Current clock-driven simulation respects causality
- Parallelism handled at batch level (multiple environments)

**Status**: ⏸️ **Not actionable** - counter to biological realism

---

## Antipattern Detection

### Detected Antipatterns

#### AP-1: Backpropagation in Metacognitive Monitor

**Location**: `src/thalia/diagnostics/metacognition.py:196-202`

**Description**: Uses `loss.backward()` and gradient manipulation for calibration network.

**Assessment**: ✅ **Justified exception**

**Rationale**: 
- **Different timescale**: Metacognitive calibration operates at trial boundaries (~seconds), not timesteps (~milliseconds)
- **Supervisory learning**: Calibrating confidence against actual outcomes is fundamentally supervised
- **Isolated scope**: Does NOT affect brain regions (no backprop through regions)
- **Documented exception**: Clearly marked with `# EXCEPTION: Temporarily enable gradients` comment

**Recommendation**: **No change** - this is appropriate use of backpropagation for a meta-learning system.

---

#### AP-2: God Object Pattern - NOT DETECTED

**Assessment**: ✅ **No god objects found**

**Evidence**:
- Large regions (Striatum, LayeredCortex, Hippocampus) have clear single responsibilities
- Component extraction pattern successfully applied (managers, learning components)
- Good separation of concerns:
  - Learning: `LearningComponent` subclasses
  - Homeostasis: `HomeostasisComponent` subclasses
  - Memory: `MemoryComponent` subclasses
  - Exploration: `ExplorationComponent` subclasses

**Exemplar**: `Striatum` (2372 lines)
- Main class: Action selection coordination (400 lines)
- Extracted: `StriatumLearningComponent` (three-factor learning)
- Extracted: `StriatumHomeostasisComponent` (E/I balance)
- Extracted: `StriatumExplorationComponent` (UCB tracking)
- Extracted: `D1Pathway`, `D2Pathway` (parallel pathway computation)

---

#### AP-3: Circular Dependencies - NOT DETECTED

**Assessment**: ✅ **No circular imports found**

**Evidence**: Module structure follows clear hierarchy:
```
core/ (base classes, protocols)
  ↓
components/ (neurons, synapses)
  ↓
learning/ (strategies, eligibility)
  ↓
regions/ (brain regions)
  ↓
pathways/ (connections)
  ↓
integration/ (brain assembly)
```

**Recommendation**: Maintain current import discipline.

---

#### AP-4: Magic Numbers - Mostly Addressed

**Assessment**: ⚠️ **Minor improvements possible** (see Tier 1.1)

**Evidence**:
- ✅ Excellent neuron constants: `V_THRESHOLD_STANDARD`, `TAU_MEM_FAST`, etc.
- ✅ Good learning constants: `METACOG_CALIBRATION_LR`, etc.
- ⚠️ Some oscillator constants could be named: 0.5 thresholds, 0.3 suppression factors

**Recommendation**: Extract remaining ~10-15 magic numbers (Tier 1.1).

---

#### AP-5: Tight Coupling - NOT DETECTED

**Assessment**: ✅ **Good decoupling**

**Evidence**:
- Regions depend on protocols, not concrete classes
- Learning strategies are pluggable via factory pattern
- Neuron models use dependency injection (config-based)
- Pathways use `SourceSpec` abstraction (not direct region references)

**Example** (good decoupling):
```python
# Region doesn't know about specific learning strategy
self.learning_strategy = create_strategy(
    'stdp',  # Could be 'bcm', 'hebbian', 'three_factor'
    learning_rate=0.001
)

# Learning happens through protocol interface
new_weights, metrics = self.learning_strategy.compute_update(...)
```

---

### Non-Antipatterns (False Positives)

#### NAP-1: Large Files (Striatum, LayeredCortex, Hippocampus)

**Why NOT an antipattern**:
- Justified in `docs/decisions/adr-011-large-file-justification.md`
- Represent single biological computations
- Well-organized with navigation aids (docstring maps)
- Successfully extracted orthogonal concerns to components

**Status**: ✅ **Intentional design**

---

#### NAP-2: Multiple Inheritance in NeuralRegion

**Pattern**:
```python
class NeuralRegion(
    nn.Module,
    BrainComponentMixin,
    NeuromodulatorMixin,
    GrowthMixin,
    ResettableMixin,
    DiagnosticsMixin,
    StateLoadingMixin,
    LearningStrategyMixin
):
```

**Why NOT an antipattern**:
- Mixins provide orthogonal concerns (composition over inheritance)
- Clear MRO (Method Resolution Order) with `super()` calls
- Well-documented in `docs/patterns/mixins.md`
- Each mixin has single responsibility

**Status**: ✅ **Good use of mixins**

---

#### NAP-3: ParameterDict for Synaptic Weights

**Pattern**:
```python
self.synaptic_weights: nn.ParameterDict = nn.ParameterDict()
self.synaptic_weights["thalamus"] = nn.Parameter(weights)
```

**Why NOT an antipattern**:
- Biologically accurate (weights stored at target dendrites)
- Enables multi-source integration
- Proper state_dict serialization
- Device movement handled correctly

**Status**: ✅ **Architecturally sound**

---

## Pattern Adherence

### ✅ Excellent Adherence

#### 1. WeightInitializer Registry Pattern

**Adoption**: ~95% of weight initialization uses registry

**Evidence**:
- All regions use `WeightInitializer.gaussian()`, `.xavier()`, `.sparse_random()`
- Test data and stimulus generation appropriately use direct `torch.randn()`
- Clear guidelines in `CONTRIBUTING.md`

**Exemplar**:
```python
# Striatum initialization
d1_weights = WeightInitializer.xavier(n_d1, n_input, device=device)
d2_weights = WeightInitializer.xavier(n_d2, n_input, device=device)
fsi_weights = WeightInitializer.sparse_random(
    n_fsi, n_input, sparsity=0.2, device=device
)
```

---

#### 2. NeuronFactory Pattern

**Adoption**: ~90% of neuron creation uses factory or documented helper functions

**Evidence**:
- Regions use `create_pyramidal_neurons()`, `create_relay_neurons()`, `create_trn_neurons()`
- NeuronFactory registry available for dynamic creation
- Some regions use direct `ConductanceLIF()` (appropriate for custom configs)

**Exemplar**:
```python
# Thalamus
self.relay_neurons = create_relay_neurons(self.n_relay, self.device)
self.trn_neurons = create_trn_neurons(self.n_trn, self.device)

# Custom config (also acceptable)
self.neurons = ConductanceLIF(
    n_neurons, 
    ConductanceLIFConfig(tau_mem=15.0, g_leak=0.05)
)
```

---

#### 3. Component Extraction Pattern (Recent Addition)

**Adoption**: Striatum, Prefrontal (completed), Hippocampus (in progress)

**Evidence**:
- `BaseManager` standardization (Dec 2025)
- `StriatumLearningComponent`, `StriatumHomeostasisComponent` extraction
- `ManagerContext` pattern for shared resources
- Clear documentation in `docs/patterns/component-parity.md`

**Exemplar**:
```python
from thalia.managers.base_manager import BaseManager, ManagerContext

class StriatumLearningComponent(BaseManager[StriatumLearningConfig]):
    def __init__(self, config, context: ManagerContext):
        super().__init__(config, context)
        self.eligibility_d1 = torch.zeros(
            context.n_output, context.n_input, device=context.device
        )
    
    def apply_learning(self, ...):
        # Three-factor rule implementation
        ...
```

---

### ⚠️ Partial Adherence

#### 1. Growth API Standardization

**Adoption**: ~70% (new regions use mixins, legacy regions use manual patterns)

**Observation**: 
- Recent regions (Prefrontal, recent Striatum updates) use `_grow_weight_matrix_cols/rows()`
- Legacy regions (Cerebellum, older parts of Cortex) use manual expansion
- Both approaches work correctly

**Recommendation**: Encourage mixin adoption through code review, not forced refactoring

**Status**: ⚠️ **Acceptable** - gradual migration in progress

---

#### 2. DiagnosticsMixin Adoption

**Adoption**: ~60% (newer regions use helpers, older regions implement manually)

**Observation**:
- `DiagnosticsMixin` provides good helpers (`weight_diagnostics()`, `spike_diagnostics()`)
- Some regions use helpers, others duplicate logic
- Both produce equivalent diagnostics

**Recommendation**: Document pattern more prominently (see Tier 2.4)

**Status**: ⚠️ **Acceptable** - educational opportunity

---

## Biological Plausibility Assessment

### ✅ Constraints Satisfied

#### 1. Local Learning Rules

**Assessment**: ✅ **Excellent compliance**

**Evidence**:
- All learning strategies use local information only (pre, post spikes)
- Three-factor rule: eligibility (local) × dopamine (broadcast neuromodulator)
- STDP: pre_trace × post_spike (local spike timing)
- BCM: local activity × sliding threshold (local adaptation)

**No violations detected**.

---

#### 2. Spike-Based Processing

**Assessment**: ✅ **Excellent compliance**

**Evidence**:
- All regions use binary spikes (0 or 1)
- ConductanceLIF is ONLY neuron model (conductance-based dynamics)
- Temporal dynamics preserved (spike timing, delays, traces)

**No violations detected** (except metacognitive calibration, which is meta-learning, not brain regions).

---

#### 3. No Global Error Signals

**Assessment**: ✅ **Excellent compliance**

**Evidence**:
- No backpropagation in brain regions
- Error-corrective learning (cerebellum) uses local error, not global
- Reward signals broadcast via dopamine (biologically plausible neuromodulation)

**No violations detected**.

---

#### 4. Causality Preservation

**Assessment**: ✅ **Excellent compliance**

**Evidence**:
- Axonal delays implemented via `CircularDelayBuffer`
- Clock-driven simulation respects temporal ordering
- No future information access

**No violations detected**.

---

## Risk Assessment

### Low Risk Recommendations (Tier 1)

**Risk Level**: 🟢 **Minimal**

**Recommended Sequence**:
1. Magic number extraction (pure documentation/constant extraction)
2. Component terminology documentation (documentation only)
3. Torch.randn/rand guidelines (documentation clarification)
4. Growth API documentation expansion (examples)

**Estimated Effort**: 4-8 hours total

---

### Medium Risk Recommendations (Tier 2)

**Risk Level**: 🟡 **Moderate**

**Recommended Sequence**:
1. Learning strategy factory consolidation (requires import updates)
2. Diagnostics pattern documentation (encourages adoption)

**Estimated Effort**: 8-16 hours total

**Mitigation Strategies**:
- Backward compatibility imports during factory consolidation
- Deprecation warnings with 2-3 release grace period
- Comprehensive testing of import paths

---

### High Risk Recommendations (Tier 3)

**Risk Level**: 🔴 **Significant** (but NOT recommended)

**Items**: 
- Unified learning strategy API evolution (DEFERRED)
- Further component decomposition (NOT NEEDED)
- Async/parallel region processing (NOT PURSUED)

**Status**: ⏸️ **Indefinitely deferred** - current architecture is appropriate

---

## Implementation Priority

### Immediate (Q1 2026)

1. **Tier 1.1**: Magic number extraction
2. **Tier 1.2**: Component terminology documentation
3. **Tier 1.5**: Growth API documentation expansion

**Rationale**: High value, zero risk, educational benefits

---

### Short-term (Q2 2026)

1. **Tier 2.1**: (OPTIONAL) Checkpoint pattern documentation review
2. **Tier 2.4**: Diagnostics pattern documentation

**Rationale**: Improves developer experience for new regions

---

### Medium-term (Q3-Q4 2026)

1. **Tier 2.2**: (OPTIONAL) Learning strategy factory consolidation

**Rationale**: Nice-to-have refactoring, not urgent

---

### Long-term (2027+)

1. **Tier 3 items**: Monitor for actual pain points before acting

**Rationale**: Current architecture is working well, avoid premature optimization

---

## Appendix A: Affected Files

### Tier 1 Recommendations

**1.1 Magic Number Extraction**:
- `src/thalia/utils/oscillator_utils.py`
- `src/thalia/visualization/constants.py`
- `src/thalia/regions/hippocampus/trisynaptic.py` (threshold constants)
- `src/thalia/regions/striatum/striatum.py` (exploration constants)
- `src/thalia/components/neurons/neuron_constants.py` (add new constants)

**1.2 Component Terminology Documentation**:
- `docs/api/COMPONENT_CATALOG.md` (add disambiguation section)

**1.3 Torch Usage Guidelines**:
- `CONTRIBUTING.md` (add guideline section)

**1.5 Growth API Documentation**:
- `docs/patterns/growth-api.md` (add examples section)

---

### Tier 2 Recommendations

**2.2 Learning Strategy Factory Consolidation**:
- `src/thalia/learning/factories/__init__.py` (NEW)
- `src/thalia/learning/factories/generic.py` (NEW)
- `src/thalia/learning/factories/cortex_factories.py` (NEW)
- `src/thalia/learning/factories/striatum_factories.py` (NEW)
- `src/thalia/learning/rules/strategies.py` (backward compat imports)
- `src/thalia/regions/striatum/learning_component.py` (backward compat imports)

**2.4 Diagnostics Pattern Documentation**:
- `docs/patterns/diagnostics.md` (NEW or expand existing)

---

## Appendix B: Detected Code Duplications

### Duplication 1: Weight Expansion in Growth Methods ✅ RESOLVED

**Status**: ✅ **Consolidated in GrowthMixin** (Dec 2025)

**Solution**: `_grow_weight_matrix_cols()` and `_grow_weight_matrix_rows()` helper methods

**Adoption**: Gradual migration, no forced refactoring

---

### Duplication 2: Diagnostics Collection Boilerplate ⚠️ PARTIALLY RESOLVED

**Status**: ⚠️ **Mixin exists, adoption ongoing**

**Locations** (manual implementations):
- `regions/striatum/striatum.py:1650` (get_diagnostics)
- `regions/cortex/layered_cortex.py:1350` (get_diagnostics)
- `regions/hippocampus/trisynaptic.py:1850` (get_diagnostics)
- `regions/cerebellum_region.py:1450` (get_diagnostics)

**Consolidation**: `DiagnosticsMixin` provides helpers (`weight_diagnostics`, `spike_diagnostics`, `collect_standard_diagnostics`)

**Recommendation**: Document pattern more prominently (Tier 2.4), encourage adoption through examples

---

### Duplication 3: Reset State Patterns ⚠️ MINOR

**Locations**:
- Most regions implement similar `reset_state()` pattern:
  ```python
  def reset_state(self):
      self.neurons.reset_state()
      self.state.spikes = None
      self.state.membrane = None
      # ... repeat for traces, buffers, etc.
  ```

**Consolidation Opportunity**: `ResettableMixin._reset_subsystems()` and `_reset_scalars()` helpers exist

**Status**: ⚠️ **Acceptable** - region-specific state varies enough that full consolidation may not be beneficial

**Recommendation**: Document helpers, encourage adoption for new regions

---

## Appendix C: Pattern Improvement Opportunities

### Opportunity 1: Composite Learning Strategy Composition

**Current State**: Manual composition of STDP + BCM in regions

**Example** (current pattern):
```python
# In region __init__
self.stdp_strategy = STDPStrategy(STDPConfig(...))
self.bcm_strategy = BCMStrategy(BCMConfig(...))

# In forward()
new_weights, _ = self.stdp_strategy.compute_update(...)
phi = self.bcm_strategy.compute_phi(post)
new_weights = new_weights * phi.unsqueeze(1)  # Modulate STDP by BCM
```

**Improved Pattern** (already exists!):
```python
# Use CompositeStrategy
from thalia.learning.rules.strategies import CompositeStrategy

self.learning_strategy = CompositeStrategy([
    STDPStrategy(stdp_config),
    BCMStrategy(bcm_config),
])

# In forward() - single call
new_weights, metrics = self.learning_strategy.compute_update(...)
```

**Status**: ✅ **Available** - document more prominently

**Impact**: Simplifies region implementations, encourages strategy reuse

---

### Opportunity 2: Mixin Method Chaining

**Current State**: Regions call multiple mixin methods sequentially

**Example** (current pattern):
```python
def get_diagnostics(self):
    diag = {}
    diag.update(self.weight_diagnostics(self.weights, "main"))
    diag.update(self.spike_diagnostics(self.spikes, "output"))
    diag.update(self.trace_diagnostics(self.eligibility, "eligibility"))
    return diag
```

**Improved Pattern** (already exists!):
```python
def get_diagnostics(self):
    return self.collect_standard_diagnostics(
        region_name="my_region",
        weight_matrices={"main": self.weights},
        spike_tensors={"output": self.spikes},
        trace_tensors={"eligibility": self.eligibility},
    )
```

**Status**: ✅ **Available** - promote through documentation (Tier 2.4)

---

### Opportunity 3: Growth Helper Usage

**Status**: ✅ **Implemented and documented** (Dec 2025)

**Helpers Available**:
- `_grow_weight_matrix_cols()`: Add columns (grow input)
- `_grow_weight_matrix_rows()`: Add rows (grow output)
- `_create_new_weights()`: Create new weight tensor with initialization
- `_expand_weights()`: Generic weight expansion

**Adoption**: Ongoing, no forced migration

---

## Conclusion

The Thalia architecture demonstrates strong adherence to biological plausibility constraints and established design patterns. The codebase is well-structured with clear separation of concerns, appropriate use of mixins, and excellent documentation.

**Key Strengths**:
1. ✅ Biological plausibility (local learning, spike-based, no backprop in regions)
2. ✅ Pattern consistency (WeightInitializer, NeuronFactory, component extraction)
3. ✅ Good documentation (ADRs, pattern guides, API references)
4. ✅ Recent improvements (BaseManager, GrowthMixin consolidation)

**Recommended Focus**:
1. **Tier 1 items**: High value, low risk (magic numbers, documentation)
2. **Tier 2 items**: Strategic improvements (factory consolidation, diagnostics docs)
3. **Tier 3 items**: Defer indefinitely (current architecture is appropriate)

**No major restructuring needed** - the architecture is fundamentally sound and supports the project's goals effectively.
