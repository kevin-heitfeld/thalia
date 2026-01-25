# Architecture Review – 2026-01-25

## Executive Summary

Comprehensive analysis of the Thalia codebase architecture conducted on January 25, 2026. The review focused on the `src/thalia/` directory, examining module organization, learning rules, neuron models, pathway implementations, and adherence to biological plausibility constraints.

**Key Findings:**
- **Strong Foundation**: Well-established architectural patterns with excellent biological plausibility
- **Good Separation**: WeightInitializer registry, NeuronFactory pattern, learning strategies all properly abstracted
- **Identified Issues**: Some code duplication in checkpoint managers (~400-500 lines), manual weight initialization still present in ~30 locations, inconsistent constant usage
- **Opportunities**: Minor naming refinements, consolidation of similar region implementations, pattern consistency improvements

**Overall Architecture Grade: B+ (Very Good)**
- Strengths: Biological accuracy, mixin architecture, registry patterns
- Areas for improvement: Code duplication reduction, constant extraction, naming consistency

---

## Table of Contents

1. [Tier 1 Recommendations - High Impact, Low Disruption](#tier-1-recommendations)
2. [Tier 2 Recommendations - Moderate Refactoring](#tier-2-recommendations)
3. [Tier 3 Recommendations - Major Restructuring](#tier-3-recommendations)
4. [Risk Assessment & Sequencing](#risk-assessment--sequencing)
5. [Appendix A: Affected Files](#appendix-a-affected-files)
6. [Appendix B: Detected Duplications](#appendix-b-detected-duplications)

---

## Tier 1 Recommendations - High Impact, Low Disruption

These improvements provide immediate value with minimal risk. Implement these first.

### 1.1 ✅ Extract Common Checkpoint Manager Logic (COMPLETED)

**Status**: Already implemented in `managers/base_checkpoint_manager.py`

**Current State**: The codebase already has `BaseCheckpointManager` with common extraction methods:
- `extract_neuron_state_common()`
- `extract_elastic_tensor_metadata()`
- `validate_state_dict_keys()`
- `validate_tensor_shapes()`
- `validate_checkpoint_compatibility()`
- `handle_elastic_tensor_growth()`

**Impact**: Low - already completed
**Files**: `src/thalia/managers/base_checkpoint_manager.py` (lines 1-100 show new helper methods)

**Note**: This was identified as needed but inspection shows it's already done. No action required.

---

### 1.2 ✅ Replace Manual Weight Initialization with WeightInitializer Registry (COMPLETED)

**Status**: Completed January 25, 2026 - All identified high and medium priority files updated

**Current State**: Despite having a comprehensive `WeightInitializer` registry, ~30 files still use manual initialization:
- `torch.randn()` for Gaussian initialization
- `torch.rand()` for uniform initialization
- Manual Xavier/Kaiming calculations

**Proposed Change**: Replace all manual weight initialization with registry calls:

```python
# BEFORE (antipattern):
weights = torch.randn(n_output, n_input, device=device) * 0.1

# AFTER (correct pattern):
from thalia.components.synapses import WeightInitializer
weights = WeightInitializer.gaussian(n_output, n_input, std=0.1, device=device)
```

**Rationale**:
- Single source of truth for initialization strategies
- Easier to experiment with different initialization methods
- Better testability and reproducibility
- Automatic handling of device placement
- Consistent parameter naming across codebase

**Affected Locations** (30 files):

**High Priority** (region weight initialization):
1. [regions/hippocampus/trisynaptic.py](../../../src/thalia/regions/hippocampus/trisynaptic.py#L672) - line 672: `torch.randn_like(base_weights)`
2. [regions/hippocampus/trisynaptic.py](../../../src/thalia/regions/hippocampus/trisynaptic.py#L573) - `_initialize_weights()` method
3. [regions/cortex/layered_cortex.py](../../../src/thalia/regions/cortex/layered_cortex.py#L396) - `_initialize_weights()` method
4. [regions/prefrontal/prefrontal.py](../../../src/thalia/regions/prefrontal/prefrontal.py#L975) - line 975: manual weight growth
5. [regions/prefrontal/goal_emergence.py](../../../src/thalia/regions/prefrontal/goal_emergence.py#L108) - line 108: value weights
6. [regions/striatum/striatum.py](../../../src/thalia/regions/striatum/striatum.py#L1912) - `_initialize_pathway_weights()` method
7. [regions/thalamus/thalamus.py](../../../src/thalia/regions/thalamus/thalamus.py#L379) - `_initialize_weights()` method
8. [regions/cerebellum/cerebellum.py](../../../src/thalia/regions/cerebellum/cerebellum.py#L406) - `_initialize_weights_tensor()` method

**Medium Priority** (component initialization):
9. [components/neurons/dendritic.py](../../../src/thalia/components/neurons/dendritic.py#L165) - line 165: branch weights
10. [components/neurons/dendritic.py](../../../src/thalia/components/neurons/dendritic.py#L311) - line 311: multi-branch weights
11. [components/synapses/stp.py](../../../src/thalia/components/synapses/stp.py#L512) - line 512: STP weight initialization

**Low Priority** (test data, examples, noise):
12. [tasks/stimulus_utils.py](../../../src/thalia/tasks/stimulus_utils.py#L45) - line 45: stimulus generation
13. [tasks/working_memory.py](../../../src/thalia/tasks/working_memory.py#L460) - line 460: random matching
14. [training/datasets/loaders.py](../../../src/thalia/training/datasets/loaders.py#L713) - line 713: random images
15-30. Various test utilities, noise generation, and example code

**Impact**:
- **Breaking**: Low (internal implementation detail)
- **Files affected**: ~30 files
- **Lines changed**: ~50-80 lines total
- **Benefits**: Improved consistency, better maintainability, easier to modify initialization strategies globally

**Implementation Approach**:
1. Start with high-priority region files (8 files)
2. Create helper script to identify and suggest replacements
3. Test each region after conversion to ensure numerical equivalence
4. Continue with medium-priority component files
5. Leave low-priority test/utility files as-is (acceptable for test code)

---

### 1.3 Extract Magic Numbers to Named Constants

**Current State**: Several numeric literals appear throughout the codebase without clear biological context:

**Examples Found**:
```python
# regions/hippocampus/trisynaptic.py
noise = torch.randn_like(active_v) * 0.002  # What is 0.002?
phase_modulation = 1.0 + jitter_scale * torch.randn_like(base_weights)  # What is jitter_scale default?

# regions/prefrontal/prefrontal.py
noise = torch.randn_like(new_wm) * wm_noise_std  # wm_noise_std not defined as constant

# regions/striatum/action_selection.py
heterosynaptic_ratio: float = 0.3  # Magic number in method signature

# components/neurons/neuron.py
ou_decay = 1.0 - self.config.noise_std  # Implicit OU noise decay calculation
```

**Proposed Changes**: Extract to `constants/neuron.py` and `constants/learning.py`:

```python
# In constants/neuron.py (NEW additions):
MEMBRANE_NOISE_STD_MV = 0.002
"""Biological membrane noise std deviation (2mV).

White noise component representing stochastic ion channel fluctuations
and synaptic noise. Typical value 1-3mV.
"""

CONDUCTANCE_NOISE_STD = 0.002
"""Conductance noise std deviation for biological realism."""

WM_NOISE_STD = 0.1
"""Working memory noise std deviation for drift prevention."""

# In constants/learning.py (NEW additions):
HETEROSYNAPTIC_RATIO_DEFAULT = 0.3
"""Default heterosynaptic plasticity ratio.

Heterosynaptic LTD/LTP affects non-coactive synapses at 30% strength
of homosynaptic changes (Chistiakova et al., 2014).
"""

WEIGHT_JITTER_SCALE = 0.1
"""Weight jitter scale for phase preference diversity."""
```

**Rationale**:
- Makes biological basis explicit via documentation
- Easier to tune parameters across regions
- Improves code readability
- Facilitates parameter sensitivity analysis
- Single source of truth for biological constants

**Affected Files** (estimate 15-20 files):
- [regions/hippocampus/trisynaptic.py](../../../src/thalia/regions/hippocampus/trisynaptic.py)
- [regions/prefrontal/prefrontal.py](../../../src/thalia/regions/prefrontal/prefrontal.py)
- [regions/striatum/d1_pathway.py](../../../src/thalia/regions/striatum/d1_pathway.py)
- [regions/striatum/d2_pathway.py](../../../src/thalia/regions/striatum/d2_pathway.py)
- [components/neurons/neuron.py](../../../src/thalia/components/neurons/neuron.py)

**Impact**:
- **Breaking**: None (internal constants)
- **Files affected**: ~15-20
- **Lines changed**: ~30-40 lines (add constants + update usage)
- **Benefits**: Better documentation, easier parameter tuning, clearer biological basis

---

### 1.4 ✅ Naming Consistency: "learning_strategy" vs "learning_rule" (COMPLETED)

**Status**: Completed January 25, 2026 - Parameter renamed with consistent terminology throughout

**Current State**: Mixed terminology for the same concept:
- `NeuralRegion.__init__`: parameter is `default_learning_rule`
- `LearningStrategyMixin`: uses `learning_strategy` attribute
- Documentation uses both "learning rule" and "learning strategy"

**Example of confusion**:
```python
# In NeuralRegion.__init__:
def __init__(self, ..., default_learning_rule="stdp"):
    self.learning_strategies = {}  # But parameter is "rule"!
```

**Proposed Change**: Standardize on **"learning_strategy"** throughout:

```python
# BEFORE:
def __init__(self, ..., default_learning_rule="stdp"):

# AFTER:
def __init__(self, ..., default_learning_strategy="stdp"):
```

**Rationale**:
- "Strategy" is the correct pattern name (Strategy design pattern)
- Aligns with `LearningStrategyMixin` class name
- "Rule" implies fixed algorithm, "Strategy" implies pluggable behavior
- More consistent with software engineering terminology

**Affected Files**:
- [core/neural_region.py](../../../src/thalia/core/neural_region.py#L100) - Parameter name
- [learning/strategy_mixin.py](../../../src/thalia/learning/strategy_mixin.py) - Documentation
- [docs/patterns/learning-strategies.md](../../../docs/patterns/learning-strategies.md) - Documentation

**Impact**:
- **Breaking**: Low (mostly internal parameter names, backward compatibility possible)
- **Files affected**: ~5 files + documentation
- **Lines changed**: ~20 lines
- **Migration**: Add deprecation warning for `default_learning_rule`, support both for 1-2 releases

---

### 1.5 ✅ Simplify D1/D2 Pathway apply_dopamine_modulation() Methods (COMPLETED)

**Status**: Completed January 25, 2026 - Common logic consolidated in base class with polarity pattern

**Current State**: D1 and D2 pathways have near-identical `apply_dopamine_modulation()` methods (~50 lines each) with only one line difference:

**D1Pathway** ([d1_pathway.py](../../../src/thalia/regions/striatum/d1_pathway.py#L47)):
```python
def apply_dopamine_modulation(self, dopamine: float, ...) -> Dict[str, Any]:
    new_weights, metrics = self.learning_strategy.compute_update(
        weights=self.weights.data,
        pre=torch.ones(1, device=self.device),
        post=torch.ones(1, device=self.device),
        modulator=dopamine,  # <-- Direct dopamine
    )
    metrics["pathway"] = "D1"
    # ... 15 more lines ...
```

**D2Pathway** ([d2_pathway.py](../../../src/thalia/regions/striatum/d2_pathway.py#L53)):
```python
def apply_dopamine_modulation(self, dopamine: float, ...) -> Dict[str, Any]:
    inverted_dopamine = -dopamine  # <-- ONLY DIFFERENCE
    new_weights, metrics = self.learning_strategy.compute_update(
        weights=self.weights.data,
        pre=torch.ones(1, device=self.device),
        post=torch.ones(1, device=self.device),
        modulator=inverted_dopamine,  # <-- Inverted dopamine
    )
    metrics["pathway"] = "D2"
    # ... 15 more lines ...
```

**Proposed Change**: Move common logic to `StriatumPathway` base class:

```python
# In pathway_base.py (NEW):
class StriatumPathway(nn.Module, GrowthMixin, ResettableMixin, ABC):
    def __init__(self, config):
        super().__init__()
        self.pathway_name = ""  # Set by subclass
        self.dopamine_polarity = 1.0  # +1 for D1, -1 for D2

    def apply_dopamine_modulation(
        self, dopamine: float, heterosynaptic_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """Apply dopamine modulation with pathway-specific polarity."""
        # Apply polarity (D1: +1, D2: -1)
        modulated_dopamine = dopamine * self.dopamine_polarity

        # Common update logic
        new_weights, metrics = self.learning_strategy.compute_update(
            weights=self.weights.data,
            pre=torch.ones(1, device=self.device),
            post=torch.ones(1, device=self.device),
            modulator=modulated_dopamine,
        )

        # Update weights
        self.weights.data = new_weights

        # Add pathway metadata
        metrics["pathway"] = self.pathway_name
        metrics["dopamine_sign"] = (
            "positive" if dopamine > 0 else "negative" if dopamine < 0 else "zero"
        )

        return metrics

# In d1_pathway.py (SIMPLIFIED):
class D1Pathway(StriatumPathway):
    def __init__(self, config):
        super().__init__(config)
        self.pathway_name = "D1"
        self.dopamine_polarity = 1.0  # Direct modulation

# In d2_pathway.py (SIMPLIFIED):
class D2Pathway(StriatumPathway):
    def __init__(self, config):
        super().__init__(config)
        self.pathway_name = "D2"
        self.dopamine_polarity = -1.0  # Inverted modulation
```

**Rationale**:
- Eliminates ~90 lines of duplicated code (2 × 45 lines)
- Makes D1/D2 difference crystal clear (just polarity)
- Easier to add D3/D4/D5 receptor pathways in future
- Single point of maintenance for dopamine modulation logic
- More testable (test base class once instead of twice)

**Impact**:
- **Breaking**: None (external API unchanged)
- **Files affected**: 3 files (pathway_base.py, d1_pathway.py, d2_pathway.py)
- **Lines changed**: +30 in base, -45 in D1, -45 in D2 = Net -60 lines
- **Benefits**: Reduced duplication, clearer design, easier maintenance

---

### 1.6 ✅ Consistent Import Path for WeightInitializer (COMPLETED)

**Status**: Completed January 25, 2026 - All WeightInitializer imports standardized to short path

**Current State**: WeightInitializer imported from different paths:
```python
# Pattern 1 (correct):
from thalia.components.synapses import WeightInitializer

# Pattern 2 (also seen):
from thalia.components.synapses.weight_init import WeightInitializer

# Pattern 3 (in __init__.py):
from thalia.components.synapses.weight_init import InitStrategy, WeightInitializer
```

**Proposed Change**: Standardize on short path:
```python
# RECOMMENDED (everywhere):
from thalia.components.synapses import WeightInitializer, InitStrategy
```

**Rationale**:
- Shorter imports are more readable
- Follows Python convention (import from package, not module)
- `components/synapses/__init__.py` already exports WeightInitializer
- Consistent with other component imports (e.g., `from thalia.components.neurons import NeuronFactory`)

**Affected Files**: 7 code files updated
**Impact**:
- **Breaking**: None (imports work either way)
- **Lines changed**: 7 lines (one per file)
- **Benefits**: Consistency, readability

---

## Tier 2 Recommendations - Moderate Refactoring

These improvements require more coordination but provide substantial architectural benefits.

### 2.1 Consolidate Checkpoint Manager Patterns

**Current State**: Three region-specific checkpoint managers with similar structure:
- `regions/striatum/checkpoint_manager.py` (~900 lines)
- `regions/hippocampus/checkpoint_manager.py` (~600 lines)
- `regions/prefrontal/checkpoint_manager.py` (~400 lines)

**Analysis**: While `BaseCheckpointManager` provides common helpers (as seen in 1.1), the region-specific managers still have structural similarities:
- All have `save_region_specific()` pattern (not found in grep, likely named differently)
- All have weight extraction with sparsity handling
- All have elastic tensor growth logic
- All have similar validation patterns

**Proposed Change**: Strengthen `BaseCheckpointManager` with template method pattern:

```python
# In managers/base_checkpoint_manager.py (ENHANCED):
class BaseCheckpointManager(ABC):
    def save_checkpoint(self, region, path: Path) -> Dict[str, Any]:
        """Template method for saving checkpoints."""
        state = {
            "format_version": self.format_version,
            "neuron_state": self._get_neuron_state(region),
            "pathway_state": self._get_pathway_state(region),
            "learning_state": self._get_learning_state(region),
            "region_specific": self._get_region_specific_state(region),
        }
        self._validate_state(state)
        return state

    @abstractmethod
    def _get_neuron_state(self, region) -> Dict[str, Any]:
        """Extract neuron state (membrane, spikes, etc.)."""

    @abstractmethod
    def _get_pathway_state(self, region) -> Dict[str, Any]:
        """Extract pathway weights and structure."""

    @abstractmethod
    def _get_learning_state(self, region) -> Dict[str, Any]:
        """Extract learning state (traces, STP, etc.)."""

    @abstractmethod
    def _get_region_specific_state(self, region) -> Dict[str, Any]:
        """Extract region-specific state (custom buffers, etc.)."""
```

**Rationale**:
- Enforces consistent checkpoint structure across regions
- Reduces boilerplate in region-specific managers
- Makes it easier to add new checkpoint formats (binary, HDF5, etc.)
- Single point for checkpoint validation and versioning

**Affected Files**:
- [managers/base_checkpoint_manager.py](../../../src/thalia/managers/base_checkpoint_manager.py)
- [regions/striatum/checkpoint_manager.py](../../../src/thalia/regions/striatum/checkpoint_manager.py)
- [regions/hippocampus/checkpoint_manager.py](../../../src/thalia/regions/hippocampus/checkpoint_manager.py)
- [regions/prefrontal/checkpoint_manager.py](../../../src/thalia/regions/prefrontal/checkpoint_manager.py)

**Impact**:
- **Breaking**: Low (internal implementation, external API preserved)
- **Files affected**: 4 files
- **Lines changed**: +150 in base, -100 per region = Net -150 lines
- **Benefits**: Better structure, easier to maintain, clearer separation of concerns

---

### 2.2 Extract Common "grow_output" Logic to GrowthMixin Helper

**Current State**: Every region implements `grow_output()` with similar pattern:
1. Create new neurons with NeuronFactory
2. Expand weights using WeightInitializer
3. Update config sizes
4. Handle elastic tensor capacity

**Example Pattern** (repeated in 10+ files):
```python
def grow_output(self, n_new: int) -> None:
    """Grow output dimension by adding neurons."""
    # Step 1: Create new neurons (DUPLICATED)
    old_size = self.n_neurons
    self.n_neurons += n_new
    # ... create neurons ...

    # Step 2: Expand weights (DUPLICATED)
    for source_name, weights in self.synaptic_weights.items():
        n_input = weights.shape[1]
        new_rows = WeightInitializer.gaussian(n_new, n_input, device=self.device)
        self.synaptic_weights[source_name] = torch.cat([weights, new_rows], dim=0)

    # Step 3: Update config (DUPLICATED)
    self.config.n_output = self.n_neurons
```

**Proposed Change**: Add helper methods to `GrowthMixin`:

```python
# In mixins/growth_mixin.py (NEW helpers):
class GrowthMixin:
    def _grow_neurons(self, n_new: int, neuron_type: str = "pyramidal") -> Any:
        """Helper: Create and append new neurons.

        Returns new neuron module that can be concatenated/appended.
        """
        from thalia.components.neurons import NeuronFactory
        return NeuronFactory.create(neuron_type, n_neurons=n_new, device=self.device)

    def _expand_synaptic_weights_output(
        self,
        n_new: int,
        initialization: str = "gaussian",
        **init_kwargs
    ) -> None:
        """Helper: Expand all synaptic weight matrices by adding output rows.

        Handles multi-source weight dicts automatically.
        """
        from thalia.components.synapses import WeightInitializer
        init_func = getattr(WeightInitializer, initialization)

        for source_name, weights in self.synaptic_weights.items():
            n_input = weights.shape[1]
            new_rows = init_func(n_new, n_input, device=self.device, **init_kwargs)
            self.synaptic_weights[source_name] = torch.cat([weights, new_rows], dim=0)
```

**Usage in Regions** (SIMPLIFIED):
```python
# BEFORE (15+ lines):
def grow_output(self, n_new: int) -> None:
    old_size = self.n_neurons
    self.n_neurons += n_new
    # ... create neurons (5 lines) ...
    # ... expand weights (8 lines) ...
    # ... update config (2 lines) ...

# AFTER (5 lines):
def grow_output(self, n_new: int) -> None:
    self._grow_neurons(n_new, neuron_type="pyramidal")
    self._expand_synaptic_weights_output(n_new, initialization="gaussian", std=0.1)
    self.n_neurons += n_new
    self.config.n_output = self.n_neurons
```

**Rationale**:
- Reduces 10-15 lines per region to 4-5 lines
- Eliminates ~150-200 lines of duplicated logic across codebase
- Makes growth logic more testable (test mixin once)
- Easier to add new initialization strategies
- Clearer intent (method names document what's happening)

**Affected Files** (10+ regions):
- [mixins/growth_mixin.py](../../../src/thalia/mixins/growth_mixin.py)
- [regions/striatum/striatum.py](../../../src/thalia/regions/striatum/striatum.py#L1294)
- [regions/hippocampus/trisynaptic.py](../../../src/thalia/regions/hippocampus/trisynaptic.py#L857)
- [regions/cortex/layered_cortex.py](../../../src/thalia/regions/cortex/layered_cortex.py#L850)
- [regions/prefrontal/prefrontal.py](../../../src/thalia/regions/prefrontal/prefrontal.py#L879)
- [regions/thalamus/thalamus.py](../../../src/thalia/regions/thalamus/thalamus.py#L1021)
- [regions/cerebellum/cerebellum.py](../../../src/thalia/regions/cerebellum/cerebellum.py#L551)
- [regions/multisensory.py](../../../src/thalia/regions/multisensory.py#L602)
- And 3+ more regions

**Impact**:
- **Breaking**: None (internal helper methods)
- **Files affected**: ~12 files (1 mixin + 11 regions)
- **Lines changed**: +50 in mixin, -10 per region = Net -60 lines
- **Benefits**: Less duplication, better testability, clearer code

---

### 2.3 Standardize "Port-Based Routing" Naming Convention

**Current State**: Port-based routing uses different naming styles:
- LayeredCortex: `"l23"`, `"l5"`, `"l6a"`, `"l6b"` (lowercase)
- Thalamus: Should use `"relay"`, `"trn"` but implementation unclear
- Hippocampus: Uses `"ca1"`, `"ca3"` internally but no port routing

**Proposed Convention**:
- **Lowercase with underscores**: `"l2_3"`, `"l5"`, `"l6a"`, `"l6b"`, `"relay"`, `"trn"`, `"ca1"`, `"ca3"`
- **Or**: Uppercase abbreviations: `"L23"`, `"L5"`, `"L6A"`, `"RELAY"`, `"TRN"`, `"CA1"`, `"CA3"`

**Recommendation**: **Lowercase** (more Pythonic, matches attribute naming)

**Rationale**:
- Consistency improves developer experience
- Makes it easier to write routing logic
- Aligns with Python naming conventions (lowercase for module-level names)
- Clearer documentation and examples

**Affected Files**:
- [regions/cortex/layered_cortex.py](../../../src/thalia/regions/cortex/layered_cortex.py) - May need renaming from "l23" to "l2_3"
- [regions/thalamus/thalamus.py](../../../src/thalia/regions/thalamus/thalamus.py) - Port routing implementation
- [pathways/axonal_projection.py](../../../src/thalia/pathways/axonal_projection.py) - Port specification
- Documentation files

**Impact**:
- **Breaking**: Medium (if changing existing ports like "l23" → "l2_3")
- **Files affected**: ~5-8 files
- **Lines changed**: ~20-30 lines
- **Benefits**: Consistency, better developer experience
- **Migration**: Provide backward compatibility layer for 1 release

---

### 2.4 Unify "Learning Strategy" Creation Functions

**Current State**: Multiple helper functions for creating learning strategies:
- `create_strategy()` - Generic factory
- `create_cortex_strategy()` - STDP+BCM composite
- `create_striatum_strategy()` - Three-factor learning
- `create_hippocampus_strategy()` - STDP (one-shot capable)
- `create_cerebellum_strategy()` - Error-corrective

**Issue**: Naming is inconsistent with registry pattern used elsewhere (e.g., `NeuronFactory.create()`, `WeightInitializer.get()`)

**Proposed Change**: Create `LearningStrategyFactory` with preset system:

```python
# In learning/strategy_registry.py (NEW):
class LearningStrategyFactory:
    """Factory for creating learning strategies with region-specific presets."""

    @staticmethod
    def create(strategy_type: str, **kwargs) -> LearningStrategy:
        """Create strategy by name (generic factory)."""
        # Existing create_strategy() logic

    @staticmethod
    def preset(preset_name: str, **overrides) -> LearningStrategy:
        """Create strategy from region-specific preset.

        Args:
            preset_name: "cortex", "striatum", "hippocampus", "cerebellum"
            **overrides: Override preset parameters
        """
        presets = {
            "cortex": lambda: CompositeStrategy([
                STDPStrategy(STDPConfig(...)),
                BCMStrategy(BCMConfig(...))
            ]),
            "striatum": lambda: ThreeFactorStrategy(
                ThreeFactorConfig(eligibility_tau_ms=500.0)
            ),
            "hippocampus": lambda: STDPStrategy(
                STDPConfig(learning_rate=0.1)  # One-shot capable
            ),
            "cerebellum": lambda: ErrorCorrectiveStrategy(
                ErrorCorrectiveConfig(...)
            ),
        }
        strategy = presets[preset_name]()
        # Apply overrides...
        return strategy
```

**Usage**:
```python
# BEFORE:
from thalia.learning import create_cortex_strategy, create_striatum_strategy
cortex_strategy = create_cortex_strategy(use_stdp=True)
striatum_strategy = create_striatum_strategy(eligibility_tau_ms=1000.0)

# AFTER:
from thalia.learning import LearningStrategyFactory
cortex_strategy = LearningStrategyFactory.preset("cortex")
striatum_strategy = LearningStrategyFactory.preset("striatum", eligibility_tau_ms=1000.0)
```

**Rationale**:
- Consistent with other factory patterns (NeuronFactory, WeightInitializer)
- Makes preset system explicit and discoverable
- Easier to add new presets without new functions
- Better documentation (presets listed in one place)
- More testable (test factory once)

**Affected Files**:
- [learning/strategy_registry.py](../../../src/thalia/learning/strategy_registry.py) - Add factory class
- [learning/__init__.py](../../../src/thalia/learning/__init__.py) - Export factory
- All regions using `create_*_strategy()` functions (~8 files)

**Impact**:
- **Breaking**: Low (keep old functions as aliases for 1-2 releases)
- **Files affected**: ~10 files
- **Lines changed**: +100 in registry, -50 in regions (net +50)
- **Benefits**: Better consistency, easier discovery, clearer preset system

---

### 2.5 Extract "Stimulus Gating" Pattern to Shared Component

**Current State**: `StimulusGating` class is duplicated/similar in multiple regions:
- [regions/stimulus_gating.py](../../../src/thalia/regions/stimulus_gating.py) - Standalone module
- Used in: LayeredCortex, TrisynapticHippocampus, Thalamus (all instantiate separately)

**Issue**: Each region instantiates with similar config:
```python
# In LayeredCortex:
self.stimulus_gating = StimulusGating(
    threshold=config.ffi_threshold,
    max_inhibition=config.ffi_strength * 10.0,
    decay_rate=1.0 - (1.0 / config.ffi_tau),
)

# In TrisynapticHippocampus (nearly identical):
self.stimulus_gating = StimulusGating(
    threshold=config.ffi_threshold,
    max_inhibition=config.ffi_strength * 10.0,
    decay_rate=1.0 - (1.0 / config.ffi_tau),
)
```

**Proposed Change**: Create factory function or mixin:

```python
# Option 1: Factory function in stimulus_gating.py:
def create_stimulus_gating(
    threshold: float,
    strength: float,
    tau_ms: float,
) -> StimulusGating:
    """Create StimulusGating with biological parameters.

    Args:
        threshold: Stimulus change detection threshold
        strength: Maximum inhibition strength (scales to appropriate range)
        tau_ms: Decay time constant in milliseconds
    """
    return StimulusGating(
        threshold=threshold,
        max_inhibition=strength * 10.0,  # Scaling factor
        decay_rate=1.0 - (1.0 / tau_ms),
    )

# Option 2: StimulusGatingMixin (if more complex logic needed):
class StimulusGatingMixin:
    def _init_stimulus_gating(self, config) -> StimulusGating:
        """Initialize stimulus gating from config."""
        return create_stimulus_gating(
            config.ffi_threshold,
            config.ffi_strength,
            config.ffi_tau,
        )
```

**Rationale**:
- Eliminates duplicated initialization logic
- Makes parameter scaling explicit (why `* 10.0`?)
- Single point to adjust gating behavior
- Easier to test gating in isolation

**Affected Files**:
- [regions/stimulus_gating.py](../../../src/thalia/regions/stimulus_gating.py) - Add factory
- [regions/cortex/layered_cortex.py](../../../src/thalia/regions/cortex/layered_cortex.py) - Use factory
- [regions/hippocampus/trisynaptic.py](../../../src/thalia/regions/hippocampus/trisynaptic.py) - Use factory
- [regions/thalamus/thalamus.py](../../../src/thalia/regions/thalamus/thalamus.py) - Use factory

**Impact**:
- **Breaking**: None (internal implementation)
- **Files affected**: 4 files
- **Lines changed**: +15 in stimulus_gating.py, -3 per region = Net +6 lines
- **Benefits**: Clearer intent, less duplication, easier maintenance

---

## Tier 3 Recommendations - Major Restructuring

These are long-term architectural improvements requiring significant coordination.

### 3.1 Consider "Learning Rules" Package Reorganization

**Current State**: Learning rules spread across multiple locations:
```
learning/
  rules/
    strategies.py       # ~1200 lines: All strategy classes
    bcm.py              # BCM-specific implementation
    __init__.py
  eligibility/
    trace_manager.py    # Eligibility trace management
    __init__.py
  homeostasis/
    synaptic_homeostasis.py
    intrinsic_plasticity.py
    metabolic.py
    __init__.py
  strategy_registry.py  # Registration and factory functions
  strategy_mixin.py     # Mixin for regions
  critical_periods.py   # Critical period learning
  ei_balance.py         # E/I balance
  social_learning.py    # Social learning mechanisms
```

**Issue**: `strategies.py` is very large (1200+ lines) and contains multiple strategy classes:
- HebbianStrategy
- STDPStrategy
- BCMStrategy
- ThreeFactorStrategy
- ErrorCorrectiveStrategy
- CompositeStrategy

**Proposed Change**: Split strategies into separate files:

```
learning/
  strategies/              # NEW: One file per strategy
    base.py               # Base classes and protocols
    hebbian.py            # HebbianStrategy (~100 lines)
    stdp.py               # STDPStrategy (~200 lines)
    bcm.py                # BCMStrategy (move from rules/)
    three_factor.py       # ThreeFactorStrategy (~200 lines)
    error_corrective.py   # ErrorCorrectiveStrategy (~150 lines)
    composite.py          # CompositeStrategy (~100 lines)
    __init__.py           # Export all strategies
  eligibility/            # Keep as-is
  homeostasis/            # Keep as-is
  factory.py              # NEW: LearningStrategyFactory (from 2.4)
  mixin.py                # Renamed from strategy_mixin.py
```

**Rationale**:
- Improved discoverability (one strategy per file)
- Easier to maintain and test individual strategies
- Follows single-responsibility principle
- Aligns with other component organization (neurons/, synapses/)
- Makes it easier to add new strategies without bloating one file

**Impact**:
- **Breaking**: Medium-High (many imports will change)
- **Files affected**: ~30-40 files (all regions that import strategies)
- **Lines changed**: ~1200 lines moved, ~50 import updates
- **Benefits**: Better organization, easier maintenance, clearer structure
- **Migration Path**:
  1. Create new structure alongside old
  2. Add backward-compatible imports in `learning/__init__.py`
  3. Update all regions over 2-3 releases
  4. Remove old structure

**Caution**: This is disruptive and should be carefully planned. Consider if current organization is actually problematic before proceeding.

---

### 3.2 Potential "Regions" Subpackage Reorganization

**Current State**: Regions directory is flat with subpackages:
```
regions/
  cerebellum/
    __init__.py
    cerebellum.py
    state.py
    granule_layer.py
    purkinje_cell.py
    deep_nuclei.py
  cortex/
    __init__.py
    layered_cortex.py
    predictive_cortex.py
    state.py
  hippocampus/
    __init__.py
    trisynaptic.py
    state.py
    learning_component.py
    checkpoint_manager.py
    spontaneous_replay.py
    synaptic_tagging.py
  prefrontal/
    __init__.py
    prefrontal.py
    state.py
    checkpoint_manager.py
    goal_emergence.py
  striatum/
    __init__.py
    striatum.py
    state.py
    d1_pathway.py
    d2_pathway.py
    pathway_base.py
    learning_component.py
    homeostasis_component.py
    exploration.py
    action_selection.py
    forward_coordinator.py
    checkpoint_manager.py
  thalamus/
    __init__.py
    thalamus.py
    state.py
  multisensory.py          # <-- Should this be a package?
  stimulus_gating.py       # <-- Should this be in components/?
```

**Issues**:
1. `multisensory.py` is a single file but could be a package (like other regions)
2. `stimulus_gating.py` is a shared component used by multiple regions - should it be in `components/` or `utils/`?
3. Inconsistent packaging: Some regions are simple (thalamus/), others complex (striatum/)

**Proposed Changes**:

**Option A - No Change (RECOMMENDED)**:
- Current structure is actually fine
- `multisensory.py` is simple enough to stay as single file
- `stimulus_gating.py` location is acceptable (it's region-adjacent)
- Inconsistent complexity is natural (striatum IS more complex biologically)

**Option B - Reorganize**:
1. Move `stimulus_gating.py` → `components/stimulus_gating.py`
2. Convert `multisensory.py` → `multisensory/` package (only if it grows)
3. Add README.md to each region subpackage explaining organization

**Recommendation**: **Option A (no change)** - Current organization works well. The apparent inconsistency reflects biological reality (some regions are inherently more complex). Only reorganize if specific pain points emerge.

**Impact**: N/A (no change recommended)

---

### 3.3 Evaluate "Large File" Justifications

**Current State**: Several files exceed 1000 lines:
- [regions/striatum/striatum.py](../../../src/thalia/regions/striatum/striatum.py) - 3661 lines
- [regions/hippocampus/trisynaptic.py](../../../src/thalia/regions/hippocampus/trisynaptic.py) - 2770 lines
- [regions/cortex/layered_cortex.py](../../../src/thalia/regions/cortex/layered_cortex.py) - 2422 lines
- [learning/rules/strategies.py](../../../src/thalia/learning/rules/strategies.py) - 1209 lines

**Documented Justification** (from ADR-011):
> "The striatum coordinates two opponent pathways (D1 'Go', D2 'No-Go') that must interact every timestep for action selection. Splitting would:
> 1. Require passing D1/D2 votes, eligibility, action selection state
> 2. Duplicate dopamine broadcast logic
> 3. Obscure the opponent pathway interaction
> 4. Break action selection coherence"

**Analysis**:
- **Striatum (3661 lines)**: Justification is reasonable. File has good internal structure (navigation comments at top). Components ARE extracted where appropriate (D1Pathway, D2Pathway, learning/homeostasis/exploration components).
- **Hippocampus (2770 lines)**: Trisynaptic circuit (DG→CA3→CA1) is tightly coupled. Similar justification applies.
- **LayeredCortex (2422 lines)**: Laminar architecture (L4→L2/3→L5→L6) requires coordination. Reasonable size.
- **strategies.py (1209 lines)**: This SHOULD be split (see 3.1)

**Recommendation**:
- **Accept**: Striatum, Hippocampus, LayeredCortex sizes are justified
- **Consider**: Split strategies.py (Tier 3.1)
- **Monitor**: If files exceed 4000 lines, revisit justification

**Action**: No immediate action required. Large files are justified and well-documented.

---

## Risk Assessment & Sequencing

### Implementation Priority

**Phase 1 (Immediate - Low Risk)**:
1. Replace manual weight initialization with WeightInitializer (1.2)
2. Extract magic numbers to constants (1.3)
3. Standardize WeightInitializer import path (1.6)

**Estimated Effort**: 2-3 days
**Risk**: Low (internal implementation, well-tested)

---

**Phase 2 (Short-term - Low-Medium Risk)**:
1. Simplify D1/D2 dopamine modulation methods (1.5)
2. Fix "learning_strategy" vs "learning_rule" naming (1.4)
3. Extract grow_output() logic to mixin helpers (2.2)

**Estimated Effort**: 3-5 days
**Risk**: Low-Medium (some API surface changes, needs testing)

---

**Phase 3 (Medium-term - Medium Risk)**:
1. Strengthen checkpoint manager template (2.1)
2. Standardize port-based routing naming (2.3)
3. Extract stimulus gating initialization (2.5)
4. Unify learning strategy creation (2.4)

**Estimated Effort**: 5-8 days
**Risk**: Medium (requires coordination, some breaking changes)

---

**Phase 4 (Long-term - High Risk)**:
1. Split strategies.py into separate files (3.1)
2. Evaluate regions/ organization (3.2)

**Estimated Effort**: 10-15 days
**Risk**: High (many imports change, significant testing required)
**Recommendation**: Only if clear pain points emerge

---

### Testing Strategy

For each phase:
1. **Unit Tests**: Verify component behavior unchanged
2. **Integration Tests**: Verify brain-level training unchanged
3. **Regression Tests**: Run full curriculum training (Stage 0-3)
4. **Performance Tests**: Verify no performance degradation

**Critical Tests**:
- Checkpoint save/load roundtrip (after Phase 3)
- All regions can grow dynamically (after Phase 2)
- Learning strategies function correctly (after Phase 2-3)

---

## Appendix A: Affected Files

### High-Priority Files (Phase 1-2)

**Weight Initialization** (1.2):
- [src/thalia/regions/hippocampus/trisynaptic.py](../../src/thalia/regions/hippocampus/trisynaptic.py)
- [src/thalia/regions/cortex/layered_cortex.py](../../src/thalia/regions/cortex/layered_cortex.py)
- [src/thalia/regions/striatum/striatum.py](../../src/thalia/regions/striatum/striatum.py)
- [src/thalia/regions/prefrontal/prefrontal.py](../../src/thalia/regions/prefrontal/prefrontal.py)
- [src/thalia/regions/thalamus/thalamus.py](../../src/thalia/regions/thalamus/thalamus.py)
- [src/thalia/regions/cerebellum/cerebellum.py](../../src/thalia/regions/cerebellum/cerebellum.py)
- [src/thalia/components/neurons/dendritic.py](../../src/thalia/components/neurons/dendritic.py)
- [src/thalia/components/synapses/stp.py](../../src/thalia/components/synapses/stp.py)

**Constants** (1.3):
- [src/thalia/constants/neuron.py](../../src/thalia/constants/neuron.py)
- [src/thalia/constants/learning.py](../../src/thalia/constants/learning.py)

**Pathways** (1.5):
- [src/thalia/regions/striatum/pathway_base.py](../../src/thalia/regions/striatum/pathway_base.py)
- [src/thalia/regions/striatum/d1_pathway.py](../../src/thalia/regions/striatum/d1_pathway.py)
- [src/thalia/regions/striatum/d2_pathway.py](../../src/thalia/regions/striatum/d2_pathway.py)

**Core** (1.4):
- [src/thalia/core/neural_region.py](../../src/thalia/core/neural_region.py)
- [src/thalia/learning/strategy_mixin.py](../../src/thalia/learning/strategy_mixin.py)

---

### Medium-Priority Files (Phase 3)

**Checkpoint Managers** (2.1):
- [src/thalia/managers/base_checkpoint_manager.py](../../src/thalia/managers/base_checkpoint_manager.py)
- [src/thalia/regions/striatum/checkpoint_manager.py](../../src/thalia/regions/striatum/checkpoint_manager.py)
- [src/thalia/regions/hippocampus/checkpoint_manager.py](../../src/thalia/regions/hippocampus/checkpoint_manager.py)
- [src/thalia/regions/prefrontal/checkpoint_manager.py](../../src/thalia/regions/prefrontal/checkpoint_manager.py)

**Growth Helpers** (2.2):
- [src/thalia/mixins/growth_mixin.py](../../src/thalia/mixins/growth_mixin.py)
- All region files with grow_output() implementations

**Learning Strategy Factory** (2.4):
- [src/thalia/learning/strategy_registry.py](../../src/thalia/learning/strategy_registry.py)
- [src/thalia/learning/__init__.py](../../src/thalia/learning/__init__.py)

---

## Appendix B: Detected Duplications

### B.1 Weight Initialization Duplication

**Pattern**: Manual `torch.randn()` / `torch.rand()` usage

**Locations** (30+ instances):

**Critical (Region Initialization)**:
1. `regions/hippocampus/trisynaptic.py:672` - Phase modulation jitter
2. `regions/hippocampus/trisynaptic.py:573` - `_initialize_weights()` method
3. `regions/cortex/layered_cortex.py:396` - `_initialize_weights()` method
4. `regions/striatum/striatum.py:1912` - `_initialize_pathway_weights()` method
5. `regions/prefrontal/prefrontal.py:975` - Manual weight growth
6. `regions/thalamus/thalamus.py:379` - `_initialize_weights()` method
7. `regions/cerebellum/cerebellum.py:406` - `_initialize_weights_tensor()` method

**Proposed Fix**: Use `WeightInitializer.gaussian()` or `WeightInitializer.uniform()`

**Code Changes Needed**: ~50-80 lines across 30 files

---

### B.2 D1/D2 Pathway Dopamine Modulation Duplication

**Pattern**: Near-identical `apply_dopamine_modulation()` methods

**Locations**:
- `regions/striatum/d1_pathway.py:47-95` (48 lines)
- `regions/striatum/d2_pathway.py:53-105` (52 lines)

**Difference**: Single line (dopamine polarity: `dopamine` vs `-dopamine`)

**Proposed Fix**: Move to base class with polarity attribute (see Tier 1.5)

**Code Changes**: -90 lines (duplicated code) + 30 lines (base implementation) = **Net -60 lines**

---

### B.3 Stimulus Gating Initialization Duplication

**Pattern**: Identical `StimulusGating()` instantiation with formula

**Locations**:
- `regions/cortex/layered_cortex.py:395` (3 lines)
- `regions/hippocampus/trisynaptic.py:292` (3 lines)
- `regions/thalamus/thalamus.py:XXX` (estimated, not confirmed)

**Formula**:
```python
StimulusGating(
    threshold=config.ffi_threshold,
    max_inhibition=config.ffi_strength * 10.0,  # Why 10.0?
    decay_rate=1.0 - (1.0 / config.ffi_tau),
)
```

**Proposed Fix**: Factory function `create_stimulus_gating()` (see Tier 2.5)

**Code Changes**: ~9 lines (3 regions × 3 lines) → ~3 lines (factory call) = **Net -6 lines + clarity**

---

### B.4 grow_output() Pattern Duplication

**Pattern**: Similar grow_output() implementations across regions

**Locations** (10+ regions):
- `regions/striatum/striatum.py:1294`
- `regions/hippocampus/trisynaptic.py:857`
- `regions/cortex/layered_cortex.py:850`
- `regions/prefrontal/prefrontal.py:879`
- `regions/thalamus/thalamus.py:1021`
- `regions/cerebellum/cerebellum.py:551`
- `regions/multisensory.py:602`
- `regions/cortex/predictive_cortex.py:995`
- Plus others...

**Common Pattern** (15 lines per region):
1. Create new neurons (5 lines)
2. Expand weight matrices (8 lines)
3. Update config (2 lines)

**Proposed Fix**: Mixin helpers `_grow_neurons()` and `_expand_synaptic_weights_output()` (see Tier 2.2)

**Code Changes**: ~150 lines (10 regions × 15 lines) → ~50 lines (5 lines per region) = **Net -100 lines**

---

### B.5 Checkpoint Manager Structure Duplication

**Pattern**: Similar structure in region-specific checkpoint managers

**Locations**:
- `regions/striatum/checkpoint_manager.py` (900 lines)
- `regions/hippocampus/checkpoint_manager.py` (600 lines)
- `regions/prefrontal/checkpoint_manager.py` (400 lines)

**Common Elements**:
- Weight extraction with sparsity (~50 lines each)
- Elastic tensor metadata (~30 lines each)
- Validation logic (~40 lines each)
- State packaging (~60 lines each)

**Proposed Fix**: Strengthen `BaseCheckpointManager` template methods (see Tier 2.1)

**Code Changes**: ~180 lines (3 × 60 duplicated) → ~60 lines (template overhead) = **Net -120 lines**

---

## Summary Statistics

**Total Duplication Identified**: ~400-500 lines across 50+ locations

**Potential Line Reduction**:
- Phase 1-2: -150 lines (weight init, D1/D2, constants)
- Phase 3: -250 lines (checkpoints, growth, gating)
- **Total**: ~400 lines reduced while improving clarity

**Code Quality Impact**:
- Reduced duplication: ✅ Significant improvement
- Improved testability: ✅ More focused tests possible
- Better maintainability: ✅ Single source of truth
- Clearer architecture: ✅ Patterns more explicit

---

## Conclusion

The Thalia codebase demonstrates strong architectural foundations with excellent biological plausibility and well-designed patterns. The identified issues are typical of a maturing codebase and can be addressed incrementally without major disruption.

**Recommended Immediate Actions**:
1. Implement Phase 1 improvements (weight initialization, constants) - **2-3 days**
2. Address Phase 2 naming and helper consolidation - **3-5 days**
3. Plan Phase 3 template strengthening with stakeholder input - **5-8 days**
4. Defer Phase 4 (major restructuring) unless clear pain points emerge

**Long-term Maintenance**:
- Continue monitoring large files (>2000 lines) for split opportunities
- Keep checkpoint managers aligned with template pattern
- Enforce WeightInitializer registry usage in code reviews
- Update documentation to reflect new patterns as they're adopted

The codebase is in good health. These recommendations will make it even better.

---

**Review Completed**: January 25, 2026
**Next Review Recommended**: After implementing Phase 1-2 (approximately 3 months)
