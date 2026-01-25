# Architecture Review – 2026-01-25

## Executive Summary

This comprehensive architectural analysis of the Thalia codebase evaluates module organization, naming consistency, separation of concerns, pattern adherence, code duplication, and antipatterns across the `src/thalia/` directory. The review focuses on core components (regions, learning, pathways, components) and their adherence to biological plausibility constraints and documented patterns.

**Key Findings:**
- **Strong foundation**: Registry patterns (ComponentRegistry, LearningStrategyRegistry, WeightInitializer, NeuronFactory) provide excellent pluggability and discoverability
- **Effective mixins**: DiagnosticsMixin, GrowthMixin, and other mixins successfully eliminate code duplication
- **Good biological accuracy**: Learning strategies properly implement local rules, spike-based processing is consistently used
- **Areas for improvement**: Some magic numbers remain inline, minor naming inconsistencies, opportunities for further extraction of shared patterns
- **Overall health**: The codebase demonstrates mature architectural patterns with clear separation of concerns

---

## Tier 1 - High Impact, Low Disruption

### 1.1 Extract Magic Numbers to Named Constants

**Current State:**
Several files contain inline magic numbers that should be moved to the `constants/` module for better maintainability and discoverability.

**Locations:**
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py#L395): `reserve_multiplier = 1.0 + self.config.reserve_capacity` (growth capacity calculation)
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py#L315): Various layer-specific scaling factors embedded in initialization
- [src/thalia/components/synapses/weight_init.py](src/thalia/components/synapses/weight_init.py): Connectivity thresholds and topographic decay constants

**Proposed Change:**
Create `src/thalia/constants/growth.py` and `src/thalia/constants/connectivity.py`:

```python
# constants/growth.py
RESERVE_CAPACITY_MULTIPLIER = 1.0  # For elastic tensor pre-allocation
GROWTH_NEW_WEIGHT_SCALE = 0.2      # Already exists, ensure it's used consistently

# constants/connectivity.py
CORTICAL_SPARSE_CONNECTIVITY = 0.15  # Typical cortical connectivity (~15%)
HIPPOCAMPAL_SPARSE_CONNECTIVITY = 0.05  # DG->CA3 sparse connectivity
STRIATAL_FSI_RATIO = 0.02  # FSI as percentage of MSNs
STRIATAL_LATERAL_SPARSITY = 0.35  # MSN->MSN lateral inhibition
```

**Rationale:**
- Improves maintainability (single source of truth)
- Enhances discoverability (developers know where to find constants)
- Facilitates experimentation (easy to adjust and track parameter changes)
- Documents biological basis in constant definitions

**Impact:**
- Files affected: 8-10 region files
- Breaking changes: None (internal refactoring)
- Severity: Low

---

### 1.2 Standardize Size Dictionary Keys ✅ IMPLEMENTED

**Status**: ✅ **Complete** (January 25, 2026)

**Implementation Summary:**
- ✅ Created comprehensive documentation in [docs/patterns/size-dictionaries.md](docs/patterns/size-dictionaries.md)
- ✅ Documented standard keys (`input_size`, region-specific layer names)
- ✅ Provided examples for all major regions (Cortex, Hippocampus, Striatum, Cerebellum, Thalamus)
- ✅ Documented integration with LayerSizeCalculator
- ✅ Verified 6 out of 7 regions already follow the pattern

**Current State:**
Region initialization uses size dictionaries with consistent naming:

```python
# Striatum (follows pattern):
sizes = {"n_actions": 4, "d1_size": 40, "d2_size": 40, "input_size": 256}

# LayeredCortex (follows pattern):
sizes = {"l4_size": 64, "l23_size": 96, "l5_size": 32, "input_size": 256}

# Hippocampus (follows pattern):
sizes = {"dg_size": 512, "ca3_size": 256, "ca1_size": 256, "input_size": 128}
```

**Known Exception:**
- MultimodalIntegration (1 region): Still uses old pattern (sizes in config)
- **Reason**: Lower priority region, not actively used in current experiments
- **Migration**: Deferred to future refactoring when region is updated

**Verified Compliant Regions:**
1. ✅ Striatum - uses `n_actions`, `d1_size`, `d2_size`, `input_size`
2. ✅ LayeredCortex - uses `l4_size`, `l23_size`, `l5_size`, `l6a_size`, `l6b_size`, `input_size`
3. ✅ TrisynapticHippocampus - uses `dg_size`, `ca3_size`, `ca2_size`, `ca1_size`, `input_size`
4. ✅ Prefrontal - uses `n_neurons`, `input_size`
5. ✅ Cerebellum - uses `granule_size`, `purkinje_size`, `input_size`
6. ✅ ThalamicRelay - uses `relay_size`, `trn_size`, `input_size`

**Documentation Delivered:**
- Pattern specification with naming guidelines
- Integration with LayerSizeCalculator
- Examples for all major regions
- Migration notes from pre-2026 pattern
- Validation recommendations

**Impact:**
- Files affected: 1 new documentation file
- Breaking changes: None (documentation only)
- Severity: Low

---

### 1.3 Consolidate Neuron Type Comments/Documentation

**Current State:**
Neuron type descriptions are scattered across multiple files with some duplication:
- [src/thalia/components/neurons/neuron_factory.py](src/thalia/components/neurons/neuron_factory.py): Factory methods with inline docstrings
- [src/thalia/constants/neuron.py](src/thalia/constants/neuron.py): Constants definitions
- Region files: Comments about neuron types when creating populations

**Proposed Change:**
Create `docs/api/NEURON_TYPES.md` with canonical neuron type documentation:

```markdown
# Neuron Types in Thalia

## Pyramidal Neurons
**Factory**: `NeuronFactory.create("pyramidal", ...)`
**Used in**: Cortex (L2/3, L5), Hippocampus (CA1, CA3)
**Properties**:
- tau_mem: 20ms (integration time constant)
- Spike-frequency adaptation: 0.05 (moderate)
- Biological basis: Excitatory projection neurons

## Fast-Spiking Interneurons (FSI)
**Factory**: `create_fast_spiking_neurons(n, device)`
**Used in**: Striatum, Cortex (PV+ interneurons)
**Properties**:
- tau_mem: 5ms (fast kinetics)
- No adaptation
- Biological basis: Parvalbumin+ GABAergic interneurons
```

**Rationale:**
- Single source of truth for neuron type documentation
- Easier to maintain than scattered comments
- Improves onboarding for new developers

**Impact:**
- Files affected: Documentation only, no code changes
- Breaking changes: None
- Severity: Low

---

### 1.4 Remove Minimal torch.randn/rand Occurrences

**Current State:**
Two instances of direct `torch.rand()` calls remain (all others use WeightInitializer):

- [src/thalia/regions/hippocampus/spontaneous_replay.py#L114](src/thalia/regions/hippocampus/spontaneous_replay.py#L114): `if torch.rand(1).item() < prob:` (probabilistic replay trigger)
- [src/thalia/regions/striatum/action_selection.py#L266](src/thalia/regions/striatum/action_selection.py#L266): `if explore and torch.rand(1).item() < exploration_prob:` (exploration sampling)

**Proposed Change:**
These are correct uses (random sampling for probabilistic events, not weight initialization). Add clarifying comments:

```python
# OK: Random sampling for probabilistic event (not weight initialization)
if torch.rand(1, device=device).item() < prob:
    self._trigger_replay()
```

**Rationale:**
- Distinguishes intentional random sampling from weight initialization
- Prevents future confusion about WeightInitializer usage
- Minor fix: add device parameter for consistency

**Impact:**
- Files affected: 2 files
- Breaking changes: None
- Severity: Low

---

### 1.5 Standardize reset_state() Signature Across Components ✅ IMPLEMENTED

**Status**: ✅ **Complete** (January 25, 2026)

**Implementation Summary:**
- ✅ Updated ResettableMixin docstring with standard signature documentation
- ✅ Documented what should be reset (membrane potentials, traces, conductances)
- ✅ Documented what should NOT be reset (weights, structural parameters)
- ✅ Enforced signature: `reset_state(self) -> None` (no optional parameters)

**Current State:**
Most regions implement `reset_state(self) -> None`, but there's minor inconsistency in whether they accept optional parameters:

```python
# Most regions (consistent):
def reset_state(self) -> None:
    """Reset all state to initial values."""

# Some subcomponents:
def reset_state(self, full_reset: bool = False) -> None:
    """Reset with optional partial reset."""
```

**Proposed Change:**
Document standard signature in ResettableMixin docstring:

```python
def reset_state(self) -> None:
    """Reset component to initial state.

    Components should reset:
    - Neuron membrane potentials and refractory states
    - Synaptic conductances
    - Learning traces (eligibility, STDP, BCM)
    - Activity history and homeostatic variables

    Do NOT reset:
    - Synaptic weights (learned knowledge)
    - Structural parameters (neuron counts, connectivity)
    """
```

**Documentation Delivered:**
- Standard signature specification: `reset_state(self) -> None`
- Clear list of what should be reset (dynamic state)
- Clear list of what should NOT be reset (learned knowledge)
- Note about no optional parameters for consistency

**Rationale:**
- Makes reset behavior predictable across components
- Clarifies what should/shouldn't be reset
- Supports proper trial-based training

**Impact:**
- Files affected: 1 file (ResettableMixin docstring)
- Breaking changes: None (documentation only)
- Severity: Low

---

## Tier 2 - Moderate Refactoring

### 2.1 Consolidate Eligibility Trace Management Patterns

**Current State:**
Multiple regions implement similar eligibility trace logic with slight variations:

**Striatum** ([src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py#L280-L295)):
```python
self._eligibility_d1: StateDict = {}
self._eligibility_d2: StateDict = {}
if config.use_multiscale_eligibility:
    self._eligibility_d1_fast: StateDict = {}
    self._eligibility_d2_fast: StateDict = {}
    self._eligibility_d1_slow: StateDict = {}
    self._eligibility_d2_slow: StateDict = {}
```

**Hippocampus** ([src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py)):
```python
# Similar pattern but single-pathway eligibility
self._ca3_ca1_eligibility: Optional[torch.Tensor] = None
self._dg_ca3_eligibility: Optional[torch.Tensor] = None
```

**Proposed Change:**
Extract common eligibility management to `EligibilityTraceManager` (already exists in `src/thalia/learning/eligibility/`), expand it to handle multi-pathway and multi-timescale cases:

```python
# In learning/eligibility/trace_manager.py
class EligibilityTraceManager:
    """Manages eligibility traces for multi-source, multi-timescale learning."""

    def __init__(self, tau_ms: float, use_multiscale: bool = False,
                 tau_fast_ms: Optional[float] = None, tau_slow_ms: Optional[float] = None):
        self.tau_ms = tau_ms
        self.use_multiscale = use_multiscale
        self.traces: Dict[str, torch.Tensor] = {}
        if use_multiscale:
            self.traces_fast: Dict[str, torch.Tensor] = {}
            self.traces_slow: Dict[str, torch.Tensor] = {}

    def add_source(self, source_name: str, shape: tuple, device: torch.device):
        """Register a new input source for eligibility tracking."""
        self.traces[source_name] = torch.zeros(shape, device=device)
        if self.use_multiscale:
            self.traces_fast[source_name] = torch.zeros(shape, device=device)
            self.traces_slow[source_name] = torch.zeros(shape, device=device)

    def update(self, source_name: str, pre_spikes: torch.Tensor,
               post_spikes: torch.Tensor, dt_ms: float):
        """Update eligibility trace with pre-post spike coincidence."""
        # Standard update
        decay = torch.exp(torch.tensor(-dt_ms / self.tau_ms))
        self.traces[source_name] *= decay
        self.traces[source_name] += torch.outer(post_spikes, pre_spikes)

        if self.use_multiscale:
            # Fast trace update (tau ~500ms)
            # Slow trace update (tau ~60s)
            pass  # Implementation details

    def get_combined(self, source_name: str, alpha_slow: float = 0.3) -> torch.Tensor:
        """Get combined eligibility (fast + alpha * slow)."""
        if not self.use_multiscale:
            return self.traces[source_name]
        return self.traces_fast[source_name] + alpha_slow * self.traces_slow[source_name]
```

**Usage in Regions:**
```python
# In Striatum.__init__
self.eligibility_manager = EligibilityTraceManager(
    tau_ms=config.eligibility_tau_ms,
    use_multiscale=config.use_multiscale_eligibility
)

# When adding a source
self.eligibility_manager.add_source(
    f"{source_name}_d1",
    shape=(self.d1_size, source_size),
    device=self.device
)
```

**Rationale:**
- Eliminates duplicated eligibility logic across 3+ regions
- Centralizes multi-timescale implementation (Striatum-specific feature becomes reusable)
- Makes eligibility trace management consistent and testable
- Easier to add new eligibility variants (e.g., dopamine-modulated decay)

**Impact:**
- Files affected: Striatum, Hippocampus, potentially Prefrontal
- Breaking changes: Medium (internal refactoring, checkpoint format preserved)
- Severity: Medium
- Estimated effort: 4-6 hours

---

### 2.2 Standardize Pathway Configuration Patterns

**Current State:**
Pathways have inconsistent configuration initialization:

**AxonalProjection** ([src/thalia/pathways/axonal_projection.py](src/thalia/pathways/axonal_projection.py#L134-L148)):
```python
def __init__(self, sources: List[Tuple[...]], device: str = "cpu",
             dt_ms: float = 1.0, config: Optional[...] = None):
    if config is None:
        config = SimpleNamespace(device=device)  # Minimal config
```

**Striatum Pathways** ([src/thalia/regions/striatum/pathway_base.py](src/thalia/regions/striatum/pathway_base.py#L104)):
```python
def __init__(self, config: StriatumPathwayConfig):
    # Requires full config object
```

**Proposed Change:**
Standardize pathway initialization to always accept a typed config dataclass:

```python
@dataclass
class AxonalProjectionConfig(NeuralComponentConfig):
    """Configuration for AxonalProjection."""
    sources: List[SourceSpec]
    dt_ms: float = 1.0

    # Inherited from NeuralComponentConfig:
    # - device: str
    # - n_input: int
    # - n_output: int

# Usage
class AxonalProjection(RoutingComponent):
    def __init__(self, config: AxonalProjectionConfig):
        super().__init__(config)
        self.sources = config.sources
        self.dt_ms = config.dt_ms
```

**Rationale:**
- Matches region initialization pattern (config + sizes + device)
- Enables proper type checking and validation
- Makes BrainBuilder logic more consistent
- Eliminates SimpleNamespace hack

**Impact:**
- Files affected: AxonalProjection, BrainBuilder
- Breaking changes: Medium (pathway creation code needs updates)
- Severity: Medium
- Estimated effort: 3-4 hours

---

### 2.3 Extract Inter-Layer Weight Initialization Pattern

**Current State:**
LayeredCortex and TrisynapticHippocampus have similar patterns for initializing inter-layer weights:

**LayeredCortex** ([src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py#L556-L620)):
```python
def _init_weights(self) -> None:
    """Initialize inter-layer connection weights."""
    # L4→L2/3
    self.l4_to_l23_weights = WeightInitializer.gaussian(
        self.l23_size, self.l4_size, mean=0.3, std=0.1, device=self.device
    )

    # L2/3→L2/3 (recurrent)
    self.l23_recurrent_weights = WeightInitializer.sparse_random(
        self.l23_size, self.l23_size, sparsity=0.85, device=self.device
    )
    # ... more layers
```

**TrisynapticHippocampus** ([src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py)):
```python
def _init_trisynaptic_weights(self) -> None:
    """Initialize DG→CA3→CA1 pathway weights."""
    # DG→CA3
    self.dg_to_ca3_weights = WeightInitializer.sparse_random(
        self.ca3_size, self.dg_size, sparsity=0.95, device=self.device
    )

    # CA3→CA3 (recurrent)
    self.ca3_recurrent_weights = WeightInitializer.sparse_random(
        self.ca3_size, self.ca3_size, sparsity=0.90, device=self.device
    )
    # ... more layers
```

**Proposed Change:**
Create `WeightInitializer.layer_cascade()` helper for sequential layer initialization:

```python
# In components/synapses/weight_init.py
@staticmethod
def layer_cascade(
    layer_specs: List[Tuple[str, int, int, dict]],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Initialize weights for a cascade of layers.

    Args:
        layer_specs: List of (name, n_output, n_input, init_kwargs)
        device: Device for tensors

    Returns:
        Dict mapping layer names to weight tensors

    Example:
        >>> specs = [
        ...     ("l4_to_l23", 96, 64, {"method": "gaussian", "mean": 0.3}),
        ...     ("l23_recurrent", 96, 96, {"method": "sparse_random", "sparsity": 0.85}),
        ...     ("l23_to_l5", 32, 96, {"method": "gaussian", "mean": 0.2}),
        ... ]
        >>> weights = WeightInitializer.layer_cascade(specs, device)
    """
    weights = {}
    for name, n_out, n_in, kwargs in layer_specs:
        method = kwargs.pop("method", "gaussian")
        init_func = getattr(WeightInitializer, method)
        weights[name] = init_func(n_out, n_in, device=device, **kwargs)
    return weights
```

**Usage:**
```python
# In LayeredCortex._init_weights()
specs = [
    ("l4_to_l23", self.l23_size, self.l4_size,
     {"method": "gaussian", "mean": 0.3, "std": 0.1}),
    ("l23_recurrent", self.l23_size, self.l23_size,
     {"method": "sparse_random", "sparsity": 0.85}),
    ("l23_to_l5", self.l5_size, self.l23_size,
     {"method": "gaussian", "mean": 0.2, "std": 0.1}),
]
layer_weights = WeightInitializer.layer_cascade(specs, self.device)
self.l4_to_l23_weights = nn.Parameter(layer_weights["l4_to_l23"])
# ...
```

**Rationale:**
- Reduces repetitive weight initialization code
- Makes layer connectivity structure more declarative and readable
- Centralizes inter-layer initialization logic
- Easier to experiment with different initialization schemes

**Impact:**
- Files affected: LayeredCortex, TrisynapticHippocampus, WeightInitializer
- Breaking changes: Low (internal refactoring)
- Severity: Medium
- Estimated effort: 3-4 hours

---

### 2.4 Create ForwardPassCoordinator Base Class

**Current State:**
Striatum has a `ForwardPassCoordinator` component ([src/thalia/regions/striatum/forward_coordinator.py](src/thalia/regions/striatum/forward_coordinator.py)), but LayeredCortex and Hippocampus implement similar coordination logic inline.

**LayeredCortex forward cascade:**
```python
def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
    # L4 processing
    l4_spikes = self._process_l4(source_spikes)
    # L2/3 processing
    l23_spikes = self._process_l23(l4_spikes)
    # L5 processing
    l5_spikes = self._process_l5(l23_spikes)
    # Return concatenated output
    return torch.cat([l23_spikes, l5_spikes])
```

**Proposed Change:**
Extract base class in `src/thalia/core/forward_coordinator.py`:

```python
class ForwardPassCoordinator(ABC):
    """Base class for coordinating multi-stage forward passes.

    Regions with sequential processing stages (L4→L2/3→L5, DG→CA3→CA1)
    can subclass this to standardize their forward pass coordination.
    """

    @abstractmethod
    def get_stage_order(self) -> List[str]:
        """Return ordered list of processing stages."""
        pass

    @abstractmethod
    def process_stage(self, stage_name: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process a single stage and return outputs."""
        pass

    def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute all stages in order."""
        stage_outputs = {}
        for stage in self.get_stage_order():
            stage_outputs[stage] = self.process_stage(stage, stage_outputs)
        return self.combine_outputs(stage_outputs)

    @abstractmethod
    def combine_outputs(self, stage_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine stage outputs into final output."""
        pass
```

**Rationale:**
- Standardizes multi-stage processing pattern
- Makes forward pass logic more testable (can test individual stages)
- Reduces boilerplate in complex regions
- Provides hooks for diagnostics and monitoring

**Impact:**
- Files affected: LayeredCortex, TrisynapticHippocampus, Striatum (refactor existing), new base class
- Breaking changes: Low (internal refactoring)
- Severity: Medium
- Estimated effort: 6-8 hours

---

### 2.5 Consolidate Checkpoint Manager Pattern

**Current State:**
Three regions implement checkpoint managers with similar structure:

- [src/thalia/regions/striatum/checkpoint_manager.py](src/thalia/regions/striatum/checkpoint_manager.py)
- [src/thalia/regions/hippocampus/checkpoint_manager.py](src/thalia/regions/hippocampus/checkpoint_manager.py)
- [src/thalia/regions/prefrontal/checkpoint_manager.py](src/thalia/regions/prefrontal/checkpoint_manager.py)

All follow similar pattern:
```python
class RegionCheckpointManager:
    def __init__(self, region: RegionType):
        self.region = weakref.ref(region)

    def save_checkpoint(self, path: str, format: str = "pytorch") -> None:
        """Save region state to checkpoint."""
        pass

    def load_checkpoint(self, path: str) -> None:
        """Load region state from checkpoint."""
        pass
```

**Proposed Change:**
Create base class in `src/thalia/io/checkpoint_manager.py`:

```python
class RegionCheckpointManager(ABC):
    """Base class for region-specific checkpoint management.

    Handles:
    - PyTorch format (state_dict)
    - Binary format (numpy arrays)
    - Elastic tensor format (variable capacity)
    - Neuromorphic format (spike-based ID tracking)
    """

    def __init__(self, region: Any):
        self.region = weakref.ref(region)

    @abstractmethod
    def get_state_components(self) -> Dict[str, Any]:
        """Return dict of region-specific state components."""
        pass

    def save_checkpoint(self, path: str, format: str = "pytorch") -> None:
        """Save checkpoint (standardized implementation)."""
        state = self.get_state_components()
        # Common save logic

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint (standardized implementation)."""
        # Common load logic
```

**Rationale:**
- Eliminates duplicated checkpoint save/load logic
- Ensures consistent checkpoint format across regions
- Makes it easier to add new checkpoint formats (e.g., ONNX, TensorFlow)
- Centralizes backward compatibility handling

**Impact:**
- Files affected: 3 checkpoint manager files, new base class
- Breaking changes: Low (internal refactoring, checkpoint format unchanged)
- Severity: Medium
- Estimated effort: 5-7 hours

---

## Tier 3 - Major Restructuring

### 3.1 Module Reorganization: Split Large Region Files

**Current State:**
Several region files exceed 2000 lines with complex internal structure:

- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py): **3659 lines**
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py): **2416 lines**
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py): **~2600 lines**

These files are already well-organized with:
- Clear section comments and line-number navigation guides
- Extracted subcomponents where appropriate (e.g., StriatumLearningComponent, ForwardPassCoordinator)
- ADR-011 justification for consolidation

**Current Justification (from ADR-011):**
> "The L4→L2/3→L5 cascade is a single biological computation within one timestep.
> Splitting by layer would require passing 15+ intermediate tensors (spikes, membrane, conductances),
> break the canonical microcircuit structure, and obscure the feedforward/feedback balance."

**Proposed Change:**
**Do NOT split** these files at this time. The consolidation is justified and splitting would:
- Break cohesion of single-timestep biological computations
- Create artificial boundaries within integrated processing
- Require extensive parameter passing between split files
- Reduce code navigability (developers would need to jump between files)

**Alternative Improvement:**
Instead of splitting, enhance navigability with better tooling:

1. **Add Folding Regions** (VSCode):
```python
# region: D1/D2 Pathway Initialization
def _init_d1_pathway(self):
    ...
# endregion

# region: Action Selection Logic
def select_action(self):
    ...
# endregion
```

2. **Generate Region Navigation Index** (script):
```bash
python scripts/generate_region_index.py src/thalia/regions/striatum/striatum.py
# Outputs: docs/api/striatum_index.md with clickable method links
```

3. **Add Method Grouping Comments** (already present, ensure consistency):
```python
# =====================================================================
# ACTION SELECTION LOGIC
# =====================================================================
# Lines 651-850: Winner-take-all competition, softmax, greedy epsilon
```

**Rationale:**
- Respects biological coherence (ADR-011 decision remains valid)
- Focuses on discoverability rather than artificial splitting
- Large file size is manageable with proper navigation tools
- Splitting would harm architecture quality, not improve it

**Impact:**
- Files affected: None (no splitting)
- Breaking changes: None
- Severity: N/A (rejected proposal)
- Decision: **MAINTAIN CURRENT STRUCTURE**

---

### 3.2 Introduce Port-Based Routing System Enhancement

**Current State:**
LayeredCortex has L6a and L6b outputs that are intended for port-based routing but not fully integrated:

```python
# In LayeredCortex forward()
output = torch.cat([l23_spikes, l5_spikes])  # Main output
# L6a and L6b spikes are computed but not exposed
```

**Current Usage:**
```python
# Intended usage (not yet implemented):
builder.connect(
    "cortex", "thalamus",
    source_port="l6a",  # L6a → TRN pathway
    target_port="trn"
)
builder.connect(
    "cortex", "thalamus",
    source_port="l6b",  # L6b → relay pathway
    target_port="relay"
)
```

**Proposed Change:**
Implement full port-based routing in NeuralRegion base class:

```python
class NeuralRegion:
    """Base class with port-based routing support."""

    def __init__(self, ...):
        self._port_outputs: Dict[str, torch.Tensor] = {}
        self._port_sizes: Dict[str, int] = {}

    def register_output_port(self, port_name: str, size: int) -> None:
        """Register an output port for routing."""
        self._port_sizes[port_name] = size

    def set_port_output(self, port_name: str, spikes: torch.Tensor) -> None:
        """Store output for a specific port."""
        self._port_outputs[port_name] = spikes

    def get_port_output(self, port_name: Optional[str] = None) -> torch.Tensor:
        """Get output from a specific port, or default output."""
        if port_name is None:
            return self.forward(...)  # Default output
        if port_name not in self._port_outputs:
            raise ValueError(f"Port '{port_name}' not registered")
        return self._port_outputs[port_name]
```

**Usage in LayeredCortex:**
```python
def __init__(self, ...):
    super().__init__(...)
    # Register L6a and L6b as separate output ports
    self.register_output_port("l6a", self.l6a_size)
    self.register_output_port("l6b", self.l6b_size)
    self.register_output_port("default", self.l23_size + self.l5_size)

def forward(self, source_spikes: Dict[str, torch.Tensor]) -> torch.Tensor:
    # ... process layers ...

    # Store port outputs
    self.set_port_output("l6a", l6a_spikes)
    self.set_port_output("l6b", l6b_spikes)
    self.set_port_output("default", torch.cat([l23_spikes, l5_spikes]))

    # Return default output
    return self.get_port_output("default")
```

**BrainBuilder Integration:**
```python
# In BrainBuilder.connect()
def connect(self, source: str, target: str,
            source_port: Optional[str] = None,
            target_port: Optional[str] = None):
    source_region = self.components[source]

    # Get port-specific output
    if hasattr(source_region, 'get_port_output'):
        output = source_region.get_port_output(source_port)
    else:
        output = source_region.forward(...)  # Fallback
```

**Rationale:**
- Enables proper L6a→TRN and L6b→relay routing
- Generalizes to other regions with multiple output pathways
- Cleaner than concatenation + slicing pattern
- Matches biological reality (different cell types project to different targets)

**Impact:**
- Files affected: NeuralRegion base class, LayeredCortex, BrainBuilder, AxonalProjection
- Breaking changes: High (requires updates to connection logic)
- Severity: High
- Estimated effort: 12-16 hours
- **Recommendation**: Design RFC/ADR before implementation

---

### 3.3 Unified Diagnostic Schema Enforcement

**Current State:**
Regions implement `get_diagnostics()` with varying return schemas:

**Striatum:**
```python
def get_diagnostics(self) -> Dict[str, Any]:
    return {
        "d1_weight_mean": ...,
        "d2_weight_mean": ...,
        "exploration_rate": ...,
        "dopamine_level": ...,
    }
```

**LayeredCortex:**
```python
def get_diagnostics(self) -> Dict[str, Any]:
    return {
        "l4_firing_rate": ...,
        "l23_activity": ...,
        "bcm_theta": ...,
    }
```

**Proposed Change:**
Enforce structured diagnostics schema using dataclasses:

```python
# In core/diagnostics_schema.py
@dataclass
class RegionDiagnostics:
    """Standard diagnostic schema for all regions."""

    # Required fields (all regions must provide)
    region_name: str
    firing_rate_hz: float
    sparsity: float
    health_score: float  # 0.0-1.0

    # Optional fields (region-specific)
    weight_stats: Optional[Dict[str, float]] = None
    learning_metrics: Optional[Dict[str, float]] = None
    neuromodulator_levels: Optional[Dict[str, float]] = None
    custom_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dict for logging/monitoring."""
        pass
```

**Enforcement:**
```python
class NeuralRegion:
    @abstractmethod
    def get_diagnostics(self) -> RegionDiagnostics:
        """Return structured diagnostics (enforced by type system)."""
        pass
```

**Migration Path:**
1. Create `RegionDiagnostics` dataclass
2. Add compatibility layer in DiagnosticsMixin:
```python
def dict_to_diagnostics(self, diag_dict: Dict[str, Any]) -> RegionDiagnostics:
    """Convert legacy dict format to RegionDiagnostics."""
    return RegionDiagnostics(
        region_name=self.__class__.__name__,
        firing_rate_hz=diag_dict.get("firing_rate_hz", 0.0),
        sparsity=diag_dict.get("sparsity", 1.0),
        health_score=diag_dict.get("health_score", 1.0),
        custom_metrics=diag_dict,
    )
```
3. Gradually migrate regions to return `RegionDiagnostics`

**Rationale:**
- Ensures consistent diagnostic output across regions
- Enables type-safe diagnostic aggregation and analysis
- Makes monitoring and visualization more reliable
- Supports automatic dashboard generation

**Impact:**
- Files affected: All regions (7+ files), DiagnosticsMixin, monitoring code
- Breaking changes: High (changes return type of get_diagnostics())
- Severity: High
- Estimated effort: 10-14 hours
- **Recommendation**: Gradual migration with backward compatibility layer

---

### 3.4 Learning Strategy Hot-Swapping Infrastructure

**Current State:**
Learning strategies are set at initialization and cannot be changed during training:

```python
class Striatum:
    def __init__(self, config):
        self.learning_strategy = create_strategy("three_factor", ...)
```

**Proposed Change:**
Add infrastructure for runtime strategy switching (e.g., for curriculum learning, critical periods):

```python
class LearningStrategyMixin:
    """Enhanced mixin with hot-swapping support."""

    def set_learning_strategy(self, source_name: str, strategy: LearningStrategy) -> None:
        """Change learning strategy for a source at runtime.

        Useful for:
        - Critical period transitions (STDP → BCM)
        - Curriculum stage changes (exploratory → exploitative)
        - Adaptive learning (performance-based strategy selection)
        """
        if source_name not in self.strategies:
            raise ValueError(f"Source '{source_name}' not registered")

        old_strategy = self.strategies[source_name]
        self.strategies[source_name] = strategy

        # Transfer learned state if compatible
        if hasattr(old_strategy, 'transfer_state'):
            old_strategy.transfer_state(strategy)

        logger.info(f"Switched strategy for '{source_name}': "
                   f"{old_strategy.__class__.__name__} → {strategy.__class__.__name__}")

    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about current strategies."""
        return {
            source: {
                "type": strategy.__class__.__name__,
                "config": strategy.config.__dict__,
                "active": strategy.enabled,
            }
            for source, strategy in self.strategies.items()
        }
```

**Example Usage:**
```python
# Critical period transition
if brain.current_timestep > critical_period_end:
    cortex.set_learning_strategy("thalamus",
        create_strategy("bcm", BCMConfig(...)))

# Curriculum stage change
if performance > threshold:
    striatum.set_learning_strategy("cortex",
        create_strategy("three_factor",
                       ThreeFactorConfig(learning_rate=0.001)))  # Reduce LR
```

**Rationale:**
- Enables dynamic learning rule adaptation
- Supports critical period modeling (developmentally-staged learning)
- Allows performance-based strategy selection
- Facilitates curriculum learning experiments

**Impact:**
- Files affected: LearningStrategyMixin, all regions using strategies
- Breaking changes: Low (additive feature)
- Severity: Medium-High
- Estimated effort: 8-12 hours

---

## Risk/Impact Assessment and Sequencing

### Recommended Implementation Order

**Phase 1 - Quick Wins (1-2 weeks):**
1. Extract magic numbers to constants (Tier 1.1)
2. Standardize size dictionary keys (Tier 1.2)
3. Remove minimal torch.rand occurrences (Tier 1.4)
4. Consolidate neuron type documentation (Tier 1.3)
5. Standardize reset_state() signature (Tier 1.5)

**Phase 2 - Moderate Refactoring (3-4 weeks):**
1. Consolidate eligibility trace management (Tier 2.1)
2. Standardize pathway configuration patterns (Tier 2.2)
3. Extract inter-layer weight initialization (Tier 2.3)
4. Consolidate checkpoint manager pattern (Tier 2.5)

**Phase 3 - Major Features (6-8 weeks):**
1. Create ForwardPassCoordinator base class (Tier 2.4)
2. Learning strategy hot-swapping infrastructure (Tier 3.4)
3. Unified diagnostic schema enforcement (Tier 3.3)
4. Port-based routing system enhancement (Tier 3.2)

**Rejected/Deferred:**
- Module reorganization (Tier 3.1) - **MAINTAIN CURRENT STRUCTURE** per ADR-011

### Risk Mitigation Strategies

1. **Backward Compatibility:**
   - Maintain deprecation warnings for 2 minor versions
   - Provide migration scripts for checkpoint format changes
   - Document breaking changes in CHANGELOG.md

2. **Testing:**
   - Add regression tests before each refactoring
   - Maintain 80%+ coverage throughout changes
   - Run full integration test suite after each tier

3. **Documentation:**
   - Update API docs immediately after changes
   - Add migration guides for breaking changes
   - Update copilot-instructions.md with new patterns

4. **Incremental Rollout:**
   - Implement changes in feature branches
   - Merge incrementally (1-2 recommendations per PR)
   - Allow community feedback between phases

---

## Appendix A: Affected Files

### Core Modules
- [src/thalia/core/neural_region.py](src/thalia/core/neural_region.py) - Base class for all regions
- [src/thalia/core/protocols/component.py](src/thalia/core/protocols/component.py) - Component protocols
- [src/thalia/core/diagnostics_schema.py](src/thalia/core/diagnostics_schema.py) - Diagnostic schemas

### Regions
- [src/thalia/regions/striatum/striatum.py](src/thalia/regions/striatum/striatum.py) - Reinforcement learning region (3659 lines)
- [src/thalia/regions/cortex/layered_cortex.py](src/thalia/regions/cortex/layered_cortex.py) - Multi-layer cortex (2416 lines)
- [src/thalia/regions/hippocampus/trisynaptic.py](src/thalia/regions/hippocampus/trisynaptic.py) - Memory formation
- [src/thalia/regions/prefrontal/prefrontal.py](src/thalia/regions/prefrontal/prefrontal.py) - Working memory
- [src/thalia/regions/cerebellum/cerebellum.py](src/thalia/regions/cerebellum/cerebellum.py) - Motor learning
- [src/thalia/regions/thalamus/thalamus.py](src/thalia/regions/thalamus/thalamus.py) - Sensory relay

### Learning System
- [src/thalia/learning/strategy_registry.py](src/thalia/learning/strategy_registry.py) - Strategy factory
- [src/thalia/learning/strategy_mixin.py](src/thalia/learning/strategy_mixin.py) - Strategy application
- [src/thalia/learning/rules/strategies.py](src/thalia/learning/rules/strategies.py) - Strategy implementations (1209 lines)
- [src/thalia/learning/eligibility/trace_manager.py](src/thalia/learning/eligibility/trace_manager.py) - Eligibility traces

### Components
- [src/thalia/components/neurons/neuron.py](src/thalia/components/neurons/neuron.py) - ConductanceLIF neuron model (620 lines)
- [src/thalia/components/neurons/neuron_factory.py](src/thalia/components/neurons/neuron_factory.py) - Neuron factory registry (498 lines)
- [src/thalia/components/synapses/weight_init.py](src/thalia/components/synapses/weight_init.py) - Weight initialization registry (415 lines)

### Pathways
- [src/thalia/pathways/axonal_projection.py](src/thalia/pathways/axonal_projection.py) - Pure axonal routing (522 lines)
- [src/thalia/pathways/dynamic_pathway_manager.py](src/thalia/pathways/dynamic_pathway_manager.py) - Pathway management

### Mixins
- [src/thalia/mixins/diagnostics_mixin.py](src/thalia/mixins/diagnostics_mixin.py) - Diagnostic utilities (362 lines)
- [src/thalia/mixins/growth_mixin.py](src/thalia/mixins/growth_mixin.py) - Growth utilities (949 lines)
- [src/thalia/mixins/resettable_mixin.py](src/thalia/mixins/resettable_mixin.py) - State reset

### Constants
- [src/thalia/constants/neuromodulation.py](src/thalia/constants/neuromodulation.py) - DA, ACh, NE levels
- [src/thalia/constants/neuron.py](src/thalia/constants/neuron.py) - Neuron parameters
- [src/thalia/constants/learning.py](src/thalia/constants/learning.py) - Learning rates
- [src/thalia/constants/architecture.py](src/thalia/constants/architecture.py) - Architectural constants

### Managers
- [src/thalia/managers/component_registry.py](src/thalia/managers/component_registry.py) - Component registration (760 lines)

---

## Appendix B: Detected Code Patterns

### Pattern 1: Registry Pattern (Excellent)

**Locations:**
- `ComponentRegistry` (regions and pathways)
- `LearningStrategyRegistry` (learning strategies)
- `WeightInitializer` (weight initialization)
- `NeuronFactory` (neuron types)

**Assessment:** ✅ **Well-implemented**
- Consistent decorator-based registration
- Type-safe creation methods
- Good discovery APIs (list_types(), has_type())
- Enables plugin architecture

### Pattern 2: Mixin Composition (Excellent)

**Locations:**
- `DiagnosticsMixin` - Shared diagnostic utilities
- `GrowthMixin` - Weight expansion helpers
- `ResettableMixin` - State reset
- `StateLoadingMixin` - Checkpoint restoration
- `NeuromodulatorMixin` - DA/ACh/NE management
- `LearningStrategyMixin` - Strategy application

**Assessment:** ✅ **Well-implemented**
- Clear separation of concerns
- No god objects (focused, single-responsibility mixins)
- Proper MRO handling
- Eliminates code duplication effectively

### Pattern 3: Strategy Pattern for Learning (Excellent)

**Locations:**
- `HebbianStrategy`, `STDPStrategy`, `BCMStrategy`
- `ThreeFactorStrategy`, `ErrorCorrectiveStrategy`
- `CompositeStrategy` for composition

**Assessment:** ✅ **Well-implemented**
- Uniform `compute_update()` interface
- Pluggable and composable
- Region-specific factories (create_cortex_strategy, create_striatum_strategy)
- Enables easy experimentation

### Pattern 4: Config + Sizes + Device Separation (Good)

**Locations:**
- Striatum: `__init__(config, sizes, device)`
- LayeredCortex: `__init__(config, sizes, device)`
- Recent pattern adoption (January 2026)

**Assessment:** ✅ **Good pattern, needs consistency**
- Separates behavioral config from structural sizes
- Better than monolithic config objects
- Needs documentation in patterns/
- Some older regions still use mixed configs

### Pattern 5: Biological Plausibility (Excellent)

**Assessment:** ✅ **Consistently maintained**
- ✅ All learning is local (no backpropagation)
- ✅ Spike-based processing (binary spikes)
- ✅ No global error signals
- ✅ Proper temporal dynamics (traces, delays)
- ✅ Neuromodulation for gating, not direct weight updates

---

## Appendix C: Antipatterns Detected (Minimal)

### Antipattern 1: Inline Magic Numbers (Minor)

**Severity:** Low
**Occurrences:** ~10-15 locations
**Fix:** Extract to constants/ (Tier 1.1)

### Antipattern 2: Inconsistent Size Dictionary Keys (Minor)

**Severity:** Low
**Occurrences:** Across 7 regions
**Fix:** Standardize naming (Tier 1.2)

### Antipattern 3: SimpleNamespace Config Hack (Minor)

**Severity:** Low
**Occurrences:** AxonalProjection only
**Fix:** Use proper dataclass config (Tier 2.2)

### Non-Antipatterns (Justified Patterns)

**Large File Sizes (Striatum: 3659 lines, LayeredCortex: 2416 lines):**
- **Justified by ADR-011**: Single-timestep biological computations
- **Mitigations in place**: Clear section comments, navigation guides, extracted subcomponents
- **Decision**: Maintain current structure

**Multi-Source Weight Dictionaries:**
- **Pattern**: `synaptic_weights: Dict[str, nn.Parameter]`
- **Justified**: Biological accuracy (synapses at target dendrites, per-source learning)
- **Not an antipattern**: Matches neuroscience architecture

---

## Conclusion

The Thalia codebase demonstrates **mature architectural practices** with strong registry patterns, effective mixin composition, and consistent biological plausibility. The vast majority of code follows documented patterns and separation of concerns.

**Key Strengths:**
- Excellent pluggability via registry pattern
- Effective code reuse via mixins
- Biologically accurate learning rules
- Clear documentation and navigation aids

**Recommended Focus Areas:**
1. **Tier 1 improvements** (1-2 weeks): Extract magic numbers, standardize naming
2. **Tier 2 refactoring** (3-4 weeks): Consolidate trace management, pathway configs, checkpoint managers
3. **Tier 3 features** (6-8 weeks): Port-based routing, diagnostic schema, hot-swapping

**What NOT to do:**
- Do not split large region files (justified by biological coherence per ADR-011)
- Do not introduce backpropagation or non-local learning (violates biological plausibility)
- Do not break mixin pattern (it works well and eliminates duplication)

This architecture is well-suited for continued development and scaling to LLM-level capabilities while maintaining biological plausibility.
