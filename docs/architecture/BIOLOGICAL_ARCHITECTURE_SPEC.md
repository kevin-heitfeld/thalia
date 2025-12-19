# Biologically Accurate Neural Communication Architecture

**Date**: 2025-12-19
**Status**: Design specification for v2.0 refactoring (Phase 2: 67% complete)
**Priority**: Critical - Foundation for entire project

## Part 1: How the Real Brain Works

### Biological Components

#### 1. Neuron Soma (Cell Body)
- **Location**: Where the neuron "lives" in a brain region
- **Function**: Integrates all synaptic inputs, decides whether to spike
- **Properties**:
  - Membrane potential (voltage)
  - Spike threshold
  - Intrinsic excitability
  - Adaptation mechanisms
- **Output**: Binary spike (action potential) when threshold crossed

#### 2. Dendrites (Input Side)
- **Location**: Part of target neuron, extending into target region
- **Function**: Receive synaptic inputs from axons
- **Properties**:
  - Synaptic weights (strength of each synapse)
  - Synaptic plasticity rules (how weights change)
  - Local dendritic computation (e.g., branch-specific nonlinearities)
  - Integration time constants
- **Key Point**: **Dendrites belong to the TARGET neuron**

#### 3. Axons (Output Side)
- **Location**: Extends from source neuron to target region(s)
- **Function**: Transmit spikes from source to target(s)
- **Properties**:
  - Conduction delay (distance-dependent, typically 0.1-10ms per mm)
  - Branches to multiple targets
  - NO weights, NO learning, NO integration
  - Can have myelin (faster conduction)
- **Key Point**: **Axons are pure communication channels**

#### 4. Synapses (Connection Points)
- **Location**: Where axon terminal meets dendrite
- **Function**: Convert presynaptic spike into postsynaptic current/conductance
- **Properties**:
  - **Synaptic weight** (efficacy): How strongly pre-spike affects post-neuron
  - **Plasticity rule**: How weight changes based on pre/post activity
  - **Short-term plasticity**: Facilitation/depression over ~100ms
  - **Neurotransmitter type**: Excitatory (glutamate) or inhibitory (GABA)
- **Key Point**: **Synapse weight is stored at TARGET dendrite, not in axon**

### Information Flow in Real Brain

```
SOURCE REGION                    TARGET REGION
┌─────────────┐                 ┌─────────────────────────┐
│  Neuron A   │                 │      Neuron B           │
│             │                 │                         │
│  Soma       │                 │  Dendrites              │
│  [integrate]│                 │  ├─ Synapse 1 (w=0.8)   │
│  [spike?]   │                 │  ├─ Synapse 2 (w=0.5)   │
│      │      │                 │  └─ Synapse 3 (w=1.2)   │
│      ▼      │                 │         │               │
│   Axon      │────[routing]────│         ▼               │
│   [0 weight]│   pure spikes   │     [integrate]         │
│   [delay]   │   delayed only  │         │               │
└─────────────┘                 │         ▼               │
                                │       Soma              │
                                │     [spike?]            │
                                └─────────────────────────┘
```

**Critical insights**:
1. **Axon carries identical spikes to ALL targets** (no per-target modification)
2. **Each target has its OWN weights** for the same source
3. **Learning happens at target synapses**, not in axon
4. **Delay is in axon**, weight is at synapse

### Example: Thalamocortical Connection

```
THALAMUS                        CORTEX L4
100 neurons                     500 neurons
│                               │
│ Each neuron has axon          │ Each neuron has dendrites
│ projecting to cortex          │ receiving from thalamus
│                               │
└──[axonal bundle]──────────────┤
   - 100 fibers                 │
   - 5ms delay                  │ Each L4 neuron receives:
   - NO weights                 │ - Synapses from ~20 thalamic neurons
   - Just spike routing         │ - Each synapse has weight (0.1-2.0)
                                │ - Total: 500 neurons × 20 synapses = 10k synapses
                                │ - Weight matrix [500, 100] with 20% sparsity
                                │ - Learning rule: STDP for these synapses
```

**Key Point**: If thalamus also projects to L6, it uses the **same axons** (delay), but L6 neurons have **different weights** for the same thalamic axons.

### Multi-Source Integration

Real example: Striatum receives from cortex, hippocampus, and thalamus.

```
CORTEX (1000 neurons)           STRIATUM (500 MSNs)
    │ axons (2ms delay)         │
    └────────────────────────┐  │ Each MSN has:
                             │  │ - Cortical synapses [500, 1000] weights
HIPPOCAMPUS (200 neurons)    │  │ - Hippocampal synapses [500, 200] weights
    │ axons (3ms delay)      ├──┤ - Thalamic synapses [500, 100] weights
    └────────────────────────┤  │
                             │  │ Integration:
THALAMUS (100 neurons)       │  │ 1. Receive 3 spike vectors (delayed)
    │ axons (2ms delay)      │  │ 2. Apply 3 weight matrices separately
    └────────────────────────┘  │ 3. Sum all currents
                                │ 4. Integrate in soma
                                │ 5. Spike if threshold crossed
```

**Key Point**:
- 3 separate axonal bundles with different delays
- 3 separate weight matrices stored in striatum
- Integration happens at striatum neurons

## Part 2: How We SHOULD Model It

### Component Hierarchy

```
Brain
├── Regions (e.g., Cortex, Thalamus, Striatum)
│   ├── Neurons (soma + dendrites + local axons)
│   │   ├── Intrinsic properties (threshold, tau, etc.)
│   │   └── Synaptic weights (input-specific)
│   └── Learning rules (region-specific)
│
└── Connections (long-range axonal projections)
    ├── Source region + port
    ├── Target region + port
    ├── Axonal delay
    └── NO weights, NO learning
```

### Proposed Architecture

#### 1. NeuralRegion (Neurons + Dendrites)

```python
class NeuralRegion(nn.Module):
    """A population of neurons with their dendritic inputs.

    Biologically accurate:
    - Neurons live here (soma + dendrites)
    - Synaptic weights live here (on dendrites)
    - Learning rules live here (region-specific)
    - Integration happens here
    """

    def __init__(
        self,
        n_neurons: int,
        neuron_config: NeuronConfig,
        learning_rule: Optional[str] = None,
    ):
        self.neurons = ConductanceLIF(n_neurons, neuron_config)
        self.n_neurons = n_neurons

        # Synaptic inputs: one weight matrix per source
        self.synaptic_weights: Dict[str, nn.Parameter] = {}
        self.plasticity_rules: Dict[str, LearningStrategy] = {}

        # Region-specific learning rule for NEW connections
        self.default_learning_rule = learning_rule  # "stdp", "bcm", "three_factor", etc.

    def add_input_source(
        self,
        source_name: str,
        n_input: int,
        learning_rule: Optional[str] = None,
    ) -> None:
        """Add synaptic weights for a new input source.

        Args:
            source_name: Name of source region (e.g., "thalamus", "hippocampus")
            n_input: Number of input neurons
            learning_rule: Override default learning rule for this input
        """
        # Create weight matrix for this source
        weights = WeightInitializer.sparse_random(
            n_output=self.n_neurons,
            n_input=n_input,
            sparsity=0.2,  # Biologically realistic
        )
        self.synaptic_weights[source_name] = nn.Parameter(weights)

        # Create plasticity rule for this source
        rule = learning_rule or self.default_learning_rule
        if rule:
            self.plasticity_rules[source_name] = create_strategy(rule)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Process inputs through synapses and neurons.

        Args:
            inputs: Dict mapping source names to spike vectors
                   e.g., {"thalamus": [128], "hippocampus": [200]}

        Returns:
            Output spikes [n_neurons]
        """
        # Compute synaptic currents for each source
        g_exc_total = torch.zeros(self.n_neurons, device=self.device)

        for source_name, input_spikes in inputs.items():
            if source_name not in self.synaptic_weights:
                raise ValueError(f"No synaptic weights for source '{source_name}'")

            weights = self.synaptic_weights[source_name]
            g_exc = torch.matmul(weights, input_spikes.float())
            g_exc_total += g_exc

            # Apply plasticity if learning enabled
            if source_name in self.plasticity_rules:
                plasticity = self.plasticity_rules[source_name]
                new_weights, metrics = plasticity.compute_update(
                    weights=weights,
                    pre_spikes=input_spikes,
                    post_spikes=self.state.output_spikes,
                )
                self.synaptic_weights[source_name].data = new_weights

        # Integrate in neurons (could add inhibition here)
        output_spikes, voltage = self.neurons(g_exc_total, g_inh=0)

        # Store state
        self.state.output_spikes = output_spikes

        return output_spikes
```

#### 2. AxonalProjection (Pure Routing)

```python
class AxonalProjection(nn.Module):
    """Long-range axonal fibers with conduction delays.

    Biologically accurate:
    - NO weights (weights are at target dendrites)
    - NO learning (learning happens at target synapses)
    - ONLY delays and routing
    - Can branch to multiple targets
    """

    def __init__(
        self,
        source: Tuple[str, Optional[str]],  # (region_name, port)
        delay_ms: float,
    ):
        self.source_region = source[0]
        self.source_port = source[1]
        self.delay_ms = delay_ms

        # Circular buffer for delay
        self.delay_buffer: Optional[torch.Tensor] = None
        self.delay_steps = int(delay_ms / dt_ms)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Route spikes with axonal delay.

        Args:
            spikes: Binary spike vector [n_source]

        Returns:
            Delayed spikes [n_source] (SAME SIZE, just delayed)
        """
        if self.delay_steps == 0:
            return spikes

        # Apply delay using circular buffer
        delayed_spikes = self._apply_delay(spikes)
        return delayed_spikes

    # NO forward_learn, NO weight matrices, NO plasticity
```

#### 3. Brain (Connection Manager)

```python
class Brain(nn.Module):
    """Manages regions and their axonal connections."""

    def __init__(self):
        self.regions: Dict[str, NeuralRegion] = {}
        self.axons: Dict[Tuple[str, str], AxonalProjection] = {}

    def add_region(
        self,
        name: str,
        n_neurons: int,
        learning_rule: str,
    ) -> NeuralRegion:
        """Add a brain region with specified learning rule."""
        region = NeuralRegion(n_neurons, learning_rule=learning_rule)
        self.regions[name] = region
        return region

    def connect(
        self,
        source: str,
        target: str,
        delay_ms: float = 2.0,
        learning_rule: Optional[str] = None,
    ) -> None:
        """Create axonal connection from source to target.

        This does TWO things (biologically accurate):
        1. Creates axonal projection (routing + delay)
        2. Adds synaptic weights to target region
        """
        source_region = self.regions[source]
        target_region = self.regions[target]

        # 1. Create axonal projection (pure routing)
        axon = AxonalProjection(
            source=(source, None),
            delay_ms=delay_ms,
        )
        self.axons[(source, target)] = axon

        # 2. Add synaptic weights to target region
        target_region.add_input_source(
            source_name=source,
            n_input=source_region.n_neurons,
            learning_rule=learning_rule,  # Can override target's default
        )

    def forward(self, sensory_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute brain for one timestep using event-driven processing."""
        # Event-driven execution (existing EventScheduler)
        # Key change: Axons route spikes, targets apply weights
        ...
```

### Key Design Principles

1. **Separation of Concerns**:
   - **Axons**: Routing + delays (communication)
   - **Dendrites**: Weights + learning (computation)
   - **Soma**: Integration + spiking (decision)

2. **Weight Ownership**:
   - Weights stored in **target region** (`region.synaptic_weights[source_name]`)
   - One weight matrix per source
   - Each region can have different weights for same source

3. **Learning Locality**:
   - Plasticity rules stored in **target region** (`region.plasticity_rules[source_name]`)
   - Can override per-source: "thalamus → cortex uses STDP, hippocampus → cortex uses BCM"
   - Region has default rule for new connections

4. **Port-Based Routing**:
   - Regions can have multiple outputs: `cortex.get_output(port="l23")` or `cortex.get_output(port="l5")`
   - Axons specify which port: `AxonalProjection(source=("cortex", "l5"), delay=2.0)`
   - Targets receive: `cortex.forward({"thalamus": spikes, "hippocampus": spikes})`

## Part 3: Migration Plan

### Phase 1: Add New Architecture (Parallel Implementation)

**Goal**: Build new system alongside old, no breaking changes yet.

**Steps**:
1. Create `NeuralRegion` base class with synaptic weight dict
2. Create minimal `AxonalProjection` (delay only, no weights)
3. Update `Brain.connect()` to create both axon + target synapses
4. Add feature flag: `use_legacy_pathways=True` (default)

**Validation**:
- New regions work in isolation
- Can create test brain with new architecture
- Old code still runs unchanged

**Time**: 2-3 days

### Phase 2: Migrate Core Regions

**Status**: In Progress (Striatum ✅, PFC ✅, Hippocampus ✅, LayeredCortex ✅, 2 regions remaining)
**Goal**: Convert all 6 regions to NeuralRegion architecture.

**Completed**:
1. ✅ **Striatum** (2025-12-19):
   - Changed inheritance from BrainComponentBase to NeuralRegion
   - Moved D1/D2 weights to parent's `synaptic_weights` dict
   - Implemented property-based weight access (@property/@setter)
   - D1 and D2 are separate full populations (both [n_output, n_input])
   - All 13 tests passing (delays, checkpoints, learning)
   - Commits: 8cb285c, 0aa9e30, 0fb9494, 0ce412d, 9bd1135
   - Files: `striatum.py`, `pathway_base.py`
   - Key insight: Opponent pathways need separate full-size matrices for opposite learning rules

2. ✅ **Prefrontal (PFC)** (2025-12-19):
   - Changed inheritance from NeuralComponent to NeuralRegion
   - Moved feedforward weights to `synaptic_weights["default"]`
   - Single input source (simpler than multi-source Striatum)
   - Internal weights (recurrent, inhibitory) remain as nn.Parameter
   - All 9 checkpoint tests passing
   - Commit: 6bbe11f
   - Files: `prefrontal.py`, `prefrontal_checkpoint_manager.py`
   - Key insight: Single-source regions straightforward, checkpoint managers need updates

3. ✅ **Hippocampus** (2025-12-19):
   - Changed inheritance from NeuralComponent to NeuralRegion
   - Moved 4 EC pathway weights to `synaptic_weights` dict (EC_spatial, EC_nonspatial, EC_temporal, EC_object)
   - Internal weights (DG→CA3, CA3→CA1, CA3 recurrent) remain as nn.Parameter
   - Added backward compatibility for Dict/Tensor inputs
   - All 8 checkpoint tests passing
   - Commit: cd87ff8
   - Files: `trisynaptic.py`
   - Key insight: Multi-source with internal cascade works cleanly - only external inputs move

4. ✅ **LayeredCortex** (2025-12-19):
   - Changed inheritance from NeuralComponent to NeuralRegion
   - Moved w_input_l4 to `synaptic_weights["input"]` (only external weight)
   - Internal cascade (w_l4_l23, w_l23_recurrent, w_l23_l5, w_l23_l6, w_l23_inhib) unchanged
   - Added backward compatibility helpers (_reset_subsystems, _reset_scalars, get_effective_learning_rate, _apply_axonal_delay)
   - Updated forward() to accept Union[Dict, Tensor]
   - 15/17 tests passing (88%), 2 failures are pre-existing behavioral issues
   - Commit: a418d9a
   - Files: `layered_cortex.py`
   - Key insight: Largest region (1789 lines) but simplest migration - only ONE external weight

**Next Steps**:
1. Convert `Thalamus` (relay nucleus TRN + mode switching)
2. Convert `Cerebellum` (granule layer, Purkinje cells, DCN)

**Remaining Regions** (2/6):
- **Thalamus**: TRN gating, relay mode switching, spatial filtering
- **Cerebellum**: Granule→Purkinje→DCN cascade, motor learning

**Time**: 2 days estimated (2 regions × 1 day each, based on current pace)

---

**Historical Migration Notes**:

3. ✅ Convert `Striatum` (COMPLETE):
   - Successfully moved weights from D1/D2 pathway objects to parent's synaptic_weights
   - Used `synaptic_weights["default_d1"]` and `synaptic_weights["default_d2"]`
   - Each pathway: full [n_output, n_input] matrix (separate populations, not split)
   - Property-based access: @property weights returns parent's tensor
   - All operations (matmul, learning, checkpoints) work transparently
   - Biological accuracy maintained: D1 (DA+→LTP) and D2 (DA+→LTD) need separate matrices

4. Update tests to use new architecture

**Validation**:
- All regions work with new `forward(inputs: Dict)` signature
- Weights successfully moved to targets
- Learning still works

**Time**: 1 week

### Phase 3: Update Event System

**Goal**: Make event system work with new architecture.

**Steps**:
1. Remove pathway weight forwarding from event system
2. Axons only route spikes (already mostly done)
3. Regions apply their own synaptic weights
4. Multi-source buffering works automatically (targets expect Dict)

**Changes**:
```python
def _schedule_downstream_events(self, source, output_spikes, time):
    # Find all axons from this source
    for (src, tgt), axon in self.axons.items():
        if src == source:
            # Route through axon (delay only)
            delayed_spikes = axon.forward(output_spikes)

            # Schedule delivery to target region
            # Target will apply its own synaptic weights
            event = Event(
                time=time + axon.delay_ms,
                target=tgt,
                payload=SpikePayload(
                    source=src,  # Target needs to know source
                    spikes=delayed_spikes
                ),
            )
            self.scheduler.schedule(event)

def _process_region_event(self, region_name, events):
    region = self.regions[region_name]

    # Collect all inputs from different sources
    inputs = {}
    for event in events:
        inputs[event.payload.source] = event.payload.spikes

    # Region applies synaptic weights and processes
    output_spikes = region.forward(inputs)

    # Schedule to downstream regions
    self._schedule_downstream_events(region_name, output_spikes, time)
```

**Validation**:
- Event system properly buffers multi-source inputs
- Axons don't try to apply weights
- Regions receive Dict and apply their own weights
- Timing still correct

**Time**: 3-4 days

### Phase 4: Remove Legacy Code

**Goal**: Delete old pathway weight system entirely.

**Steps**:
1. Remove `SpikingPathway` class (weights, learning, STP)
2. Remove `MultiSourcePathway` class (replaced by regions with Dict input)
3. Remove feature flag `use_legacy_pathways`
4. Update all documentation
5. Clean up imports

**Validation**:
- All tests pass with new architecture only
- No legacy code paths remain
- Documentation reflects new model

**Time**: 2 days

### Phase 5: Optimize & Polish

**Goal**: Make new architecture fast and ergonomic.

**Steps**:
1. Benchmark: Compare performance to legacy system
2. Optimize: Fuse operations, reduce dict overhead
3. Add helpers: `brain.add_connection_with_learning(src, tgt, rule="stdp")`
4. Update examples and tutorials
5. No migration guide needed (no external users yet)

**Time**: 1 week

## Part 4: Benefits of New Architecture

### Biological Accuracy ✓
- Weights at target dendrites (correct)
- Axons are pure routing (correct)
- Learning happens locally at synapses (correct)
- Multi-source integration at target (correct)

### Conceptual Clarity ✓
- One pathway type: `AxonalProjection` (simple)
- Weights in regions, not pathways
- Learning rules are region properties
- Multi-source "just works" (targets expect Dict)

### Implementation Simplicity ✓
- No more `isinstance(MultiSourcePathway)` checks
- No more "should I use SpikingPathway or AxonalProjection?"
- Event system simpler (just route, don't compute)
- Regions own their data

### Flexibility ✓
- Easy to add region-specific learning rules
- Easy to have different weights for same source
- Easy to add new connection types
- Easy to implement advanced dendritic computation

### Performance ✓
- Fewer objects (one axon vs many pathway instances)
- Better memory locality (weights grouped by region)
- Easier to parallelize (regions are independent)
- Could use sparse tensors for weights

## Part 5: Timeline

**Total**: ~4-5 weeks for complete migration (revised)

- **Week 1**: Phase 1 ✅ + Phase 2 start ✅ (Striatum complete in 1 day)
- **Week 2-3**: Phase 2 continue ✅ (PFC, Hippocampus complete in 1 day each)
- **Week 3-4**: Phase 2 final (Cortex, Thalamus, Cerebellum)
- **Week 5**: Phase 3 (event system update)
- **Week 6**: Phase 4 + Phase 5 (cleanup + optimization)

**Progress**: 4/6 regions migrated (67%)
**Elapsed**: 4 days total (Striatum, PFC, Hippocampus, LayeredCortex each completed in 1 day)
**Last Updated**: 2025-12-19

**Can parallelize**:
- Region conversions (multiple people)
- Test updates (while regions being converted)
- Documentation (while code being written)

**Risk mitigation**:
- Keep legacy code until Phase 4
- Feature flag allows switching between implementations
- Extensive tests at each phase
- Can rollback any phase if problems found

## Part 6: Alternative Considered

### Why not keep SpikingPathway?

Some might argue: "Pathways with weights work fine, why change?"

**Counterarguments**:
1. **Biological inaccuracy**: Real axons don't have weights
2. **Conceptual confusion**: "Where are synapses?" becomes unclear
3. **Duplication**: Same source → multiple targets needs multiple weight copies
4. **Inflexibility**: Hard to have target-specific plasticity for same source
5. **Growth complexity**: When cortex grows, do pathways grow? Or cortex internal weights?

**Example problem with current architecture**:
```python
# Current: Thalamus → Cortex L4 pathway has weights
thal_to_cortex = SpikingPathway(n_input=128, n_output=500, learning_rule="stdp")

# Problem: Thalamus also projects to Striatum
thal_to_striatum = SpikingPathway(n_input=128, n_output=150, learning_rule="three_factor")

# Now thalamus has TWO weight matrices for its outputs!
# But biologically, it has ONE set of axons that branch.
# Weights are different because TARGETS have different synapses, not because SOURCE has different axons.
```

**New architecture solves this**:
```python
# New: One axonal projection
thalamus_axons = AxonalProjection(source="thalamus", delay=2.0)

# Cortex has its own thalamic weights
cortex.synaptic_weights["thalamus"] = [500, 128] with STDP

# Striatum has DIFFERENT thalamic weights
striatum.synaptic_weights["thalamus"] = [150, 128] with three_factor

# Biologically accurate: Same axons, different synapses at targets
```

## Decision

**Recommendation**: Implement this refactoring immediately.

**Rationale**:
1. Project not released yet - perfect time for breaking changes
2. Foundation for everything else - better to get it right now
3. Makes future development easier (growth, multi-region learning, etc.)
4. Aligns with stated goals ("biologically accurate", "local learning")
5. Timeline is reasonable (~4 weeks)

**What's needed from you**:
1. Approval to proceed with migration
2. Priority: Should this block other features?
3. Any concerns or modifications to plan?

---

**Next Steps**:
- [ ] Get approval for refactoring
- [ ] Create feature branch: `refactor/biological-synapses`
- [ ] Begin Phase 1: Parallel implementation
- [ ] Set up daily progress tracking
- [ ] Coordinate if multiple people working on this
