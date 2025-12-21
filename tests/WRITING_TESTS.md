# Writing Tests for Thalia

**Date:** December 21, 2025
**Version:** 1.1 (Updated after Test Quality Audit)

This guide documents test quality patterns for the Thalia spiking neural network framework. Tests should validate biological plausibility and learning correctness, not implementation details.

**Recent Updates:**
- Added parameterization best practices (December 21, 2025)
- Added test consolidation guidelines (December 21, 2025)
- Clarified hardcoded assertion anti-patterns (December 21, 2025)
- Added examples of good parameterization from codebase (December 21, 2025)

## Table of Contents
1. [Core Principles](#core-principles)
2. [Good vs Bad Test Patterns](#good-vs-bad-test-patterns)
3. [Parameterization Best Practices](#parameterization-best-practices) ⭐ NEW
4. [Test Consolidation Guidelines](#test-consolidation-guidelines) ⭐ NEW
5. [Edge Case Testing](#edge-case-testing)
6. [Network Integrity Validation](#network-integrity-validation)
7. [Test Organization](#test-organization)
8. [Common Pitfalls](#common-pitfalls)

---

## Core Principles

### 1. Test the Contract, Not the Implementation

**Contract:** What the component promises to do (behavior, outputs, side effects)
**Implementation:** How it achieves that (internal variables, algorithms, data structures)

✅ **DO:** Test observable behavior
```python
def test_striatum_produces_action_votes():
    """Test striatum produces valid action votes."""
    striatum = Striatum(config)
    input_spikes = torch.rand(100) > 0.8

    _ = striatum(input_spikes)
    d1_votes, d2_votes = striatum.get_accumulated_votes()  # PUBLIC API

    # Test CONTRACT: votes are valid tensors with correct shape
    assert d1_votes.shape == (striatum.n_actions,)
    assert d2_votes.shape == (striatum.n_actions,)
    assert not torch.isnan(d1_votes).any()
    assert not torch.isnan(d2_votes).any()
```

❌ **DON'T:** Test internal implementation
```python
def test_striatum_delay_configuration():
    """Test delay configuration (BRITTLE)."""
    striatum = Striatum(config)

    # ❌ Accessing private attributes
    d1_delay_steps = striatum._d1_delay_steps
    d2_delay_steps = striatum._d2_delay_steps

    assert d1_delay_steps == 15  # Breaks when implementation changes
```

---

### 2. Test Biologically Plausible Behavior

Focus on neural properties that matter for biological accuracy:
- Spike timing and temporal dynamics
- Neuromodulator effects
- Homeostatic regulation
- Pathway delays and competition
- Learning rule correctness (Hebbian, STDP, BCM, etc.)

✅ **DO:** Test biological properties
```python
def test_d1_pathway_arrives_before_d2():
    """Test D1 direct pathway is faster than D2 indirect pathway."""
    config = StriatumConfig(
        d1_to_output_delay_ms=15.0,  # Direct pathway
        d2_to_output_delay_ms=25.0,  # Indirect pathway (slower)
        dt_ms=1.0,
    )
    striatum = Striatum(config)

    # Strong consistent input
    input_spikes = torch.ones(100, dtype=torch.bool)

    # Track when votes first appear
    d1_first_time = None
    d2_first_time = None

    for t in range(50):
        _ = striatum(input_spikes)
        d1_votes, d2_votes = striatum.get_accumulated_votes()

        if d1_first_time is None and d1_votes.sum() > 0:
            d1_first_time = t
        if d2_first_time is None and d2_votes.sum() > 0:
            d2_first_time = t

    # Biologically important: D1 arrives first (temporal competition)
    assert d1_first_time < d2_first_time
```

---

### 3. Use Explicit Values, Not Defaults

Don't test default configuration values - they will change. Test with explicit values.

❌ **DON'T:** Assert default values
```python
def test_config_defaults():
    config = StreamConfig()
    assert config.eval_frequency == 1000  # ❌ Hardcoded default
    assert config.replay_buffer_size == 10000  # ❌ Will break when tuned
```

✅ **DO:** Test config contract
```python
def test_config_has_required_fields():
    """Test config provides required fields."""
    config = StreamConfig()

    # Test that fields exist and have valid types/ranges
    assert hasattr(config, 'eval_frequency')
    assert config.eval_frequency > 0
    assert isinstance(config.enable_replay, bool)

def test_config_accepts_custom_values():
    """Test config accepts custom values."""
    # Use EXPLICIT values (not defaults)
    config = StreamConfig(
        eval_frequency=500,
        replay_buffer_size=5000,
    )

    # OK to assert values we explicitly set
    assert config.eval_frequency == 500
    assert config.replay_buffer_size == 5000
```

---

### 4. Validate State Transitions, Not Just Final State

Test how the system changes over time, not just the end result.

✅ **DO:** Test state transitions
```python
def test_learning_updates_weights():
    """Test that learning strategy actually modifies weights."""
    strategy = HebbianStrategy(HebbianConfig(learning_rate=0.1))

    weights_initial = torch.ones(10, 20) * 0.5
    pre = torch.ones(20)
    post = torch.ones(10)

    # Apply learning
    weights_after, metrics = strategy.compute_update(
        weights_initial.clone(), pre, post
    )

    # Validate STATE TRANSITION: weights changed
    assert not torch.allclose(weights_after, weights_initial)

    # Validate DIRECTION: positive pre/post should strengthen
    assert weights_after.mean() > weights_initial.mean()

    # Validate METRICS: learning occurred
    assert metrics['mean_change'] > 0
    assert metrics['ltp'] > 0
```

---

### 5. Test Dimension Compatibility, Not Exact Values

Validate that components are compatible, not that they match hardcoded expectations.

❌ **DON'T:** Test hardcoded dimensions
```python
def test_cortex_receives_thalamus_input():
    brain = create_test_brain()
    cortex = brain.components["cortex"]

    # ❌ Hardcoded value - breaks when thalamus size changes
    assert cortex.config.n_input == 64
```

✅ **DO:** Test dimensional compatibility
```python
def test_cortex_receives_thalamus_input():
    """Test cortex input size matches thalamus output size."""
    brain = create_test_brain()
    cortex = brain.components["cortex"]
    thalamus = brain.components["thalamus"]

    # ✅ Tests the CONTRACT: dimensions are compatible
    assert cortex.config.n_input == thalamus.n_output

    # ✅ Tests INVARIANT: positive dimension
    assert cortex.config.n_input > 0
```

**Why this matters:** Configuration defaults will change during tuning, growth experiments, and optimization. Tests should survive these changes as long as dimensional compatibility is maintained.

---

## Parameterization Best Practices

### When to Use `@pytest.mark.parametrize`

Parameterize tests when:
1. **Same validation logic, different input values**
2. **Testing across ranges** (sizes, rates, delays)
3. **Boundary value testing** (0, min, max, beyond max)
4. **Multiple valid configurations** (different modes, strategies)

**DO NOT parameterize when:**
- Validation logic differs significantly
- Each case tests a different contract
- Readability suffers from over-parameterization

### Example 1: Size Variations (GOOD ✅)

From `tests/unit/components/neurons/test_neuron_growth.py`:

```python
@pytest.mark.parametrize("initial_n,growth_amount", [
    (50, 10),
    (100, 20),
    (200, 50),
    (500, 100),
])
def test_grow_neurons_various_sizes(self, initial_n, growth_amount):
    """Test growth works with various population sizes."""
    config = ConductanceLIFConfig()
    neurons = ConductanceLIF(n_neurons=initial_n, config=config)
    neurons.reset_state()

    neurons.grow_neurons(growth_amount)

    expected_n = initial_n + growth_amount
    assert neurons.n_neurons == expected_n
    assert neurons.v_threshold.shape[0] == expected_n
    assert neurons.membrane.shape[0] == expected_n
```

**Why this is good:**
- Same contract tested across multiple scales
- Validates growth works for small and large populations
- Clear test ID for each case
- Single test function is maintainable

### Example 2: Delay Durations (GOOD ✅)

From `tests/unit/test_phase1_v2_architecture.py`:

```python
@pytest.mark.parametrize("delay_ms,expected_steps", [
    (1.0, 2),  # 1ms delay = 2 steps to see spikes
    (2.0, 3),  # 2ms delay = 3 steps
    (5.0, 6),  # 5ms delay = 6 steps
])
def test_axonal_delays_various_durations(self, delay_ms, expected_steps):
    """Test axonal delays with various durations."""
    projection = AxonalProjection(
        sources=[("cortex", None, 5, delay_ms)],
        device="cpu",
        dt_ms=1.0,
    )

    # First timestep: input spikes
    spikes_t0 = torch.ones(5, dtype=torch.bool)
    output_t0 = projection.forward({"cortex": spikes_t0})
    assert not output_t0["cortex"].any()

    # Advance until expected_steps - 1
    for step in range(1, expected_steps - 1):
        output = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)})
        assert not output["cortex"].any(), f"Spikes appeared too early at step {step}"

    # At expected_steps, delayed spikes should appear
    final_output = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)})
    assert final_output["cortex"].all()
```

**Why this is good:**
- Tests biological property (axonal delay) across timescales
- Clear expected behavior for each delay
- Comments explain the biological/temporal logic

### Example 3: Learning Rates (GOOD ✅)

From `tests/integration/test_learning_strategy_pattern.py`:

```python
@pytest.mark.parametrize("learning_rate", [0.001, 0.01, 0.1])
def test_learning_rate_affects_weight_change(learning_rate):
    """Test that learning rate scales weight changes."""
    strategy = HebbianStrategy(HebbianConfig(learning_rate=learning_rate))

    weights = torch.ones(10, 20) * 0.5
    pre = torch.ones(20)
    post = torch.ones(10)

    new_weights, metrics = strategy.compute_update(weights.clone(), pre, post)

    # Higher learning rate → larger changes
    change_magnitude = (new_weights - weights).abs().mean()
    assert change_magnitude > 0
    # ... additional validation
```

### Example 4: Extreme Values (GOOD ✅)

From `tests/unit/learning/test_learning_strategy_stress.py`:

```python
@pytest.mark.parametrize("acetylcholine,description", [
    (-0.5, "negative_ach"),
    (0.0, "no_ach"),
    (1.0, "max_ach"),
    (2.0, "excessive_ach"),
])
def test_hippocampus_extreme_acetylcholine(hippocampus, acetylcholine, description):
    """Test hippocampus stability with extreme acetylcholine values."""
    hippocampus.set_neuromodulators(acetylcholine=acetylcholine)

    for _ in range(100):
        input_spikes = torch.rand(hippocampus.config.n_input) > 0.8
        output = hippocampus(input_spikes)

        assert not torch.isnan(output.float()).any(), \
            f"NaN with ACh={acetylcholine} ({description})"
        assert not torch.isinf(output.float()).any(), \
            f"Inf with ACh={acetylcholine} ({description})"
```

**Why this is good:**
- Tests edge cases (negative, zero, normal, excessive)
- Descriptive parameter names
- Clear error messages with parameter values

### Anti-Pattern: Over-Parameterization (AVOID ❌)

```python
# ❌ BAD: Different validation logic for each mode
@pytest.mark.parametrize("mode,config,validation_func", [
    (SelectionMode.GREEDY, {...}, validate_greedy),
    (SelectionMode.SOFTMAX, {...}, validate_softmax),
    (SelectionMode.EPSILON_GREEDY, {...}, validate_epsilon),
])
def test_action_selection_modes(mode, config, validation_func):
    """Test different selection modes."""
    selector = ActionSelector(n_actions=4, config=config)
    # ...
    validation_func(selector, action, info)
```

**Why this is bad:**
- Each mode has completely different validation logic
- Hard to understand what each test case does
- Harder to debug failures (which validation_func failed?)

**Better approach:** Separate tests for each mode
```python
# ✅ GOOD: Clear, separate tests
def test_greedy_selection_deterministic():
    """Test greedy always selects max votes."""
    # ... specific greedy validation

def test_softmax_selection_probabilistic():
    """Test softmax uses probability distribution."""
    # ... specific softmax validation

def test_epsilon_greedy_explores():
    """Test epsilon-greedy balances exploit/explore."""
    # ... specific epsilon-greedy validation
```

---

## Test Consolidation Guidelines

### When to Consolidate Tests

Consolidate when tests have:
1. **Identical structure** (same setup, action, assertions)
2. **Only parameter values differ** (sizes, rates, counts)
3. **Same behavioral contract** being validated

### When to Keep Tests Separate

Keep separate when:
1. **Different validation logic** (checking different properties)
2. **Different biological contracts** (different neural behaviors)
3. **Component-specific behavior** (striatum dopamine vs hippocampus ACh)
4. **Readability suffers** from combining

### Example: Component-Specific Edge Cases (KEEP SEPARATE ✅)

These tests all check "silent input" but validate different component behaviors:

```python
# tests/unit/test_thalamus.py
def test_thalamus_silent_input(thalamus, device):
    """Test thalamus handles zero input with TRN interactions."""
    silent = torch.zeros(thalamus.config.n_input, dtype=torch.bool, device=device)

    for _ in range(10):
        output = thalamus(silent)

        # Thalamus-specific: TRN should remain inactive
        assert thalamus.state.trn_spikes.sum() == 0
        # ... TRN-specific validation

# tests/unit/test_striatum_d1d2_delays.py
def test_striatum_silent_input():
    """Test striatum handles zero input with D1/D2 pathways."""
    silent = torch.zeros(striatum.config.n_input, dtype=torch.bool)

    for _ in range(20):
        _ = striatum(silent)
        d1_votes, d2_votes = striatum.get_accumulated_votes()

        # Striatum-specific: No votes accumulate with no input
        assert d1_votes.sum() == 0
        assert d2_votes.sum() == 0
        # ... delay buffer validation
```

**Why keep separate:**
- Each component has unique behavior with silent input
- Thalamus tests TRN inhibition
- Striatum tests D1/D2 delay buffers
- Cannot be parameterized without losing component-specific validation

### Example: Universal Contracts (CONSOLIDATE ✅)

From `tests/unit/test_component_contracts.py`:

```python
@pytest.mark.parametrize(
    "component_factory,input_size",
    [
        (create_thalamus, 100),
        # Could add more components here
    ],
    ids=["thalamus"],
)
def test_component_reset_contract(component_factory, input_size):
    """Test that components properly reset state (universal contract)."""
    component = component_factory()

    # Run forward to dirty state
    test_input = create_spike_input(input_size)
    _ = component(test_input)

    # Reset
    component.reset_state()

    # Validate clean state
    if hasattr(component, "neurons") and component.neurons is not None:
        if hasattr(component.neurons, "membrane"):
            membrane = component.neurons.membrane
            assert not torch.isnan(membrane).any()
            assert torch.abs(membrane).max() < 0.5

    # Can run forward again
    output = component(test_input)
    assert output is not None
```

**Why consolidate:**
- Tests universal contract (all components should reset properly)
- Same validation logic for all components
- Easy to add new components to parameterization
- Reduces redundant test code

### Checklist: Should I Consolidate?

Ask these questions:

1. ☑️ **Is the setup code identical?**
2. ☑️ **Are the assertions checking the same property?**
3. ☑️ **Is only the input parameter different?**
4. ☐ **Does each test validate component-specific behavior?**
5. ☐ **Would combining tests make failures harder to debug?**

**If 1-3 are YES and 4-5 are NO → Consolidate with parameterization**
**If 4 or 5 is YES → Keep separate tests**

---

## Good vs Bad Test Patterns

### Pattern 1: Hardcoded Dimension Values

The most common brittleness issue in the test suite.

❌ **BAD:**
```python
def test_cortex_input_size():
    """Test cortex receives input from thalamus."""
    brain = create_default_brain()
    cortex = brain.components["cortex"]

    # ❌ Hardcoded value - breaks when config changes
    assert cortex.config.n_input == 64
```

**Problems:**
- Breaks when thalamus size changes
- Breaks when input routing changes
- Tests implementation detail (specific config value), not contract

✅ **GOOD:**
```python
def test_cortex_input_matches_thalamus_output():
    """Test cortex input size matches thalamus output (dimension compatibility)."""
    brain = create_default_brain()
    cortex = brain.components["cortex"]
    thalamus = brain.components["thalamus"]

    # ✅ Tests dimensional compatibility contract
    assert cortex.config.n_input == thalamus.n_output

    # ✅ Tests invariant (positive dimension)
    assert cortex.config.n_input > 0
```

**More examples of dimension compatibility tests:**

```python
# ✅ Test target input matches source output
def test_pathway_dimensions_compatible():
    pathway = brain.connections[("thalamus", "cortex")]
    thalamus = brain.components["thalamus"]
    cortex = brain.components["cortex"]

    # For AxonalProjection (routing only)
    assert pathway.n_output == sum(src.size for src in pathway.sources)

    # Target receives correct input size
    assert cortex.n_input == thalamus.n_output

# ✅ Test weight matrix dimensions
def test_synaptic_weights_match_connectivity():
    region = brain.components["cortex"]

    for source_name, weights in region.synaptic_weights.items():
        source_region = brain.components[source_name]

        # Weights should be [n_neurons, n_input_from_source]
        assert weights.shape[0] == region.n_neurons
        assert weights.shape[1] == source_region.n_output
```

**Common Refactoring Pattern:**

```python
# BEFORE (brittle)
assert brain.components["thalamus"].n_input == 128
assert brain.components["thalamus"].n_output == 128
assert brain.components["striatum"].n_output == 50  # 5 actions * 10 neurons

# AFTER (robust)
config = brain.config
assert brain.components["thalamus"].n_input == config.brain.sizes.input_size
assert brain.components["thalamus"].n_output == config.brain.sizes.thalamus_size
assert brain.components["striatum"].n_output == config.brain.sizes.n_actions * striatum.neurons_per_action
```

---

### Pattern 1a: Hardcoded Config Defaults

Testing that default config values match hardcoded expectations.

❌ **BAD:**
```python
def test_neuron_count():
    region = Cortex(config)
    assert region.n_neurons == 1000  # ❌ Hardcoded default
```

✅ **GOOD:**
```python
def test_neuron_count_matches_config():
    """Test region neuron count matches provided config."""
    config = CortexConfig(n_neurons=1000)  # Explicit value
    region = Cortex(config)

    # ✅ Tests the contract: region uses config value
    assert region.n_neurons == config.n_neurons
```

**Alternative: Test config contract directly**
```python
def test_config_has_valid_neuron_count():
    """Test config provides valid neuron count."""
    config = CortexConfig()  # Use defaults

    # ✅ Test invariants, not exact values
    assert hasattr(config, 'n_neurons')
    assert config.n_neurons > 0
    assert config.n_neurons == int(config.n_neurons)  # Integer
```

---

### Pattern 2: Trivial Assertions

❌ **BAD:**
```python
def test_reset_clears_spikes():
    region = Cortex(config)
    region.reset_state()
    assert torch.all(region.spikes == 0)  # ❌ Trivial
```

✅ **GOOD:**
```python
def test_region_functions_after_reset():
    """Test region processes inputs correctly after reset."""
    region = Cortex(config)

    # Run forward pass
    _ = region(torch.ones(100, dtype=torch.bool))

    # Reset
    region.reset_state()

    # Test BEHAVIOR: should work normally after reset
    output = region(torch.rand(100) > 0.8)

    assert output.dtype == torch.bool
    assert output.shape == (region.n_neurons,)
    assert not torch.isnan(region.membrane).any()

    # Can run multiple steps without issues
    for _ in range(5):
        output = region(torch.rand(100) > 0.8)
        assert not torch.isnan(region.membrane).any()
```

---

### Pattern 3: Internal State Access

❌ **BAD:**
```python
def test_trace_accumulation():
    strategy = STDPStrategy(config)
    strategy.compute_update(weights, pre, post)

    # ❌ Accessing private internal state
    assert strategy._trace_manager.input_trace.sum() > 0
```

✅ **GOOD:**
```python
def test_stdp_uses_spike_traces():
    """Test STDP produces different updates for different spike timing."""
    strategy = STDPStrategy(config)
    weights = torch.ones(10, 20) * 0.5

    # Case 1: Pre before post (LTP)
    pre = torch.tensor([1, 0, 0, 0])
    post = torch.tensor([0, 1, 0, 0])
    weights1, metrics1 = strategy.compute_update(weights.clone(), pre, post)

    strategy.reset()  # Clear traces

    # Case 2: Post before pre (LTD)
    pre = torch.tensor([0, 1, 0, 0])
    post = torch.tensor([1, 0, 0, 0])
    weights2, metrics2 = strategy.compute_update(weights.clone(), pre, post)

    # Test BEHAVIOR: spike timing matters
    assert not torch.allclose(weights1, weights2)
    assert metrics1['ltp'] > metrics2['ltp']  # Pre-before-post → more LTP
```

---

### Pattern 4: Over-Mocking

❌ **BAD:**
```python
def test_visualize_brain(tmp_path):
    # ❌ Everything mocked - doesn't test real behavior
    brain = Mock()
    brain.regions = {'cortex': Mock()}
    brain.pathway_manager = Mock()

    export_topology(brain, tmp_path / "graph.dot")
```

✅ **GOOD:**
```python
def test_visualize_real_brain(tmp_path):
    """Test visualization with real brain structure."""
    # ✅ Use real (minimal) brain
    config = ThaliaConfig(
        global_=GlobalConfig(device="cpu"),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=10,
                thalamus_size=20,
                cortex_size=30,
                n_actions=5,
            ),
        ),
    )
    brain = DynamicBrain.from_thalia_config(config)

    # Test with REAL components
    output_file = tmp_path / "topology.dot"
    export_topology_to_graphviz(brain, str(output_file))

    # Validate real structure
    content = output_file.read_text()
    assert '"thalamus"' in content
    assert '"cortex"' in content
    assert '->' in content  # Has pathways
```

---

### Pattern 5: Non-Pythonic Boolean Assertions

Using `== True` or `== False` instead of direct boolean assertions.

❌ **BAD:**
```python
def test_cerebellum_mode():
    state = cerebellum.state_dict()
    assert state["config"]["use_enhanced"] == True  # ❌ Explicit == True
    assert state["config"]["use_classic"] == False  # ❌ Explicit == False
```

✅ **GOOD:**
```python
def test_cerebellum_mode():
    """Test cerebellum configuration mode."""
    state = cerebellum.state_dict()

    # ✅ Direct boolean assertions (Pythonic)
    assert state["config"]["use_enhanced"]
    assert not state["config"]["use_classic"]
```

**Why this matters:**
- More Pythonic and idiomatic
- Clearer intent ("assert it is true" vs "assert it equals the value True")
- Avoids confusion with truthy/falsy values
- `== True` can fail for truthy non-boolean values (1, non-empty strings)

**Additional examples:**

```python
# ❌ Avoid
assert region.is_active == True
assert pathway.has_delays == False
assert config.enable_learning == True

# ✅ Prefer
assert region.is_active
assert not pathway.has_delays
assert config.enable_learning
```

**Exception:** When testing actual boolean type
```python
# ✅ This is OK when you need to validate the type
assert isinstance(config.enable_learning, bool)
assert type(output) is bool  # Validate dtype
```

---

## Edge Case Testing

### Required Edge Cases for All Components

1. **Silent Input (Zero Spikes)**
2. **Saturated Input (All Spikes)**
3. **Dimension Mismatches**
4. **Invalid Parameter Values**
5. **Reset/State Clearing**

### Template: Region Edge Cases

```python
def test_region_handles_silent_input():
    """Test region handles zero spikes correctly."""
    region = create_test_region()
    silent_input = torch.zeros(region.input_size, dtype=torch.bool)

    output = region(silent_input)

    # Should not crash
    assert output.shape == (region.n_neurons,)
    assert output.dtype == torch.bool
    assert not torch.isnan(region.membrane).any()

def test_region_handles_saturated_input():
    """Test region handles all spikes without overflow."""
    region = create_test_region()
    saturated_input = torch.ones(region.input_size, dtype=torch.bool)

    output = region(saturated_input)

    # Should not overflow or produce NaN
    assert output.shape == (region.n_neurons,)
    assert not torch.isnan(region.membrane).any()
    assert not torch.isinf(region.membrane).any()

def test_region_rejects_wrong_input_size():
    """Test region validates input dimensions."""
    region = create_test_region()

    # Wrong size input
    wrong_input = torch.zeros(region.input_size * 2, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="dimension|shape"):
        _ = region(wrong_input)
```

### Template: Learning Strategy Edge Cases

```python
def test_strategy_no_spikes():
    """Test strategy handles zero spikes."""
    strategy = create_test_strategy()
    weights = torch.rand(10, 20)

    # No spikes
    pre = torch.zeros(20)
    post = torch.zeros(10)

    new_weights, _ = strategy.compute_update(weights.clone(), pre, post)

    # Weights should be unchanged
    assert torch.allclose(new_weights, weights)

def test_strategy_extreme_learning_rates():
    """Test strategy with extreme learning rates."""
    # Zero learning rate - no change
    strategy = create_test_strategy(learning_rate=0.0)
    weights = torch.rand(10, 20)
    new_weights, _ = strategy.compute_update(
        weights.clone(), torch.ones(20), torch.ones(10)
    )
    assert torch.allclose(new_weights, weights)

    # Very high learning rate - should saturate but not NaN
    strategy = create_test_strategy(learning_rate=100.0, w_max=1.0)
    new_weights, _ = strategy.compute_update(
        weights.clone(), torch.ones(20), torch.ones(10)
    )
    assert not torch.isnan(new_weights).any()
    assert new_weights.max() <= 1.0  # Should clip at w_max

def test_strategy_weight_bounds():
    """Test strategy respects weight bounds."""
    strategy = create_test_strategy(w_min=0.0, w_max=1.0)

    # Start at boundaries
    weights_at_max = torch.ones(10, 20)
    weights_at_min = torch.zeros(10, 20)

    # LTP should not exceed max
    new_weights, _ = strategy.compute_update(
        weights_at_max.clone(), torch.ones(20), torch.ones(10)
    )
    assert new_weights.max() <= 1.0

    # LTD should not go below min
    new_weights, _ = strategy.compute_update(
        weights_at_min.clone(), torch.zeros(20), torch.ones(10)
    )
    assert new_weights.min() >= 0.0
```

---

## Network Integrity Validation

### Critical Tests for All Brain Architectures

```python
def test_pathway_dimensions_match_regions(brain):
    """Test that all pathways have compatible dimensions."""
    errors = []

    for pathway_name, pathway in brain.pathway_manager.pathways.items():
        source = brain.regions[pathway.source_name]
        target = brain.regions[pathway.target_name]

        # Validate input dimension
        source_size = getattr(source, 'n_neurons', None) or source.n_output
        if pathway.input_size != source_size:
            errors.append(
                f"{pathway_name}: input_size ({pathway.input_size}) != "
                f"source output ({source_size})"
            )

        # Validate output dimension
        target_size = getattr(target, 'input_size', None) or target.n_input
        if pathway.output_size != target_size:
            errors.append(
                f"{pathway_name}: output_size ({pathway.output_size}) != "
                f"target input ({target_size})"
            )

        # Validate weight shape
        expected = (pathway.output_size, pathway.input_size)
        if pathway.weights.shape != expected:
            errors.append(
                f"{pathway_name}: weights shape {pathway.weights.shape} != {expected}"
            )

    assert len(errors) == 0, "\n".join(errors)

def test_brain_has_no_disconnected_regions(brain):
    """Test that all regions have connectivity."""
    has_input = {name: False for name in brain.regions}
    has_output = {name: False for name in brain.regions}

    for pathway in brain.pathway_manager.pathways.values():
        has_output[pathway.source_name] = True
        has_input[pathway.target_name] = True

    disconnected = []
    for name in brain.regions:
        if name in ['input', 'output', 'sensory', 'motor']:
            continue  # Allow one-way I/O regions

        if not (has_input[name] or has_output[name]):
            disconnected.append(name)

    assert len(disconnected) == 0, f"Disconnected: {disconnected}"

def test_pathway_weights_are_valid(brain):
    """Test that all pathway weights are valid (no NaN, no Inf)."""
    for pathway_name, pathway in brain.pathway_manager.pathways.items():
        weights = pathway.weights

        assert not torch.isnan(weights).any(), \
            f"{pathway_name} has NaN weights"
        assert not torch.isinf(weights).any(), \
            f"{pathway_name} has Inf weights"
```

---

## Test Organization

### Unit Tests (`tests/unit/`)
- **Purpose:** Test individual components in isolation
- **Scope:** Single class or function
- **Mocking:** Minimal (only external dependencies)
- **Files:** One test file per source file

Example:
```
src/thalia/regions/cortex.py
tests/unit/test_cortex.py
```

### Integration Tests (`tests/integration/`)
- **Purpose:** Test component interactions
- **Scope:** Multiple regions, pathways, or systems
- **Mocking:** Minimal (use real components)
- **Files:** Organized by feature or subsystem

Example:
```
tests/integration/test_sensorimotor_loop.py  # Brain → action → brain cycle
tests/integration/test_learning_strategy_pattern.py  # Strategies with regions
tests/integration/test_checkpoint_growth_edge_cases.py  # Error handling
```

### Test File Structure

```python
"""
Brief description of what is being tested.

Test Coverage:
- Feature 1
- Feature 2
- Edge cases
"""

import pytest
import torch

from thalia.regions import RegionClass
from thalia.config import RegionConfig


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def config(device):
    """Standard configuration for tests."""
    return RegionConfig(
        n_input=100,
        n_output=50,
        device=device,
    )


@pytest.fixture
def region(config):
    """Create region instance."""
    region = RegionClass(config)
    region.reset_state()
    return region


class TestBasicBehavior:
    """Test basic functionality."""

    def test_forward_pass(self, region):
        """Test region processes input correctly."""
        # Test implementation
        pass


class TestEdgeCases:
    """Test boundary conditions."""

    def test_silent_input(self, region):
        """Test region handles zero spikes."""
        # Test implementation
        pass

    def test_saturated_input(self, region):
        """Test region handles all spikes."""
        # Test implementation
        pass


class TestBiologicalPlausibility:
    """Test biological properties."""

    def test_spike_timing(self, region):
        """Test temporal dynamics."""
        # Test implementation
        pass
```

---

## Common Pitfalls

### 1. Testing Random Behavior

❌ **BAD:**
```python
def test_softmax_selection():
    selector = ActionSelector(mode=SelectionMode.SOFTMAX)
    votes = torch.tensor([10.0, 5.0, 3.0])

    action, _ = selector.select_action(votes)

    # ❌ Can't reliably test random selection in single trial
    assert action == 0  # Might fail randomly!
```

✅ **GOOD:**
```python
def test_softmax_selection():
    """Test softmax produces probabilistic distribution."""
    selector = ActionSelector(mode=SelectionMode.SOFTMAX, temperature=1.0)
    votes = torch.tensor([10.0, 5.0, 3.0, 2.0])

    # Run multiple trials
    action_counts = torch.zeros(4)
    n_trials = 100

    for _ in range(n_trials):
        action, info = selector.select_action(votes)
        action_counts[action] += 1
        assert info['probabilities'] is not None

    # Test DISTRIBUTION: highest vote should be chosen most often
    assert action_counts[0] > action_counts[1]
    assert action_counts[1] > action_counts[2]
    assert action_counts.sum() == n_trials
```

---

### 2. Not Using Fixtures

❌ **BAD:**
```python
def test_something():
    config = ThaliaConfig(...)  # Repeated in every test
    brain = DynamicBrain.from_thalia_config(config)
    # Test code
```

✅ **GOOD:**
```python
@pytest.fixture
def test_brain():
    """Create standard brain for tests."""
    config = ThaliaConfig(...)
    return DynamicBrain.from_thalia_config(config)

def test_something(test_brain):
    # Use fixture
    output = test_brain.step(input_spikes)
```

---

### 3. Testing Multiple Things in One Test

❌ **BAD:**
```python
def test_region_everything():
    """Test region (DOES TOO MUCH)."""
    region = Cortex(config)

    # Test 1: Forward pass
    output = region(input1)
    assert output.shape == (100,)

    # Test 2: Reset
    region.reset_state()
    assert region.membrane.sum() == 0

    # Test 3: Learning
    region.learn(pre, post)
    assert region.weights.mean() > 0.5

    # Test 4: Neuromodulation
    region.set_neuromodulators(dopamine=0.8)
    # ... more tests
```

✅ **GOOD:**
```python
def test_forward_pass():
    """Test region forward pass."""
    region = Cortex(config)
    output = region(input1)
    assert output.shape == (100,)

def test_reset():
    """Test region reset clears state."""
    region = Cortex(config)
    region(input1)
    region.reset_state()
    # Test reset behavior

def test_learning():
    """Test region learning updates weights."""
    region = Cortex(config)
    # Test learning

def test_neuromodulation():
    """Test dopamine modulates learning."""
    region = Cortex(config)
    # Test neuromodulation
```

---

### 4. Not Testing Error Conditions

❌ **BAD:**
```python
def test_load_checkpoint():
    """Test loading checkpoint."""
    region = Striatum(config)
    region.load_checkpoint(valid_checkpoint)
    # Only tests happy path
```

✅ **GOOD:**
```python
def test_load_checkpoint():
    """Test loading valid checkpoint."""
    region = Striatum(config)
    region.load_checkpoint(valid_checkpoint)
    # Test successful load

def test_load_corrupted_checkpoint():
    """Test loading corrupted checkpoint raises error."""
    region = Striatum(config)
    corrupted = {"wrong": "structure"}

    with pytest.raises(ValueError, match="Invalid checkpoint"):
        region.load_checkpoint(corrupted)

def test_load_version_mismatch():
    """Test loading old checkpoint version raises error."""
    region = Striatum(config)
    old_checkpoint = {"version": "1.0.0", ...}

    with pytest.raises(ValueError, match="Incompatible version"):
        region.load_checkpoint(old_checkpoint)
```

---

## Test Quality Checklist

Before submitting a PR with new tests, verify:

- [ ] **No Private Attributes:** Tests don't access `_variables`
- [ ] **No Hardcoded Defaults:** Tests don't assert default config values
- [ ] **Behavioral Contracts:** Tests validate what component does, not how
- [ ] **Edge Cases:** Tests include silent input, saturated input, invalid input
- [ ] **Error Conditions:** Tests validate error handling
- [ ] **No Over-Mocking:** Tests use real components when possible
- [ ] **Network Integrity:** Integration tests validate dimension compatibility
- [ ] **Fixtures:** Tests use fixtures to reduce duplication
- [ ] **Single Responsibility:** Each test validates one behavior
- [ ] **Descriptive Names:** Test names clearly state what is tested

---

## Examples from Codebase

### ✅ Excellent Tests

**`test_checkpoint_growth_edge_cases.py`**
- Tests corrupted checkpoints
- Tests version mismatches
- Tests memory limits
- Tests recovery strategies
- Uses descriptive error messages

**`test_sparse_learning.py`**
- Tests no spikes
- Tests single spike
- Tests all spikes
- Tests threshold crossover
- Validates sparse/dense equivalence

**`test_thalamus.py`**
- Tests silent input
- Tests saturated input
- Tests neuromodulation
- Tests reset behavior
- Uses behavioral contracts

### ⚠️ Tests Needing Improvement

**`test_striatum_d1d2_delays.py`**
- ❌ Accesses `_d1_delay_steps` (private)
- ❌ Accesses `_d2_delay_buffer` (private)
- ✅ Good edge case testing (saturated input)
- **Fix:** Replace private attribute access with behavioral tests

**`test_network_visualization.py`**
- ❌ Over-uses mocks
- ❌ Hardcodes neuron counts
- **Fix:** Use real minimal brains instead of mocks

---

## Refactoring Patterns (From Phase 2 Improvements)

The following patterns were established during Phase 2 test refactoring (December 2025), where 46+ internal state assertions were successfully replaced with behavioral contracts.

### Pattern 1: Config-Based Validation

**Problem:** Tests access private implementation details to verify configuration.

❌ **BEFORE (Brittle):**
```python
def test_delay_configuration():
    config = StriatumConfig(d1_to_output_delay_ms=15.0, dt_ms=1.0)
    striatum = Striatum(config)

    # Accessing private attributes
    assert striatum._d1_delay_steps == 15
    assert striatum._d2_delay_steps == 25
```

✅ **AFTER (Robust):**
```python
def test_delay_configuration():
    config = StriatumConfig(d1_to_output_delay_ms=15.0, dt_ms=1.0)

    # Calculate expected value from config (public contract)
    expected_d1_delay_steps = int(config.d1_to_output_delay_ms / config.dt_ms)
    assert expected_d1_delay_steps == 15

    # Or better: test the BEHAVIOR, not the config
    # (See Pattern 4: Behavioral Validation)
```

---

### Pattern 2: Public API Access

**Problem:** Tests access private attributes instead of using public getters.

❌ **BEFORE (Brittle):**
```python
def test_vote_accumulation():
    striatum = Striatum(config)
    striatum(input_spikes)

    # Accessing private state
    d1_votes = striatum._d1_votes_accumulated
    d2_votes = striatum._d2_votes_accumulated
```

✅ **AFTER (Robust):**
```python
def test_vote_accumulation():
    striatum = Striatum(config)
    striatum(input_spikes)

    # Use public API
    d1_votes, d2_votes = striatum.get_accumulated_votes()
```

---

### Pattern 3: Checkpoint Metadata Inspection

**Problem:** Tests call private decision-making methods instead of inspecting results.

❌ **BEFORE (Brittle):**
```python
def test_format_selection():
    region = Cortex(small_config)

    # Calling private decision method
    should_use_neuromorphic = region._should_use_neuromorphic()
    assert should_use_neuromorphic == True
```

✅ **AFTER (Robust):**
```python
def test_format_selection():
    region = Cortex(small_config)

    # Save checkpoint and inspect what format was actually used
    checkpoint = region.save_checkpoint()

    # Verify behavior, not decision logic
    assert checkpoint['format'] == 'neuromorphic'
    assert 'neurons' in checkpoint
```

---

### Pattern 4: Behavioral Validation

**Problem:** Tests verify implementation details instead of observable behavior.

❌ **BEFORE (Brittle):**
```python
def test_circular_buffer():
    striatum = Striatum(config)

    # Multiple forward passes
    for _ in range(100):
        striatum(input_spikes)

    # Checking internal buffer pointer
    assert striatum._delay_ptr == (100 % striatum._buffer_size)
```

✅ **AFTER (Robust):**
```python
def test_long_run_stability():
    """Test striatum remains stable over many iterations."""
    striatum = Striatum(config)

    # Run many iterations
    for step in range(1000):
        output = striatum(input_spikes)

        # Verify BEHAVIOR: no crashes, no NaN, valid output
        assert not torch.isnan(output).any(), f"NaN at step {step}"
        assert output.shape == (striatum.n_actions,)
```

---

### Pattern 5: Learning Strategy Metrics

**Problem:** Tests access private trace managers instead of using metrics.

❌ **BEFORE (Brittle):**
```python
def test_stdp_traces():
    stdp = create_strategy("stdp", learning_rate=0.001)

    # Accessing private trace manager
    assert stdp._trace_manager is not None
    assert stdp._trace_manager.input_trace.sum() == 0
```

✅ **AFTER (Robust):**
```python
def test_stdp_learning():
    """Test STDP produces learning signal."""
    stdp = create_strategy("stdp", learning_rate=0.001)
    weights = torch.rand(10, 20)
    pre_spikes = torch.rand(20) > 0.8
    post_spikes = torch.rand(10) > 0.8

    # Apply learning
    new_weights, metrics = stdp.compute_update(
        weights=weights,
        pre_spikes=pre_spikes,
        post_spikes=post_spikes,
    )

    # Verify BEHAVIOR via metrics (public API)
    assert 'weight_change_mean' in metrics
    assert 'weight_change_std' in metrics
    assert not torch.isnan(new_weights).any()
```

---

### Pattern 6: Coordination via Public Methods

**Problem:** Tests verify helper methods instead of end-to-end coordination.

❌ **BEFORE (Brittle):**
```python
def test_find_input_pathways():
    coordinator = GrowthCoordinator(brain)

    # Testing private helper method
    input_pathways = coordinator._find_input_pathways('cortex')
    assert len(input_pathways) == 2
```

✅ **AFTER (Robust):**
```python
def test_coordinate_growth_updates_input_pathways():
    """Test that growing a region also grows connected pathways."""
    brain = create_test_brain()
    coordinator = GrowthCoordinator(brain)

    # Record initial pathway dimensions
    pathway = brain.pathway_manager.pathways['visual_to_cortex']
    initial_output_size = pathway.output_size

    # Grow region via public API
    events = coordinator.coordinate_growth(
        region_name='cortex',
        n_new_neurons=100,
        reason='test'
    )

    # Verify BEHAVIOR: pathway updated to match new region size
    assert pathway.output_size == initial_output_size + 100
    assert len(events) >= 2  # Region + pathway events
```

---

## Test Quality Audit (December 21, 2025)

### Audit Summary

A comprehensive audit was conducted on December 21, 2025, analyzing 50+ test files to identify weak patterns and improvement opportunities. The audit revealed the test suite has **STRONG fundamentals** with excellent edge case coverage, but contains opportunities to reduce brittleness.

**Overall Assessment:** B+ (Good, with improvement opportunities)

**Key Findings:**

1. ✅ **Strengths:**
   - Comprehensive edge case testing (silent/saturated input)
   - Excellent network integrity validation
   - Good use of behavioral assertions
   - Well-documented patterns in `WRITING_TESTS.md`

2. ⚠️ **Areas for Improvement:**
   - 20+ hardcoded value assertions (brittle)
   - 18 private attribute tests (coupled to implementation)
   - 19 trivial "not None" checks (low value)
   - Limited learning strategy stress tests
   - Some tests lack intermediate state validation

**Full Reports:**
- `TEST_QUALITY_AUDIT_REPORT.md` - Detailed findings and analysis
- `TEST_IMPROVEMENT_CHECKLIST.md` - Step-by-step implementation plan

---

### Critical Anti-Patterns Identified

#### Anti-Pattern 1: Hardcoded Default Values ❌

**Found in:** 14 test files | **Severity:** Medium

```python
# ❌ BRITTLE: Breaks when default changes
def test_initialization():
    region = NeuralRegion(n_neurons=100, device="cpu")
    assert region.n_neurons == 100  # Hardcoded exact value

# ✅ ROBUST: Tests contract, survives refactoring
def test_initialization():
    config = NeuralRegionConfig(n_neurons=100)
    region = NeuralRegion(config=config, device="cpu")
    assert region.n_neurons == config.n_neurons  # Matches config
    assert region.n_neurons > 0  # Positive count invariant
```

**Fix:** Replace exact assertions with config-based or range checks.

---

#### Anti-Pattern 2: Private Attribute Testing ❌

**Found in:** 5 test files, 18 assertions | **Severity:** High

```python
# ❌ BRITTLE: Coupled to internal implementation
def test_dopamine_storage():
    striatum = Striatum(config)
    striatum.set_neuromodulators(dopamine=1.0)
    assert striatum._tonic_dopamine == 1.0  # PRIVATE

# ✅ ROBUST: Tests through public API
def test_dopamine_affects_behavior():
    striatum = Striatum(config)
    striatum.set_neuromodulators(dopamine=1.0)

    # Verify dopamine propagated (public API)
    diagnostics = striatum.get_diagnostics()
    assert "dopamine" in diagnostics
    assert 0.5 <= diagnostics["dopamine"] <= 1.5  # Valid range
```

**Fix:** Use public APIs or test observable behavior instead of internals.

---

#### Anti-Pattern 3: Trivial "not None" Checks ❌

**Found in:** 7 test files, 19 assertions | **Severity:** Low

```python
# ❌ TRIVIAL: Checks existence, not validity
def test_output():
    output = region(input_spikes)
    assert output is not None  # Low value

# ✅ MEANINGFUL: Validates the value
def test_output():
    output = region(input_spikes)
    assert output.shape == (region.n_neurons,)
    assert output.dtype == torch.bool
    assert not torch.isnan(output.float()).any()
    assert 0 <= output.sum() <= region.n_neurons
```

**Fix:** Either remove or enhance with actual validation.

---

### Improvement Roadmap

**Phase 1 (P1 - High Priority):** 12-16 hours
1. Replace hardcoded value assertions (14 files)
2. Remove private attribute tests (5 files, 18 assertions)
3. Add learning strategy stress tests (15-20 new tests)

**Phase 2 (P2 - Medium Priority):** 8-10 hours
1. Enhance/remove trivial assertions (7 files, 19 assertions)
2. Add value validation to shape assertions (30+ occurrences)
3. Add neuromodulator edge case tests (15+ new tests)

**Phase 3 (P3 - Low Priority):** 4-6 hours
1. Parameterize similar tests
2. Improve test documentation
3. Remove redundant assertions

**Detailed Plan:** See `TEST_IMPROVEMENT_CHECKLIST.md` for step-by-step actions.

---

### Key Lessons from Audit

1. **Test Contracts, Not Defaults**
   - ❌ Don't assert `region.n_neurons == 100`
   - ✅ Do assert `region.n_neurons == config.n_neurons`

2. **Avoid Private Attributes**
   - ❌ Don't access `region._internal_buffer`
   - ✅ Do use `diagnostics = region.get_diagnostics()`

3. **Meaningful Assertions Only**
   - ❌ Don't assert `output is not None`
   - ✅ Do validate shape, dtype, NaN, and range

4. **Comprehensive Edge Cases**
   - Silent input (zero spikes)
   - Saturated input (all spikes)
   - Extreme parameters (LR=100.0)
   - Boundary conditions (weights at w_min/w_max)
   - Invalid inputs (dimension mismatches)

5. **Behavioral Validation**
   - Test what the component does (behavior)
   - Not how it does it (implementation)
   - Use public APIs exclusively
   - Verify observable effects

---

## Getting Help

- **Questions?** Ask in team channel or open GitHub discussion
- **Found a test that doesn't follow these patterns?** Open an issue or PR
- **Want to add new patterns?** Update this guide and submit PR

---

## Version History

- **1.0** (2025-12-13): Initial version based on test quality audit
- **1.1** (2025-12-13): Added Phase 2 refactoring patterns (6 patterns established)
- **1.2** (2025-12-21): Added comprehensive audit findings and improvement roadmap
