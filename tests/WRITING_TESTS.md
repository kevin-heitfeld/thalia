# Quick Reference: Writing Good Tests

A concise guide for writing high-quality tests in the THALIA project.

## ✅ DO: Best Practices

### 1. Descriptive Names
```python
# ✅ GOOD
def test_membrane_decays_toward_rest_without_input(self):
    """Test that membrane potential decays to rest when no input is applied."""
    
# ❌ BAD
def test_membrane(self):
    """Test membrane."""
```

### 2. Clear Single Responsibility
```python
# ✅ GOOD - Tests one thing
def test_spike_generation_above_threshold(self):
    neuron.membrane = torch.full((1, 10), 1.5)  # Above threshold
    spikes, _ = neuron(torch.zeros(1, 10))
    assert spikes.sum() > 0

# ❌ BAD - Tests multiple things
def test_neuron(self):
    # Tests initialization, forward pass, reset, and more...
```

### 3. Use Custom Assertions
```python
# ✅ GOOD - Clear intent and error messages
from tests.test_utils import assert_spike_train_valid, assert_membrane_potential_valid

assert_spike_train_valid(spikes)
assert_membrane_potential_valid(membrane, v_rest=0.0, v_threshold=1.0)

# ❌ BAD - Generic assertions
assert spikes.sum() > 0
assert membrane.max() < 2.0
```

### 4. Meaningful Constants
```python
# ✅ GOOD - Self-documenting
MIN_HEALTHY_ACTIVITY = 2.0  # Spikes per timestep
RUNAWAY_THRESHOLD = 2.0  # 2x growth indicates runaway
assert avg_spikes > MIN_HEALTHY_ACTIVITY

# ❌ BAD - Magic numbers
assert avg_spikes > 0.1
assert late_spikes < early_spikes * 2
```

### 5. Proper State Management
```python
# ✅ GOOD - Reset once, let mechanisms adapt
cortex.reset_state()
for t in range(100):
    output = cortex.forward(test_inputs[t])

# ❌ BAD - Resetting every iteration
for t in range(100):
    cortex.reset_state()  # Prevents adaptation!
    output = cortex.forward(test_inputs[t])
```

### 6. Pre-generate Test Data
```python
# ✅ GOOD - Reproducible
test_inputs = torch.randn(100, 1, 32) * 0.5
for t in range(100):
    output = cortex.forward(test_inputs[t])

# ❌ BAD - New random data each iteration
for t in range(100):
    input = torch.randn(1, 32) * 0.5  # Different every time
    output = cortex.forward(input)
```

### 7. Comprehensive Error Testing
```python
# ✅ GOOD - Test error paths
def test_forward_before_reset_raises_error(self):
    neuron = LIFNeuron(n_neurons=10)
    with pytest.raises((RuntimeError, AttributeError)):
        neuron(torch.randn(1, 10))

# ✅ GOOD - Test edge cases
def test_handles_zero_input(self):
    spikes, _ = neuron(torch.zeros(4, 10))
    assert_spike_train_valid(spikes)
```

### 8. Use Fixtures for Common Setup
```python
# ✅ GOOD - Reusable setup
@pytest.fixture
def standard_cortex():
    """Create standard cortex for testing."""
    config = LayeredCortexConfig(n_input=128, n_output=64)
    cortex = LayeredCortex(config)
    cortex.reset_state()
    return cortex

def test_cortex_forward(standard_cortex):
    output = standard_cortex.forward(torch.randn(4, 128))
    assert output.shape == (4, 64)
```

### 9. Mark Slow Tests
```python
# ✅ GOOD - Marked so can be skipped
@pytest.mark.slow
def test_long_training_run(self):
    """Test training over 1000 epochs (takes ~30s)."""
    # ...

# Run fast tests only: pytest -m "not slow"
```

### 10. Test Utilities
```python
# ✅ GOOD - Use helper functions
from tests.test_utils import (
    create_test_spike_pattern,
    assert_no_memory_leak,
    benchmark_function,
)

spikes = create_test_spike_pattern('poisson', rate=0.1, n_neurons=100)
assert_no_memory_leak(training_step, n_iterations=100)
stats = benchmark_function(forward_pass, n_runs=50)
```

## ❌ DON'T: Anti-patterns

### 1. Don't Test Implementation Details
```python
# ❌ BAD - Tests internal implementation
def test_uses_specific_algorithm(self):
    assert neuron._use_euler_method == True  # Brittle!

# ✅ GOOD - Tests behavior
def test_membrane_integrates_input(self):
    neuron(torch.ones(1, 10) * 0.3)
    assert neuron.membrane.mean() > initial_membrane
```

### 2. Don't Make Tests Dependent
```python
# ❌ BAD - Tests depend on order
class TestNeuron:
    def test_1_initialize(self):
        self.neuron = LIFNeuron(10)  # Shared state!
    
    def test_2_forward(self):
        self.neuron(...)  # Depends on test_1

# ✅ GOOD - Independent tests
class TestNeuron:
    def test_forward(self):
        neuron = LIFNeuron(10)  # Fresh instance
        neuron.reset_state(batch_size=4)
        neuron(...)
```

### 3. Don't Use Weak Assertions
```python
# ❌ BAD - Vague
assert spikes.sum() > 0  # Could be 0.001, still passes

# ✅ GOOD - Specific
MIN_EXPECTED_SPIKES = 5
assert spikes.sum() >= MIN_EXPECTED_SPIKES, \
    f"Expected at least {MIN_EXPECTED_SPIKES}, got {spikes.sum()}"
```

### 4. Don't Skip Error Handling
```python
# ❌ BAD - Only test happy path
def test_neuron_forward(self):
    neuron.reset_state(batch_size=4)
    spikes, _ = neuron(torch.randn(4, 10))
    # What about wrong batch size? NaN input? etc.

# ✅ GOOD - Test error paths too
def test_wrong_batch_size_raises_error(self):
    neuron.reset_state(batch_size=4)
    with pytest.raises((ValueError, RuntimeError)):
        neuron(torch.randn(8, 10))
```

### 5. Don't Make Flaky Tests
```python
# ❌ BAD - Flaky due to randomness
def test_learns_pattern(self):
    # Might work 90% of the time, fail 10%
    if random.rand() > 0.9:
        ...

# ✅ GOOD - Deterministic with seed
def test_learns_pattern(self):
    torch.manual_seed(42)  # Reproducible
    # Now deterministic
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test single components in isolation
- Fast (<100ms per test)
- No complex dependencies
- High coverage of edge cases

### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- May be slower (100ms-1s)
- Test realistic scenarios
- Focus on critical paths

### Ablation Tests (`@pytest.mark.ablation`)
- Quantify mechanism contributions
- Compare baseline vs ablated
- Print detailed metrics
- Document architectural decisions

### Performance Tests (`@pytest.mark.benchmark`)
- Measure execution time
- Detect regressions
- Track memory usage
- Set clear thresholds

## Common Test Patterns

### Pattern 1: Arrange-Act-Assert
```python
def test_spike_generation(self):
    # Arrange
    neuron = LIFNeuron(n_neurons=10)
    neuron.reset_state(batch_size=4)
    
    # Act
    spikes, _ = neuron(torch.randn(4, 10) * 2.0)
    
    # Assert
    assert_spike_train_valid(spikes)
    assert spikes.sum() > 0
```

### Pattern 2: Parametrized Tests
```python
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_various_batch_sizes(self, batch_size):
    neuron = LIFNeuron(n_neurons=10)
    neuron.reset_state(batch_size=batch_size)
    spikes, _ = neuron(torch.randn(batch_size, 10))
    assert spikes.shape == (batch_size, 10)
```

### Pattern 3: Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    n_neurons=st.integers(min_value=1, max_value=1000),
    batch_size=st.integers(min_value=1, max_value=128),
)
def test_shape_consistency(self, n_neurons, batch_size):
    neuron = LIFNeuron(n_neurons=n_neurons)
    neuron.reset_state(batch_size=batch_size)
    spikes, _ = neuron(torch.randn(batch_size, n_neurons))
    assert spikes.shape == (batch_size, n_neurons)
```

### Pattern 4: Fixture-Based Setup
```python
@pytest.fixture
def trained_network():
    """Fixture providing a pre-trained network."""
    network = create_network()
    train(network, n_epochs=10)
    return network

def test_inference(trained_network):
    output = trained_network(test_input)
    assert output.shape == expected_shape
```

## Error Messages

### Good Error Messages
```python
# ✅ Provides context
assert avg_spikes > MIN_HEALTHY_ACTIVITY, \
    f"Activity too low: {avg_spikes:.1f} < {MIN_HEALTHY_ACTIVITY} " \
    f"(robustness mechanisms may have failed)"

# ✅ Includes actual vs expected
assert actual_shape == expected_shape, \
    f"Shape mismatch: got {actual_shape}, expected {expected_shape}"
```

### Bad Error Messages
```python
# ❌ No context
assert x > 0.1  # Why 0.1? What is x?

# ❌ Not helpful
assert condition, "Failed"  # What failed? Why?
```

## Quick Checklist

Before committing a test, check:

- [ ] Test name clearly describes what is tested
- [ ] Has docstring explaining the test
- [ ] Uses specific assertions with clear error messages
- [ ] Properly manages state (reset once, not in loops)
- [ ] Pre-generates test data for reproducibility
- [ ] Tests both happy path and error cases
- [ ] Marks slow tests with `@pytest.mark.slow`
- [ ] Independent (doesn't depend on other tests)
- [ ] Uses fixtures for common setup
- [ ] Follows project conventions

## Resources

- Full test documentation: `tests/README.md`
- Test utilities: `tests/test_utils.py`
- Recent improvements: `tests/TEST_IMPROVEMENTS.md`
- Pytest docs: https://docs.pytest.org/
- Hypothesis docs: https://hypothesis.readthedocs.io/
