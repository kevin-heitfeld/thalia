# THALIA Test Suite

Comprehensive testing framework for the THALIA spiking neural network library.

## Quick Start

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m ablation      # Ablation studies only

# Run with coverage report
pytest --cov=thalia --cov-report=html --cov-report=term

# Run fast tests only (exclude slow tests)
pytest -m "not slow"

# Run specific test file
pytest tests/unit/test_core.py -v

# Run specific test
pytest tests/unit/test_core.py::TestLIFNeuron::test_spike_generation -v
```

## Test Organization

The test suite is organized into three main categories:

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation:

- **`test_core.py`**: Core neuron models (LIF, ConductanceLIF, Dendritic neurons)
- **`test_robustness.py`**: Stability mechanisms (E/I balance, normalization, homeostasis)
- **`test_language.py`**: Language interface (encoding, decoding, position encoding)
- **`test_brain_regions.py`**: Brain regions (cortex, hippocampus, striatum, etc.)
- **`test_predictive_attention.py`**: Predictive coding and attention mechanisms
- **`test_health_dashboard.py`**: Health monitoring and diagnostics
- **`test_validation.py`**: Input validation and edge cases
- **`test_properties.py`**: Property-based tests (requires hypothesis)

**Run unit tests:**
```bash
pytest tests/unit/ -v
```

### 2. Integration Tests (`tests/integration/`)

Test component interactions and full system behavior:

- **`test_cortex_with_robustness.py`**: Cortex with stability mechanisms
- **`conftest.py`**: Shared fixtures for integration testing

**Run integration tests:**
```bash
pytest tests/integration/ -v
```

### 3. Ablation Tests (`tests/ablation/`)

Quantify the contribution of individual mechanisms:

- **`test_without_ei_balance.py`**: Impact of E/I balance removal
- **`test_without_divisive_norm.py`**: Impact of divisive normalization removal
- **`test_without_intrinsic_plasticity.py`**: Impact of intrinsic plasticity removal
- **`test_without_any_robustness.py`**: Impact of removing all robustness mechanisms

**Run ablation tests:**
```bash
pytest tests/ablation/ -v -s  # -s shows print output for analysis
```

## Test Markers

Tests are marked with pytest markers for flexible execution:

- **`@pytest.mark.unit`**: Unit tests (fast, isolated)
- **`@pytest.mark.integration`**: Integration tests (may be slower)
- **`@pytest.mark.ablation`**: Ablation studies (quantify mechanism contributions)
- **`@pytest.mark.slow`**: Tests that take >1 second
- **`@pytest.mark.cuda`**: Tests requiring GPU/CUDA
- **`@pytest.mark.visual`**: Tests that generate visualizations
- **`@pytest.mark.benchmark`**: Performance regression tests

### Filtering by Markers

```bash
# Run only unit tests
pytest -m unit

# Run integration and unit tests
pytest -m "unit or integration"

# Run everything except slow tests
pytest -m "not slow"

# Run CUDA tests only (if GPU available)
pytest -m cuda

# Run ablation studies with output
pytest -m ablation -s
```

## Coverage

The test suite aims for >80% code coverage.

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=thalia --cov-report=term

# HTML report (opens in browser)
pytest --cov=thalia --cov-report=html
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows

# Fail if coverage < 80%
pytest --cov=thalia --cov-fail-under=80
```

### Coverage Configuration

Coverage settings are in `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/thalia"]
omit = ["*/tests/*", "*/temp/*", "*/experiments/*"]

[tool.coverage.report]
fail_under = 80
```

## Test Utilities

Common utilities are in `tests/test_utils.py`:

### Assertion Helpers

```python
from tests.test_utils import (
    assert_spike_train_valid,
    assert_weights_healthy,
    assert_membrane_potential_valid,
    assert_activity_in_range,
    assert_convergence,
)

# Example usage
def test_my_neuron():
    neuron = LIFNeuron(n_neurons=10)
    neuron.reset_state(batch_size=4)

    spikes, _ = neuron(torch.randn(4, 10))

    # Validate spike train
    assert_spike_train_valid(spikes)

    # Validate membrane potential
    assert_membrane_potential_valid(
        neuron.membrane,
        v_rest=0.0,
        v_threshold=1.0
    )
```

### Test Data Generators

```python
from tests.test_utils import (
    generate_poisson_spikes,
    generate_clustered_spikes,
    generate_pattern_sequence,
)

# Generate Poisson spike train
spikes = generate_poisson_spikes(
    rate=0.1,
    n_neurons=100,
    n_timesteps=1000,
    batch_size=4
)

# Generate clustered patterns
clustered = generate_clustered_spikes(
    cluster_size=10,
    n_clusters=5,
    n_timesteps=100
)
```

## Writing Good Tests

### 1. Use Descriptive Names

```python
# ❌ Bad
def test_1():
    ...

# ✅ Good
def test_membrane_decays_toward_rest_without_input():
    ...
```

### 2. Add Docstrings

```python
def test_spike_generation(self):
    """Test that spikes are generated when threshold is crossed.

    This verifies the core spiking mechanism:
    1. Membrane potential integrates input
    2. Spike is generated when V > threshold
    3. Membrane resets after spike
    """
    ...
```

### 3. Use Specific Assertions

```python
# ❌ Vague
assert spikes.sum() > 0

# ✅ Specific
MIN_EXPECTED_SPIKES = 5
assert spikes.sum() >= MIN_EXPECTED_SPIKES, \
    f"Expected at least {MIN_EXPECTED_SPIKES} spikes, got {spikes.sum()}"
```

### 4. Test Edge Cases

```python
def test_handles_zero_input(self):
    """Test that zero input doesn't crash."""
    neuron = LIFNeuron(n_neurons=10)
    neuron.reset_state(batch_size=4)

    # Should handle gracefully
    spikes, _ = neuron(torch.zeros(4, 10))
    assert_spike_train_valid(spikes)

def test_handles_single_neuron(self):
    """Test edge case of single neuron."""
    neuron = LIFNeuron(n_neurons=1)
    ...
```

### 5. Use Fixtures for Common Setup

```python
@pytest.fixture
def standard_cortex():
    """Create a standard cortex for testing."""
    config = LayeredCortexConfig(n_input=128, n_output=64)
    return LayeredCortex(config)

def test_cortex_forward(standard_cortex):
    """Test cortex forward pass."""
    output = standard_cortex.forward(torch.randn(4, 128))
    assert output.shape == (4, 64)
```

### 6. Mark Slow Tests

```python
@pytest.mark.slow
def test_long_training_run(self):
    """Test training over 1000 epochs (takes ~30s)."""
    ...
```

## Continuous Integration

### Pre-commit Hooks

Run tests before committing:

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest -m "not slow" --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

### GitHub Actions

Example workflow (`.github/workflows/test.yml`):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest --cov=thalia --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests are flaky (sometimes pass, sometimes fail)

- **Cause**: Non-deterministic behavior (random seeds not set)
- **Solution**: All tests automatically set seeds via `conftest.py` fixture
- If still flaky, add explicit seed in test:
  ```python
  def test_my_test(self):
      torch.manual_seed(42)  # Explicit seed
      ...
  ```

### Tests are too slow

```bash
# Profile slow tests
pytest --durations=10

# Run only fast tests
pytest -m "not slow"

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

### Coverage is low

```bash
# Find uncovered lines
pytest --cov=thalia --cov-report=term-missing

# Focus on specific module
pytest --cov=thalia.core --cov-report=term-missing
```

### GPU tests fail on CPU-only machine

```bash
# Skip CUDA tests
pytest -m "not cuda"
```

## Best Practices

1. **Keep tests fast**: Unit tests should run in <100ms each
2. **Keep tests isolated**: Each test should set up its own state
3. **Keep tests deterministic**: Use fixtures to set random seeds
4. **Keep tests readable**: Use descriptive names and clear assertions
5. **Keep tests maintainable**: Extract common patterns to `test_utils.py`

## Test Quality Metrics

Our test suite targets:

- **Organization**: 9/10 ✅
- **Coverage**: 8/10 ✅ (>80%)
- **Assertions**: 9/10 ✅ (Specific, with clear error messages)
- **Reliability**: 9/10 ✅ (Deterministic, reproducible)
- **Edge Cases**: 8/10 ✅ (Validation tests, property-based tests)
- **Performance**: 7/10 ✅ (Benchmark tests, profiling)
- **Documentation**: 9/10 ✅ (This README, docstrings)

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/) (property-based testing)
- [THALIA documentation](../README.md)

## Questions?

See [GitHub Issues](https://github.com/kevin-heitfeld/thalia/issues) or the main [README](../README.md).
