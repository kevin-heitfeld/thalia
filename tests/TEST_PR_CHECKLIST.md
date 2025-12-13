# Test Quality PR Review Checklist

Use this checklist when reviewing PRs that add or modify tests.

## ✅ Good Test Characteristics

### Contract Testing (Not Implementation)
- [ ] Tests observable behavior, not internal variables
- [ ] Does NOT access private attributes (`_variables`)
- [ ] Tests what component does, not how it does it
- [ ] Survives refactoring of internal implementation

### Explicit Values (Not Defaults)
- [ ] Does NOT assert default configuration values
- [ ] Uses explicit values in test setup
- [ ] Tests that configs have required fields and valid ranges
- [ ] Tests contracts, not exact default values

### Real Components (Not Mocks)
- [ ] Uses real components when possible
- [ ] Mocks only external dependencies (file I/O, plotting)
- [ ] Integration tests use real regions/pathways
- [ ] Tests validate real behavior, not mock behavior

### Edge Case Coverage
- [ ] Tests silent input (zero spikes)
- [ ] Tests saturated input (all spikes)
- [ ] Tests dimension mismatches
- [ ] Tests invalid parameter values
- [ ] Tests error conditions with descriptive error messages

### Network Integrity (Integration Tests)
- [ ] Validates pathway dimensions match regions
- [ ] Validates no disconnected regions
- [ ] Validates weight matrices have correct shape
- [ ] Validates no NaN/Inf in outputs

### Test Organization
- [ ] Uses fixtures to reduce duplication
- [ ] Each test validates one behavior
- [ ] Test names clearly state what is tested
- [ ] Tests are independent (don't depend on execution order)
- [ ] Tests are deterministic (seed torch/numpy random)

## ❌ Bad Test Patterns (Reject These)

### Internal State Access
```python
# ❌ BAD
assert region._internal_buffer.shape == (100,)
assert pathway._delay_steps == 15

# ✅ GOOD
assert region.get_buffer_size() == 100  # Public API
assert pathway.get_delay_ms() == 15.0   # Public API
```

### Hardcoded Configuration Values
```python
# ❌ BAD
config = StreamConfig()
assert config.eval_frequency == 1000  # Default value

# ✅ GOOD
config = StreamConfig()
assert config.eval_frequency > 0  # Valid range
assert hasattr(config, 'eval_frequency')  # Has field
```

### Over-Mocking
```python
# ❌ BAD
brain = Mock()
brain.regions = {'cortex': Mock()}

# ✅ GOOD
config = ThaliaConfig(...)
brain = EventDrivenBrain.from_config(config)
```

### Trivial Assertions
```python
# ❌ BAD
region.reset_state()
assert region.spikes.sum() == 0  # Trivial

# ✅ GOOD
region.reset_state()
output = region(input_spikes)
assert output.shape == (region.n_neurons,)  # Behavior
assert not torch.isnan(region.membrane).any()
```

### Testing Multiple Things
```python
# ❌ BAD
def test_region_everything():
    # Test forward pass
    # Test reset
    # Test learning
    # Test neuromodulation

# ✅ GOOD
def test_forward_pass(): ...
def test_reset(): ...
def test_learning(): ...
def test_neuromodulation(): ...
```

## Quick Red Flags

Look for these patterns - they indicate weak tests:

1. **`._variable`** - Accessing private attributes
2. **`assert ... == 1000`** - Hardcoded exact values (unless explicitly set)
3. **`Mock()` everywhere** - Over-mocking
4. **`assert x == 0`** after reset - Trivial assertions
5. **No edge case tests** - Only happy path
6. **Single test, multiple assertions** - Testing multiple things

## Approval Criteria

Only approve PR if:
- [ ] All checklist items above are satisfied
- [ ] Tests follow patterns in `WRITING_TESTS.md`
- [ ] No red flags present
- [ ] Tests have descriptive names and docstrings
- [ ] Edge cases are covered (or issue created to add them)

## Need Help?

- **Unsure if pattern is good?** Check `WRITING_TESTS.md`
- **Found weak test?** Point to `TEST_QUALITY_AUDIT_REPORT.md`
- **Need examples?** See "Excellent Tests" section in audit report

## Common Review Comments

Copy these for quick feedback:

### Internal State Access
```
❌ This test accesses private attributes (`_variable`).
Please test behavioral contracts instead.

See: WRITING_TESTS.md > Pattern 3: Internal State Access
```

### Hardcoded Config Values
```
❌ This test asserts exact default configuration values.
Please test that config has required fields and valid ranges instead.

See: WRITING_TESTS.md > Pattern 1: Hardcoded Values
```

### Over-Mocking
```
❌ This test over-uses mocks. Please use real components when possible.
Mocks should only be used for external dependencies (file I/O, plotting).

See: WRITING_TESTS.md > Pattern 4: Over-Mocking
```

### Missing Edge Cases
```
❌ This test only covers the happy path. Please add edge case tests:
- [ ] Silent input (zero spikes)
- [ ] Saturated input (all spikes)
- [ ] Invalid dimensions
- [ ] Error conditions

See: WRITING_TESTS.md > Edge Case Testing
```

### Network Integrity Missing
```
❌ This integration test creates brain structures but doesn't validate
dimensional consistency. Please add network integrity validation.

See: WRITING_TESTS.md > Network Integrity Validation
```
