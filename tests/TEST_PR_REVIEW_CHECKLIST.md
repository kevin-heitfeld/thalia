# Test Quality PR Review Checklist

Use this checklist when reviewing test code to ensure high-quality, maintainable tests.

## ✅ Core Quality Checks

### 1. No Internal State Access
- [ ] Tests don't access private attributes (no `._variable` access)
- [ ] Tests don't call private methods (no `._method()` calls)
- [ ] Tests use public API (getters, public methods)

**Why:** Private attributes/methods are implementation details that can change without breaking the component's contract.

**Example violation:**
```python
assert striatum._d1_delay_steps == 15  # ❌ Private attribute
```

**Correct approach:**
```python
d1_votes, d2_votes = striatum.get_accumulated_votes()  # ✅ Public API
```

---

### 2. No Hardcoded Configuration Values
- [ ] Tests don't assert exact default config values
- [ ] Tests use explicit config values in test setup
- [ ] Tests validate config contracts, not specific defaults

**Why:** Default values can change. Tests should validate the config system works, not lock in specific values.

**Example violation:**
```python
config = StreamConfig()
assert config.eval_frequency == 1000  # ❌ Hardcoded default
```

**Correct approach:**
```python
config = StreamConfig(eval_frequency=500)  # ✅ Explicit value
assert config.eval_frequency == 500
assert config.eval_frequency > 0  # ✅ Contract validation
```

---

### 3. Behavioral Contracts
- [ ] Tests verify observable behavior, not implementation
- [ ] Tests validate outputs, side effects, or state changes
- [ ] Tests don't check how something is done, only what is done

**Why:** Implementation can change. Behavior should not.

**Example violation:**
```python
assert striatum._delay_ptr == 42  # ❌ Implementation detail
```

**Correct approach:**
```python
# Run 1000 steps, verify no crashes/NaN
for _ in range(1000):
    output = striatum(input_spikes)
    assert not torch.isnan(output).any()  # ✅ Behavioral contract
```

---

## 🧪 Test Coverage Checks

### 4. Edge Cases
- [ ] Tests include silent input (all zeros)
- [ ] Tests include saturated input (all ones)
- [ ] Tests include dimension mismatches (wrong input size)
- [ ] Tests include invalid parameters (negative, NaN, Inf)

**Why:** Edge cases catch bugs that normal inputs don't reveal.

---

### 5. Error Conditions
- [ ] Tests verify error handling (raises appropriate exceptions)
- [ ] Tests validate error messages are helpful
- [ ] Tests don't just test the happy path

**Example:**
```python
def test_load_corrupted_checkpoint():
    """Test loading corrupted checkpoint raises error."""
    region = Cortex(config)

    with pytest.raises(CheckpointError, match="Invalid format"):
        region.load_checkpoint({"wrong": "structure"})
```

---

## 🏗️ Test Structure Checks

### 6. Minimal Mocking
- [ ] Tests use real components when possible
- [ ] Mocks are only for external dependencies (file system, network, plotting)
- [ ] Mocks don't replace core brain components

**Why:** Over-mocking reduces confidence that real code works.

**Example violation:**
```python
brain = Mock()  # ❌ Mock entire brain
brain.regions = {'cortex': Mock()}  # ❌ Mock regions
```

**Correct approach:**
```python
# ✅ Use real brain
config = ThaliaConfig(...)
brain = EventDrivenBrain.from_thalia_config(config)
```

---

### 7. Network Integrity
- [ ] Integration tests validate pathway dimensions match regions
- [ ] Tests verify no NaN/Inf in weights or membrane potentials
- [ ] Tests check all regions are connected (no orphans)

**Example:**
```python
def test_pathway_dimensions_match_regions(brain):
    """Test all pathways have correct dimensions."""
    for name, pathway in brain.pathway_manager.pathways.items():
        source_region = brain.regions[pathway.source_name]
        target_region = brain.regions[pathway.target_name]

        assert pathway.input_size == source_region.n_neurons
        assert pathway.output_size == target_region.n_neurons
```

---

### 8. Test Organization
- [ ] Test name clearly describes what is tested
- [ ] Test docstring explains the behavioral contract
- [ ] One test validates one behavior (single responsibility)
- [ ] Tests use fixtures to reduce duplication

**Example:**
```python
def test_d1_pathway_arrives_before_d2():
    """Test D1 direct pathway is faster than D2 indirect pathway.

    BEHAVIORAL CONTRACT: D1 votes should appear before D2 votes
    because the direct pathway has shorter axonal delays.
    """
    # Test implementation...
```

---

## 🔍 Code Review Questions

When reviewing test PRs, ask:

1. **"If the implementation changes but behavior stays the same, will this test still pass?"**
   - If no → test is too coupled to implementation

2. **"Does this test validate a contract or an implementation detail?"**
   - Contract: ✅ Good
   - Implementation: ❌ Refactor

3. **"What happens if we run this test with extreme inputs?"**
   - If it crashes → add edge case tests

4. **"Are we using real components or mocks?"**
   - Real: ✅ Better confidence
   - Mocks: ⚠️ Only if necessary

5. **"Does this test have a clear failure message?"**
   - Good: "D1 votes should appear before D2 votes (D1 at t=10, D2 at t=5)"
   - Bad: "AssertionError: False is not True"

---

## 📋 Quick Reference

### ✅ Good Test Patterns
- Public API access: `region.get_state()`
- Explicit configs: `Config(learning_rate=0.01)`
- Behavioral validation: "Does it spike correctly?"
- Edge cases: silent, saturated, invalid inputs
- Real components: `EventDrivenBrain.from_config()`
- Clear assertions: `assert d1_time < d2_time, "D1 should arrive first"`

### ❌ Bad Test Patterns
- Private access: `region._internal_state`
- Default assertions: `assert config.value == 1000`
- Implementation checks: `assert buffer._ptr == 42`
- Missing edge cases: only happy path
- Over-mocking: `Mock()` for brain components
- Vague assertions: `assert x`

---

## 📚 References

- **Test Quality Audit:** `tests/TEST_QUALITY_AUDIT_REPORT.md`
- **Writing Tests Guide:** `tests/WRITING_TESTS.md`
- **Refactoring Patterns:** See WRITING_TESTS.md § Refactoring Patterns

---

**Version:** 1.0 (December 13, 2025)
