---
agent: agent
---
# Test Quality Improvement Prompt

You are an expert software engineer specializing in test quality and reliability for complex neural simulation frameworks. Your task is to audit the existing test suite, identify weak tests, and create a comprehensive improvement plan to enhance test effectiveness, focusing on biological plausibility and learning correctness rather than implementation details.

## Objective
Analyze the test suite to identify **weak tests** (implementation detail testing), categorize test quality issues, and create an actionable plan to improve test effectiveness and reduce brittleness.

## Context
- Test framework: **pytest**
- Test location: `tests/` directory
- Focus: Biological plausibility, learning rule correctness, pathway routing, state management
- Goal: Tests should validate **neural behavior and learning contracts**, not **implementation details**

## Criteria for Good vs Bad Tests

### ❌ **Bad Test Characteristics** (Brittle, Low Value)

1. **Tests Hardcoded Implementation Details**
   - Asserts exact default values: `assert region.n_neurons == 1000`
   - Problem: Breaking change when you update neurogenesis or config defaults
   - Example: Hardcoding exact spike counts instead of testing spike rate ranges
   - **Fix**: Test that value exists and is within biologically plausible range, not exact value

2. **Tests Trivial Behavior**
   - Asserts things so obvious they add zero confidence

3. **Tests Only Happy Path (No Edge Cases)**
   - Never tests with empty inputs, null values, boundary conditions
   - Example: Only tests `forward()` with normal spike patterns
   - Never tests: Silent neurons (0 spikes), saturated neurons (all spikes), invalid dimensions
   - **Fix**: Add edge case tests (empty spikes, dimension mismatches, extreme neuromodulator values)

4. **Tests Implementation Instead of Contract**
   - Tests how something is done, not what it does
   - Example: Asserting exact weight matrix values after initialization
   - Problem: Refactoring weight initialization method breaks tests (but learning still works)
   - **Fix**: Test behavior (weights in valid range, correct shape, sparsity), not exact values

5. **No Network Integrity or Connectivity Validation**
   - Tests create pathways but don't validate connectivity
   - Example: Tests create pathways between regions but never verify dimensions match
   - Example: Tests don't validate that pathway input_size == source region output_size
   - **Fix**: Add contract validation tests (dimensions compatible, no disconnected components)

6. **Tests That Are Duplicates or Highly Redundant**
   - Multiple tests asserting the same contract
   - Example: `test_forward_visual()`, `test_forward_auditory()`, `test_forward_language()` - identical logic, different inputs
   - Example: `test_grow_neurons_50()`, `test_grow_neurons_100()`, `test_grow_neurons_200()` - same test, different sizes
   - **Fix**: Consolidate into parameterized tests using `@pytest.mark.parametrize`

7. **Tests That Could Be Parameterized But Aren't**
   - Repetitive test code with only input values changing
   - Example: Separate tests for different learning rates, delay values, or growth amounts
   - Problem: Code duplication makes maintenance harder, adds to test count without adding coverage
   - **Fix**: Use `@pytest.mark.parametrize` to test multiple cases with single test function
   - Pattern:
     ```python
     @pytest.mark.parametrize("learning_rate,expected_change", [
         (0.001, "small"), (0.01, "medium"), (0.1, "large")
     ])
     def test_learning_with_various_rates(learning_rate, expected_change):
         # Single test function handles all cases
     ```

8. **Mock-Heavy Tests That Don't Test Real Behavior**
   - Over-mocked to the point where test doesn't validate actual code path
   - Example: Mocking return values that match code exactly (not testing error handling)
   - **Fix**: Use real services when possible (stateless services), mock only external dependencies

9. **No Validation of Intermediate States**
   - Tests only check final result, not state during operations
   - Example: Batch operations should validate intermediate states during processing
   - **Fix**: Add assertions for state transitions, partial completion, rollback scenarios

10. **Weak Assertions on Error Conditions**
   - Tests that error occurred but don't validate error message or type

11. **Tests Coupled to Internal Implementation Details**
    - Asserts internal state variables instead of observable behavior
    - Example: `assert region._internal_buffer.shape == (100,)` (private attribute)
    - Problem: Refactoring internal implementation breaks tests (but learning still works)
    - **Fix**: Test behavior (region produces expected spikes, learns correctly) not internals

### ✅ **Good Test Characteristics** (Robust, High Value)

1. **Tests the Contract, Not Implementation**
   - What does this function promise? Does it deliver?
   - Asserts behavior that matters to users/calling code
   - Survives refactoring, class name changes, logic optimizations

2. **Tests Edge Cases and Boundaries**
   - Empty inputs, null values, maximum sizes, boundary conditions
   - Error conditions and exception paths
   - Silent neurons, saturated neurons, extreme parameter values
   - Example: `region.forward()` with all-zero spikes, invalid dopamine levels, mismatched dimensions

3. **Tests Property Invariants (Network Consistency)**
   - Validates that neural architectures maintain consistency
   - Example: All pathway connections have compatible dimensions
   - Example: Weights stay within bounds after learning
   - Example: Neuron states are valid (no NaN, no negative firing rates)

4. **Tests State Transitions and Side Effects**
   - Not just final result, but how we got there
   - Example: Does learning update weights? Does neuromodulator affect plasticity?
   - Example: After reset, is state properly initialized? Are traces cleared?

5. **Tests Meaningful Assertions Only**
   - Every assertion either:
     - Validates a requirement
     - Prevents a regression
     - Tests a boundary condition
   - Trivial assertions are removed

6. **Tests Both Success and Failure Paths**
   - Happy path AND error conditions
   - Validates error messages are descriptive

7. **Uses Real Objects When Possible**
   - Neural components are created with real implementations
   - Mocks only used for external dependencies (data loaders, visualization)
   - Reduces brittleness and increases confidence in biological accuracy

8. **Tests Are Independent and Deterministic**
   - Don't depend on test execution order
   - Don't have flaky random seed dependencies
   - Properly seed torch.manual_seed(), np.random.seed() consistently
   - Clean up after themselves (reset states, clear buffers)

## Test Quality Analysis Workflow

### Phase 1: Identify Weak Tests
1. Search for patterns of bad tests:
   - Exact default value assertions
   - Trivial assertions
   - Only happy-path tests
   - Internal state assertions (private attributes)
   - Over-mocking without validation
   - Redundant/duplicate tests
   - Tests that could be parameterized

2. Create inventory of issues by file

3. Categorize by severity:
   - **P0**: Tests that failed to catch real bugs (biological implausibility, dimension mismatches)
   - **P1**: Tests coupling to implementation details (brittle)
   - **P2**: Tests missing edge cases (silent/saturated neurons, extreme parameters)
   - **P3**: Tests with redundant assertions or duplicate test logic

### Phase 2: Coverage Analysis
1. Map test files to core components (regions, pathways, services)
2. Identify critical components with weak test coverage
3. Prioritize components by impact on system reliability and biological accuracy

### Phase 3: Create Improvement Plan
For each test file:
1. **Identify and remove redundant tests** (exact duplicates or tests covered by other tests)
2. **Consolidate repetitive tests into parameterized tests** (same logic, different inputs)
3. Replace hardcoded assertions with contract assertions
4. Add edge case tests (silent/saturated neurons, extreme parameters)
5. Add network integrity/invariant tests (dimension compatibility, weight bounds)
6. Replace internal state assertions with behavioral assertions (spike patterns, learning)
7. Reduce mock depth where possible

### Phase 4: Implement Improvements
1. Start with highest priority/impact files
2. Update one test at a time, verify passing
3. Add new tests incrementally
4. Document patterns in `tests/WRITING_TESTS.md`

## Search Strategy

- Search for Hardcoded Assertions: Replace with range checks or existence checks
- Search for Trivial Assertions: Remove these tests entirely
- Search for Internal State Coupling: Replace with behavioral assertions (test spikes, learning, not internals)
- Search for Incomplete Error Testing: Find error tests that don't validate error type
- Search for Mock Over-Use: Review if real neural components could be used instead
- Search for Missing Edge Case Tests: Verify each positive test has corresponding edge case test
- **Search for Redundant Tests**: Look for duplicate tests or tests with identical logic and different inputs
  - Pattern: `test_X_with_Y`, `test_X_with_Z` - same function, different parameter
  - Pattern: `test_X_size_10`, `test_X_size_20`, `test_X_size_50` - same test, different size
  - Pattern: `test_forward_visual`, `test_forward_auditory` - same logic, different modality
- **Search for Parameterization Opportunities**: Look for repetitive test patterns
  - Multiple tests with same structure but different input values
  - Tests that only differ in one or two parameters
  - Consolidate using `@pytest.mark.parametrize` decorator

## Expected Output

### 1. **Comprehensive Audit Report**
- List of weak tests by file
- Categorized by issue type (hardcoded values, trivial tests, missing edge cases, etc.)
- Severity ratings
- Impact assessment
- Priority-ordered list of files to improve
- Specific recommendations for each test
- Patterns to follow going forward

### 2. **Updated Testing Guidelines**
- Add to `tests/WRITING_TESTS.md`
- Include examples of good vs bad tests
- Include patterns for:
  - Contract validation (learning rules, spike generation)
  - Edge case testing (silent/saturated neurons, extreme parameters)
  - Network integrity testing (dimension compatibility, weight bounds)
  - State transition testing (reset, learning, neuromodulation)
  - **Parameterized testing** (consolidating repetitive tests)
  - **Test consolidation patterns** (identifying and removing redundancy)

## Success Criteria

1. ✅ All weak test patterns identified and documented
2. ✅ Improvement plan created with priority order
3. ✅ At least 20% reduction in trivial/hardcoded assertions
4. ✅ **Redundant tests identified and consolidated/removed**
5. ✅ **Parameterization opportunities identified and implemented where beneficial**
6. ✅ All core region/pathway tests include edge case coverage
7. ✅ All network architecture tests include connectivity validation
8. ✅ No internal state assertions (testing private attributes)
9. ✅ Tests maintain current coverage or increase (from adding edge case tests)
10. ✅ All tests continue to pass
11. ✅ Type checking (Pyright) succeeds with no critical errors

## Instructions to AI

1. **Audit Phase**: Search the test suite for patterns of weak tests
2. **Categorize**: Group issues by type and severity
3. **Document**: Create comprehensive audit report
4. **Recommend**: Propose specific improvements with examples
5. **Prioritize**: Rank improvements by impact and effort
6. **Identify Redundancy**: Find duplicate tests and parameterization opportunities

Use semantic search across the test suite to understand:
- Which tests lack edge case coverage
- Which tests are tightly coupled to implementation
- Which tests have redundant assertions
- Which tests are duplicates or near-duplicates
- Which test patterns repeat across multiple test functions
- Which tests could benefit from parameterization

Output should be suitable for implementation in follow-up sessions.
