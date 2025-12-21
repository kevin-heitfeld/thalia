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

7. **Mock-Heavy Tests That Don't Test Real Behavior**
   - Over-mocked to the point where test doesn't validate actual code path
   - Example: Mocking return values that match code exactly (not testing error handling)
   - **Fix**: Use real services when possible (stateless services), mock only external dependencies

8. **No Validation of Intermediate States**
   - Tests only check final result, not state during operations
   - Example: Batch operations should validate intermediate states during processing
   - **Fix**: Add assertions for state transitions, partial completion, rollback scenarios

9. **Weak Assertions on Error Conditions**
   - Tests that error occurred but don't validate error message or type

10. **Tests Coupled to Internal Implementation Details**
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

2. Create inventory of issues by file

3. Categorize by severity:
   - **P0**: Tests that failed to catch real bugs (biological implausibility, dimension mismatches)
   - **P1**: Tests coupling to implementation details (brittle)
   - **P2**: Tests missing edge cases (silent/saturated neurons, extreme parameters)
   - **P3**: Tests with redundant assertions

### Phase 2: Coverage Analysis
1. Map test files to core components (regions, pathways, services)
2. Identify critical components with weak test coverage
3. Prioritize components by impact on system reliability and biological accuracy

### Phase 3: Create Improvement Plan
For each test file:
1. Remove trivial/duplicate tests
2. Replace hardcoded assertions with contract assertions
3. Add edge case tests (silent/saturated neurons, extreme parameters)
4. Add network integrity/invariant tests (dimension compatibility, weight bounds)
5. Replace internal state assertions with behavioral assertions (spike patterns, learning)
6. Reduce mock depth where possible

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

## Success Criteria

1. ✅ All weak test patterns identified and documented
2. ✅ Improvement plan created with priority order
3. ✅ At least 20% reduction in trivial/hardcoded assertions
4. ✅ All core region/pathway tests include edge case coverage
5. ✅ All network architecture tests include connectivity validation
6. ✅ No internal state assertions (testing private attributes)
7. ✅ Tests maintain current coverage or increase (from adding edge case tests)
8. ✅ All tests continue to pass
9. ✅ Type checking (Pyright) succeeds with no critical errors

## Instructions to AI

1. **Audit Phase**: Search the test suite for patterns of weak tests
2. **Categorize**: Group issues by type and severity
3. **Document**: Create comprehensive audit report
4. **Recommend**: Propose specific improvements with examples
5. **Prioritize**: Rank improvements by impact and effort

Use semantic search across the test suite to understand:
- Which tests lack edge case coverage
- Which tests are tightly coupled to implementation
- Which tests have redundant assertions

Output should be suitable for implementation in follow-up sessions.
