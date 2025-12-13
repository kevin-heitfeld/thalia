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
   - Example: `expect([]).toEqual([])`
   - Example: `expect(undefined).toBeUndefined()`
   - **Fix**: Delete. Don't test initialization of variables to their initialized values.

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
   - Example: Both `should create conversation with defaults` AND `should create conversation with all custom parameters`
   - **Fix**: Combine or remove redundant assertions

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
   - Example: `expect(() => fn()).toThrow()` (too vague)
   - **Fix**: `expect(() => fn()).toThrow(ValidationError)` and check message

10. **Tests Coupled to Internal Implementation Details**
    - Asserts internal state variables instead of observable behavior
    - Example: `assert region._internal_buffer.shape == (100,)` (private attribute)
    - Problem: Refactoring internal implementation breaks tests (but learning still works)
    - **Fix**: Test behavior (region produces expected spikes, learns correctly) not internals

### ✅ **Good Test Characteristics** (Robust, High Value)

1. **Tests the Contract, Not Implementation**
   - What does this function promise? Does it deliver?
   - Asserts behavior that matters to users/calling code
   - Example: `expect(result.messages).toHaveLength(0)` (not implementation)
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
   - Example: `expect(() => service.deleteConversation('invalid-id')).toThrow('Conversation not found')`

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

### 1. Search for Hardcoded Assertions
```
assert [a-zA-Z0-9_.]+ == [0-9.+-eE]+
```
Replace with range checks or existence checks

### 2. Search for Trivial Assertions
```
expect\((undefined|null|true|false|\[\]|{}\))\.to(Be|Equal|DeepEqual)\(\1\)
```
Remove these tests entirely

### 3. Search for Internal State Coupling
```
\._[a-z_]+|assert.*\._|region\.state\.[a-z_]+(?!spikes|dopamine)
```
Replace with behavioral assertions (test spikes, learning, not internals)

### 4. Search for Incomplete Error Testing
```
toThrow\(\)(?!\(.*(Error|Exception)\))
```
Find error tests that don't validate error type

### 5. Search for Mock Over-Use
```
@patch|Mock\(|MagicMock\(|mocker\.
```
Review if real neural components could be used instead

### 6. Search for Missing Edge Case Tests
```
it\('should .*'?\)[\s\S]*?\}\);
```
Verify each positive test has corresponding edge case test

## Expected Output

### 1. **Comprehensive Audit Report**
- List of weak tests by file
- Categorized by issue type (hardcoded values, trivial tests, missing edge cases, etc.)
- Severity ratings
- Impact assessment

### 2. **Improvement Plan Document**
- Priority-ordered list of files to improve
- Specific recommendations for each test
- Patterns to follow going forward
- Estimated effort per file

### 3. **Test Improvement Examples**
- For each category of problem, show BEFORE/AFTER examples
- Document new patterns for edge case testing
- Document contract-based assertion patterns

### 4. **Updated Testing Guidelines**
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
6. **Provide Examples**: Show BEFORE/AFTER code for each improvement type

Use semantic search across the test suite to understand:
- What services/utils are most critical
- Which tests lack edge case coverage
- Which tests are tightly coupled to implementation
- Which tests have redundant assertions

Output should be suitable for implementation in follow-up sessions.
