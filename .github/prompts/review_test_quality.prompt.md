---
mode: agent
---
# Test Quality Improvement Prompt

## Objective
Analyze the test suite to identify **weak tests** (implementation detail testing), categorize test quality issues, and create an actionable plan to improve test effectiveness and reduce brittleness.

## Context
- Test framework: **Vitest**
- Test location: `tests/` directory
- Previous issue: Tests failed to catch a real bug in `BranchService.ts` (message ID generation)
- Root cause: Tests validated **implementation details**, not **behavioral contracts**

## Criteria for Good vs Bad Tests

### ❌ **Bad Test Characteristics** (Brittle, Low Value)

1. **Tests Hardcoded Implementation Details**
   - Asserts exact default values: `expect(title).toBe('New Conversation')`
   - Problem: Breaking change when you update defaults
   - Example: Mocking `Date.now()` to expect exact timestamp value `3000`
   - **Fix**: Test that value exists and has correct type, not the exact value

2. **Tests Trivial Behavior**
   - Asserts things so obvious they add zero confidence
   - Example: `expect([]).toEqual([])`
   - Example: `expect(undefined).toBeUndefined()`
   - **Fix**: Delete. Don't test initialization of variables to their initialized values.

3. **Tests Only Happy Path (No Edge Cases)**
   - Never tests with empty inputs, null values, boundary conditions
   - Example: Only tests `getMessages()` with populated message arrays
   - Never tests: Empty arrays, undefined conversation, invalid message IDs
   - **Fix**: Add edge case tests (empty, null, boundary, malformed data)

4. **Tests Implementation Instead of Contract**
   - Tests how something is done, not what it does
   - Example: Asserting `querySelector('.specific-class')` found 3 elements
   - Problem: Refactoring CSS class names breaks tests (but code still works)
   - **Fix**: Test behavior (`deleteButton exists and is clickable`), not selectors

5. **No Referential Integrity or Graph Validation**
   - Related to the `BranchService.ts` bug that slipped through
   - Tests create data but don't validate consistency
   - Example: Tests create messages with `parentMessageId` refs but never verify those IDs exist
   - **Fix**: Add contract validation tests (all references resolve, no orphans, graph is valid)

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

10. **Tests Coupled to UI Implementation Details**
    - Asserts DOM structure, CSS classes, element IDs instead of user-facing behavior
    - Example: `expect(document.querySelector('.modal__overlay')).toExist()`
    - Problem: Refactoring HTML/CSS breaks tests (but UI still works)
    - **Fix**: Test behavior (modal is displayed, user can close it) not DOM structure

### ✅ **Good Test Characteristics** (Robust, High Value)

1. **Tests the Contract, Not Implementation**
   - What does this function promise? Does it deliver?
   - Asserts behavior that matters to users/calling code
   - Example: `expect(result.messages).toHaveLength(0)` (not implementation)
   - Survives refactoring, class name changes, logic optimizations

2. **Tests Edge Cases and Boundaries**
   - Empty inputs, null values, maximum sizes, boundary conditions
   - Error conditions and exception paths
   - Slow/fast variations, concurrency issues
   - Example: `createConversation()` with empty title, null tags, duplicate IDs

3. **Tests Property Invariants (Graph Consistency)**
   - Validates that data structures maintain consistency
   - Example: All `parentMessageId` references resolve to actual messages
   - Example: No circular references in message trees
   - Example: Parent messages always appear before children in arrays

4. **Tests State Transitions and Side Effects**
   - Not just final result, but how we got there
   - Example: Does update trigger subscriber notification? Does localStorage persist?
   - Example: After delete, is state cleaned up? Are listeners removed?

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
   - Stateless services are created with real implementations
   - Mocks only used for external dependencies (API calls, storage)
   - Reduces brittleness and increases confidence

8. **Tests Are Independent and Deterministic**
   - Don't depend on test execution order
   - Don't have flaky timing dependencies
   - Properly mock Date.now(), Math.random() consistently
   - Clean up after themselves (destroy, remove listeners)

## Test Quality Analysis Workflow

### Phase 1: Identify Weak Tests
1. Search for patterns of bad tests:
   - Exact default value assertions
   - Trivial assertions
   - Only happy-path tests
   - DOM selector assertions
   - Over-mocking without validation

2. Create inventory of issues by file

3. Categorize by severity:
   - **P0**: Tests that failed to catch real bugs (like BranchService)
   - **P1**: Tests coupling to implementation details (brittle)
   - **P2**: Tests missing edge cases
   - **P3**: Tests with redundant assertions

### Phase 2: Audit Specific Test Files
Priority order:
1. `tests/services/ConversationService.test.ts` - Core service, most critical
2. `tests/services/BranchService.test.ts` - Had bug that tests missed
3. `tests/utils/state.test.ts` - Core state management
4. All component tests (`tests/components/**`) - Often brittle due to DOM coupling
5. All modal tests - Often test implementation details, not user interaction

### Phase 3: Create Improvement Plan
For each test file:
1. Remove trivial/duplicate tests
2. Replace hardcoded assertions with contract assertions
3. Add edge case tests
4. Add referential integrity/invariant tests
5. Replace DOM selector assertions with behavioral assertions
6. Reduce mock depth where possible

### Phase 4: Implement Improvements
1. Start with highest priority/impact files
2. Update one test at a time, verify passing
3. Add new tests incrementally
4. Document patterns in `docs/Developer-Guides/TESTING.md`

## Search Strategy

### 1. Search for Hardcoded Assertions
```
expect\(.*(toBe|toEqual)\(\s*['"`].*['"`]\s*\)
```
Focus on:
- Default value assertions (`'New Conversation'`, `'llama2'`, etc.)
- Exact timestamp assertions (`toBe(3000)`)
- Hardcoded array assertions (`toEqual([])` on initialization)

### 2. Search for Trivial Assertions
```
expect\(.*(toBeUndefined|toBeNull|toBeDefined|toBeTruthy)\(\)
```
Remove assertions that are tautologies

### 3. Search for UI/DOM Coupling
```
querySelector|querySelectorAll|getElementById|getElementsBy|\.css|\.html|innerHTML
```
Replace with behavioral assertions

### 4. Search for Incomplete Error Testing
```
toThrow\(\)(?!\(.*(Error|Exception)\))
```
Find error tests that don't validate error type

### 5. Search for Mock Over-Use
```
vi\.mock|vi\.spyOn.*mockReturnValue|mockImplementation
```
Review if real services could be used instead

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
- Add to `docs/Developer-Guides/TESTING.md`
- Include examples of good vs bad tests
- Include patterns for:
  - Contract validation
  - Edge case testing
  - Referential integrity testing
  - State transition testing

## Success Criteria

1. ✅ All weak test patterns identified and documented
2. ✅ Improvement plan created with priority order
3. ✅ At least 20% reduction in trivial/hardcoded assertions
4. ✅ All core service tests include edge case coverage
5. ✅ All data structure tests include referential integrity checks
6. ✅ No DOM selector assertions in component tests
7. ✅ Tests remain at ~1359 count or increase (from adding edge case tests)
8. ✅ All tests continue to pass
9. ✅ Build succeeds with zero TypeScript errors

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
