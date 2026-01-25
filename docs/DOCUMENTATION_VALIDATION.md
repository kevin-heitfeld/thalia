# Automated Documentation Validation

**Status**: ✅ Active
**Last Updated**: December 21, 2025

## Overview

Thalia uses automated validation to ensure documentation stays synchronized with the codebase. The validation system catches common issues like outdated function names, incorrect imports, and references to removed classes.

## How It Works

### Validation Script

**Location**: `scripts/validate_docs.py`

The validation script:
1. **Indexes the codebase** - Scans all Python files to build a registry of:
   - Function definitions
   - Class definitions
   - Importable modules

2. **Extracts code blocks** - Parses all markdown files and extracts Python code examples

3. **Validates examples** - Checks each code block for:
   - Outdated patterns (e.g., `create_learning_strategy` → `create_strategy`)
   - Removed classes (e.g., `SimpleLIF`)
   - Incorrect attribute access (e.g., `brain.cortex` → `brain.components["cortex"]`)
   - Invalid imports using **AST parsing** (handles multi-line imports correctly)
   - Non-existent functions

4. **Reports issues** - Provides clear error messages with file locations

**Key Features:**
- ✅ **AST-based import validation** - Properly handles multi-line and parenthesized imports
- ✅ **Low false-positive rate** - Smart detection avoids warnings on `__init__.py` exports
- ✅ **Fast execution** - Completes in seconds even for large codebases

### What Gets Validated

**Validated Files:**
- ✅ All `docs/**/*.md` files
- ✅ `.github/copilot-instructions.md`
- ✅ Root `README.md`

**Validation Checks:**
- ❌ Outdated function names
- ❌ References to removed classes
- ❌ Incorrect component access patterns
- ❌ Invalid import statements
- ⚠️ Potentially missing modules/functions

## Running Validation

### Manual Validation

```bash
# From repository root
python scripts/validate_docs.py
```

**Output:**
```
================================================================================
Thalia Documentation Validation
================================================================================
Repository: /path/to/thalia

Indexing codebase...
  Found 342 functions
  Found 89 classes
  Found 156 modules

Validating documentation...

✅ All documentation validation checks passed!

================================================================================
```

### Pre-commit Hook (Automatic)

Install pre-commit hooks to validate automatically before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

Now validation runs automatically when you commit changes to `.md` or `.py` files.

### CI/CD Integration (GitHub Actions)

**Location**: `.github/workflows/validate-docs.yml`

Validation runs automatically on:
- Every push to docs or source code
- Every pull request affecting docs or source code

**Workflow:**
1. Checkout code
2. Set up Python environment
3. Run validation script
4. Comment on PR if validation fails

## Common Validation Errors

### Error: Uses outdated function name

```
docs/patterns/learning-strategies.md (block 2): Uses outdated 'create_learning_strategy',
should be 'create_strategy'
```

**Fix:** Update the function name in the code example.

### Error: References removed class

```
docs/design/neuron_models.md (block 1): Uses outdated 'SimpleLIF',
should be 'ConductanceLIF'
```

**Fix:** Replace references to removed classes with current equivalents.

### Error: Incorrect attribute access

```
docs/examples/quickstart.md (block 3): Uses outdated 'brain.cortex',
should be 'brain.components["cortex"]'
```

**Fix:** Use the component dictionary access pattern.

## Validation Rules

### Outdated Patterns Detected

| Old Pattern | New Pattern | Severity |
|-------------|-------------|----------|
| `create_learning_strategy()` | `create_strategy()` | Error |
| `SimpleLIF` | `ConductanceLIF` | Error |
| `brain.cortex` | `brain.components["cortex"]` | Error |
| `brain.hippocampus` | `brain.components["hippocampus"]` | Error |
| `brain.striatum` | `brain.components["striatum"]` | Error |
| `brain.pfc` | `brain.components["pfc"]` | Error |

### Import Validation

The validator checks:
- ✅ Module exists in `src/thalia/`
- ✅ Imported names (functions/classes) exist
- ⚠️ Warns if module path looks suspicious

## Extending Validation

### Adding New Patterns

Edit `scripts/validate_docs.py` and add to `outdated_patterns`:

```python
outdated_patterns = [
    # Existing patterns...
    (r"your_pattern", "old_name", "new_name"),
]
```

### Adding Custom Checks

Extend the `DocumentationValidator` class:

```python
class DocumentationValidator:
    def _validate_custom_check(self, code: str, ...) -> None:
        """Your custom validation logic."""
        if some_condition:
            result.add_error("Your error message")
```

## Best Practices

### For Documentation Writers

1. **Run validation before committing**
   ```bash
   python scripts/validate_docs.py
   ```

2. **Keep examples simple and testable**
   - Use actual API patterns from the codebase
   - Avoid pseudo-code that might fail validation

3. **Test code examples**
   - Copy examples into a Python file and run them
   - Ensure imports work and functions exist

### For Code Changes

When changing APIs:

1. **Update documentation first** (or simultaneously)
2. **Run validation** to find affected docs
3. **Update all affected examples**
4. **Verify validation passes** before merging

## Maintenance

### Regular Updates

- **Monthly**: Review validation rules for new patterns
- **Quarterly**: Check for false positives and refine patterns
- **After major refactors**: Update validation rules immediately

### Troubleshooting

**Validation fails but code is correct:**
- Check if it's a false positive
- Add exception to validation script
- Document the exception in this file

**Too many warnings:**
- Warnings are informational, not blocking
- Review and fix if time permits
- Suppress false-positive warnings

## Metrics

Track documentation health over time:

```bash
# Count validation errors
python scripts/validate_docs.py 2>&1 | grep "ERRORS:" -A 100 | wc -l

# Count warnings
python scripts/validate_docs.py 2>&1 | grep "WARNINGS:" -A 100 | wc -l
```

**Current Status (Dec 21, 2025):**
- ✅ 0 errors
- ✅ 0 warnings (improved AST parsing eliminated false positives!)
- 📊 Health Score: 100/100

**Recent Improvements:**
- **Dec 21, 2025**: Replaced regex-based import validation with AST parsing
  - Correctly handles multi-line imports
  - Handles parenthesized imports: `from thalia.learning import (strategy1, strategy2)`
  - Reduces false positives by 76 warnings → 0 warnings
  - Zero impact on actual error detection

## Future Enhancements

Planned improvements:

1. ~~**Multi-line import handling**~~ ✅ **DONE (Dec 21, 2025)** - Using AST parsing
2. **Link checking** - Validate internal documentation links
3. **API signature validation** - Check function signatures match
4. **Example execution** - Run code examples in sandbox
5. **Coverage metrics** - Track which APIs are documented
6. **Type hint validation** - Verify type hints in examples match actual code

## References

- Validation script: `scripts/validate_docs.py`
- Pre-commit config: `.pre-commit-config.yaml`
- GitHub workflow: `.github/workflows/validate-docs.yml`
- Update summary: `docs/DOCUMENTATION_UPDATE_2025-12-21.md`
