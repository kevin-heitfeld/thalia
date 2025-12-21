# Auto-Documentation Opportunities

**Last Updated**: December 21, 2025

This document tracks opportunities for auto-generating documentation from code to reduce maintenance burden and ensure documentation stays synchronized.

## ✅ Currently Auto-Generated (6 documents)

### API Reference (docs/api/)

1. **COMPONENT_CATALOG.md** - Registered regions and pathways
   - Source: `@register_region`, `@register_pathway` decorators
   - Coverage: 8 regions, 4 pathways
   
2. **LEARNING_STRATEGIES_API.md** - Learning strategy factories
   - Source: `create_*_strategy()` functions
   - Coverage: 5 factory functions
   
3. **CONFIGURATION_REFERENCE.md** - Config dataclasses
   - Source: `*Config` dataclass definitions
   - Coverage: 3 config classes
   
4. **DATASETS_REFERENCE.md** - Dataset classes and factories
   - Source: `*Dataset` classes, `create_stage*()` functions
   - Coverage: 4 classes, 4 factory functions
   
5. **DIAGNOSTICS_REFERENCE.md** - Diagnostic monitors
   - Source: `*Monitor` classes
   - Coverage: 4 monitor classes
   
6. **EXCEPTIONS_REFERENCE.md** - Exception hierarchy
   - Source: Custom exception classes
   - Coverage: Exception hierarchy from `core/errors.py`

**Generation**: Run `python scripts/generate_api_docs.py`

---

## 🚀 Future Auto-Generation Opportunities

### High Value (Low Effort)

#### 1. Public API Exports Reference
**What**: Document all public exports from `__init__.py` files
**Source**: `__all__` lists and public imports in `__init__.py`
**Value**: 
- Shows what users should import from each module
- Prevents importing internal implementation details
- Catches missing exports

**Implementation**:
```python
def _find_public_exports():
    """Extract __all__ definitions from __init__.py files."""
    for init_file in self.src_dir.rglob("__init__.py"):
        # Parse __all__ = [...] assignments
        # Generate MODULE_EXPORTS.md
```

**Output**: `docs/api/MODULE_EXPORTS.md`

#### 2. Mixin Capabilities Reference
**What**: Document all mixin classes and their methods
**Source**: Classes in `thalia/core/mixins/`
**Value**:
- Shows available mixin functionality
- Documents standard NeuralRegion capabilities
- Helps when composing new regions

**Implementation**:
```python
def _find_mixins():
    """Extract mixin classes and their public methods."""
    mixins_dir = self.src_dir / "core" / "mixins"
    # Find all mixin classes
    # Extract public methods with signatures
```

**Output**: `docs/api/MIXINS_REFERENCE.md`

#### 3. Constants and Defaults Reference
**What**: Document all constants from `constants.py` files
**Source**: Module-level constants and config defaults
**Value**:
- Single source of truth for magic numbers
- Shows biological time constants, size ratios, etc.
- Helps when tuning hyperparameters

**Implementation**:
```python
def _find_constants():
    """Extract module-level constants."""
    # Find all UPPERCASE variables
    # Group by module and category
```

**Output**: `docs/api/CONSTANTS_REFERENCE.md`

### Medium Value (Medium Effort)

#### 4. Protocol/Interface Reference
**What**: Document all Protocol classes defining interfaces
**Source**: `@runtime_checkable` protocols in `protocols/`
**Value**:
- Shows required interfaces for components
- Documents expected method signatures
- Helps implement custom components

**Output**: `docs/api/PROTOCOLS_REFERENCE.md`

#### 5. CLI Commands Reference
**What**: Document all command-line scripts and arguments
**Source**: argparse definitions in scripts/
**Value**:
- Shows available training/analysis commands
- Documents script parameters
- Reduces need for manual CLI documentation

**Output**: `docs/api/CLI_REFERENCE.md`

#### 6. Checkpoint Format Documentation
**What**: Auto-generate from actual checkpoint save/load code
**Source**: `save_checkpoint()` and `load_checkpoint()` in `io/`
**Value**:
- Always matches actual checkpoint structure
- Shows required vs optional fields
- Documents version compatibility

**Output**: Update `docs/design/checkpoint_format.md` (partially auto-generate)

### Lower Priority (Higher Effort)

#### 7. Code Examples from Tests
**What**: Extract working code examples from unit tests
**Source**: Test files with clear setup/usage patterns
**Value**:
- Examples guaranteed to work (they're tested!)
- Shows actual usage patterns
- Reduces need for manual examples

**Challenge**: Need to identify "example-worthy" tests vs internal tests

#### 8. Performance Benchmarks
**What**: Auto-generate benchmark results
**Source**: `tests/benchmarks/` execution results
**Value**:
- Shows actual performance characteristics
- Tracks performance over time
- Helps identify regressions

**Challenge**: Requires running benchmarks as part of doc generation

#### 9. Architecture Diagrams from Code
**What**: Generate component relationship diagrams
**Source**: Component registry, connections, and imports
**Value**:
- Visual system overview
- Shows component dependencies
- Always matches actual architecture

**Challenge**: Requires graphviz/diagram generation library

---

## 📊 Impact Analysis

### Current Coverage
- **6 docs auto-generated** (100% synchronized with code)
- **49 docs manually maintained** (55 total - 6 auto)
- **11% automation rate**

### After High-Value Additions (+3 docs)
- **9 docs auto-generated** 
- **46 docs manually maintained**
- **16% automation rate**
- **Maintenance time saved**: ~15-20 hours/year

### Why Not Auto-Generate Everything?

Some documentation **should** remain manual:
- **Architecture guides** - Explain *why*, not just *what*
- **Design rationale** - Decision context and tradeoffs
- **Tutorials** - Pedagogical narrative flow
- **ADRs** - Historical decisions and consequences
- **Patterns** - Best practices and anti-patterns

**Principle**: Auto-generate **reference** documentation (API facts), manually maintain **explanatory** documentation (concepts, decisions, guidance).

---

## 🎯 Recommendations

### Immediate (This Session)
✅ **Done**: Enhanced to 6 auto-generated docs

### Short Term (Next Week)
1. ✅ Add **MODULE_EXPORTS.md** - High value, low effort
2. ✅ Add **MIXINS_REFERENCE.md** - Helps component developers
3. ✅ Add **CONSTANTS_REFERENCE.md** - Centralize magic numbers

### Medium Term (Next Month)
4. Add **PROTOCOLS_REFERENCE.md** - Document interfaces
5. Add **CLI_REFERENCE.md** - Document scripts
6. Auto-update **checkpoint_format.md** sections

### Long Term (Next Quarter)
7. Extract examples from tests
8. Generate architecture diagrams
9. Track performance benchmarks

---

## 📝 Implementation Notes

### Adding a New Auto-Generated Doc

1. **Add dataclass** to `scripts/generate_api_docs.py`:
   ```python
   @dataclass
   class NewInfo:
       """Description of what this captures."""
       name: str
       # ... other fields
   ```

2. **Add extraction method**:
   ```python
   def _find_new_items(self):
       """Extract data from codebase."""
       for py_file in relevant_files:
           # Parse and extract
           self.new_items.append(NewInfo(...))
   ```

3. **Add generation method**:
   ```python
   def _generate_new_reference(self):
       """Generate NEW_REFERENCE.md."""
       output_file = self.api_dir / "NEW_REFERENCE.md"
       # Write markdown
   ```

4. **Update navigation**:
   - Add to `docs/api/README.md`
   - Add to `docs/README.md`
   - Add to `docs/DOCUMENTATION_INDEX.md`

5. **Run generator**: `python scripts/generate_api_docs.py`

### Best Practices

- ✅ Include "Auto-generated - Do not edit!" warning
- ✅ Add timestamp for freshness indication
- ✅ Link to source files for deep dives
- ✅ Keep format consistent across all auto-docs
- ✅ Validate in `scripts/validate_docs.py`
- ✅ Run generator in CI/CD before releases

---

## 🔄 Maintenance Schedule

**Pre-commit** (Optional):
- Run generator if component files changed
- Use git pre-commit hook

**Pre-PR** (Recommended):
- Run `python scripts/generate_api_docs.py`
- Commit updated API docs with code changes

**Pre-release** (Required):
- Run generator to ensure docs match release code
- Validate all links and references

**Continuous**:
- Generator runs in CI to detect drift
- Fails build if auto-docs are stale

---

## ✨ Success Metrics

Track these to measure auto-documentation value:

- **Coverage**: % of API surface documented automatically
- **Drift incidents**: # of times manual docs were wrong
- **Maintenance time**: Hours saved per month
- **Discoverability**: Time to find API information
- **Freshness**: Max age of stale documentation

**Target**: 20% automation rate, <1 drift incident/month, 20+ hours saved/year
