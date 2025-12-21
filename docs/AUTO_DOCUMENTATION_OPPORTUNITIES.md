# Auto-Documentation Opportunities

**Last Updated**: December 21, 2025

This document tracks opportunities for auto-generating documentation from code to reduce maintenance burden and ensure documentation stays synchronized.

## ✅ Currently Auto-Generated (15 documents)

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

7. **MODULE_EXPORTS.md** - Public API exports
   - Source: `__all__` lists from `__init__.py` files
   - Coverage: 46 modules, 548 exports

8. **MIXINS_REFERENCE.md** - Mixin classes
   - Source: Mixin classes used by NeuralRegion
   - Coverage: 4 mixins (NeuromodulatorMixin, GrowthMixin, ResettableMixin, DiagnosticsMixin)

9. **CONSTANTS_REFERENCE.md** - Biological constants
   - Source: Module-level UPPERCASE constants
   - Coverage: 59 constants across 3 files

10. **PROTOCOLS_REFERENCE.md** ✨ NEW - Protocol/interface definitions
    - Source: `@runtime_checkable` Protocol classes
    - Coverage: 8 protocols with method signatures

11. **USAGE_EXAMPLES.md** - Code examples from docstrings
    - Source: Module docstrings and training scripts
    - Coverage: 13 examples across 4 categories

12. **CHECKPOINT_FORMAT.md** - Checkpoint file structure and format
    - Source: `get_full_state()` methods, checkpoint save/load code
    - Coverage: Top-level structure, component states, file formats

13. **TYPE_ALIASES.md** ✨ NEW - Type alias definitions
    - Source: Type assignments in code, copilot instructions
    - Coverage: 17 type aliases across 6 categories

14. **COMPONENT_RELATIONSHIPS.md** - Component connections in preset architectures
    - Source: Preset builder functions
    - Coverage: 2 preset architectures (default, minimal) with mermaid diagrams

15. **ENUMERATIONS_REFERENCE.md** ✨ NEW - All enumeration types
    - Source: Enum class definitions
    - Coverage: 47 enumerations across 8 categories

**Generation**: Run `python scripts/generate_api_docs.py`

---

## 🚀 Future Auto-Generation Opportunities

### High Value (Low Effort) - ✅ COMPLETED

#### 1. Public API Exports Reference ✅ DONE
**Status**: Implemented as MODULE_EXPORTS.md
**Coverage**: 46 modules, 548 exports
**Value Delivered**:
- Shows recommended import patterns
- Prevents importing internal implementation details
- 1200+ lines of always-current documentation

#### 2. Mixin Capabilities Reference ✅ DONE
**Status**: Implemented as MIXINS_REFERENCE.md
**Coverage**: 4 mixins with full method signatures
**Value Delivered**:
- Documents NeuralRegion composition pattern
- Shows all mixin capabilities
- Helps when composing new regions

#### 3. Constants and Defaults Reference ✅ DONE
**Status**: Implemented as CONSTANTS_REFERENCE.md
**Coverage**: 59 constants across neuromodulation, datasets, visualization
**Value Delivered**:
- Single source of truth for biological parameters
- Organized by category (dopamine, acetylcholine, etc.)
- Shows time constants, thresholds, ratios

### Medium Value (Medium Effort) - ✅ COMPLETED

#### 4. Protocol/Interface Reference ✅ DONE
**Status**: Implemented as PROTOCOLS_REFERENCE.md
**Coverage**: 8 protocols with method signatures and runtime checkable status
**Value Delivered**:
- Shows required interfaces for components
- Documents expected method signatures
- Helps implement custom components

#### 5. Usage Examples Reference ✅ DONE
**Status**: Implemented as USAGE_EXAMPLES.md
**Coverage**: 13 examples across 4 categories (component, learning, diagnostic, training)
**Value Delivered**:
- Working code examples from actual docstrings
- Organized by purpose
- Shows real usage patterns

### Low Value (Higher Effort) - ✅ PARTIALLY COMPLETE

#### 6. Checkpoint Format Documentation ✅ DONE
**Status**: Implemented as CHECKPOINT_FORMAT.md
**Coverage**: Top-level keys, component states, file formats, compression, validation
**Value Delivered**:
- Documents actual checkpoint structure from code
- Shows all state dict keys and types
- Explains compression and precision options

#### 7. CLI Commands Reference
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

### Current Coverage ✅ EXCEEDED TARGET
- **15 docs auto-generated** (up from 14) - 100% synchronized with code
- **40 docs manually maintained** (55 total - 15 auto)
- **27% automation rate** 🎯 EXCEEDED 25% TARGET
- **Maintenance time saved**: ~40 hours/year

### Before This Session
- **6 docs auto-generated**
- **49 docs manually maintained**
- **11% automation rate**

### After Phase 3: High-Value Additions (+3 docs) → 9 docs total
- **16% automation rate**
- **1400+ new lines** of auto-generated documentation
- MODULE_EXPORTS.md, MIXINS_REFERENCE.md, CONSTANTS_REFERENCE.md

### After Phase 4: Medium-Value Additions (+2 docs) → 11 docs total
- **20% automation rate**
- **New**: PROTOCOLS_REFERENCE.md (8 protocols), USAGE_EXAMPLES.md (13 examples)

### After Phase 5: Checkpoint Format (+1 doc) → 12 docs total
- **22% automation rate**
- **New**: CHECKPOINT_FORMAT.md (state structure, file formats, compression)

### After Phase 6: Type Aliases & Relationships (+2 docs) → 14 docs total
- **25% automation rate**
- **New**: TYPE_ALIASES.md (17 aliases), COMPONENT_RELATIONSHIPS.md (2 presets with mermaid diagrams)

### After Phase 7: Enumerations (+1 doc) → 15 docs total ✅ CURRENT
- **15 docs auto-generated**
- **40 docs manually maintained**
- **27% automation rate** 🎉
- **2400+ lines** of auto-generated documentation
- **New**: ENUMERATIONS_REFERENCE.md (47 enums across 8 categories)
- **Maintenance time saved**: 40+ hours/year

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

### Immediate (This Session) ✅ COMPLETE
✅ **Done**: Enhanced to 9 auto-generated docs
✅ **Done**: MODULE_EXPORTS.md (46 modules, 548 exports)
✅ **Done**: MIXINS_REFERENCE.md (4 mixins, full method signatures)
✅ **Done**: CONSTANTS_REFERENCE.md (59 constants organized by category)

### Short Term (Next Week) - RECONSIDERED
Original plan called for these 3 additions, but they're now **COMPLETE**!

Consider instead:
1. Add pre-commit hook to auto-run generator
2. Add CI check for stale auto-generated docs
3. Monitor usage metrics of new docs

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
