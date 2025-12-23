# API Reference (Auto-Generated)

> **⚠️ Do not edit these files manually!**
> All documentation in this directory is auto-generated from the codebase.

## Contents

- **[API_INDEX.md](API_INDEX.md)** — 🆕 Comprehensive searchable index of all components
- **[COMPONENT_CATALOG.md](COMPONENT_CATALOG.md)** — All registered brain regions and pathways (with statistics)
- **[LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md)** — Learning strategy factory functions (with metrics & best practices)
- **[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)** — Configuration dataclasses (with "Used By" tracking)
- **[DATASETS_REFERENCE.md](DATASETS_REFERENCE.md)** — Dataset classes grouped by curriculum stage
- **[DIAGNOSTICS_REFERENCE.md](DIAGNOSTICS_REFERENCE.md)** — Diagnostic monitor classes
- **[EXCEPTIONS_REFERENCE.md](EXCEPTIONS_REFERENCE.md)** — Custom exception classes (with usage guidance)
- **[MODULE_EXPORTS.md](MODULE_EXPORTS.md)** — Public API exports from `__init__.py`
- **[MIXINS_REFERENCE.md](MIXINS_REFERENCE.md)** — Mixin classes providing NeuralRegion functionality
- **[CONSTANTS_REFERENCE.md](CONSTANTS_REFERENCE.md)** — Biological constants with ranges and references
- **[NEURON_FACTORIES_REFERENCE.md](NEURON_FACTORIES_REFERENCE.md)** — Pre-configured neuron populations
- **[PROTOCOLS_REFERENCE.md](PROTOCOLS_REFERENCE.md)** — Protocol/interface definitions
- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** — Code examples from docstrings
- **[CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md)** — Checkpoint file structure and format
- **[TYPE_ALIASES.md](TYPE_ALIASES.md)** — Type alias definitions
- **[COMPONENT_RELATIONSHIPS.md](COMPONENT_RELATIONSHIPS.md)** — Component connections in preset architectures
- **[ENUMERATIONS_REFERENCE.md](ENUMERATIONS_REFERENCE.md)** — All enumeration types
- **[STATE_CLASSES_REFERENCE.md](STATE_CLASSES_REFERENCE.md)** — State classes with versioning

## 🚀 Quick Start

**New to Thalia?** Start with [API_INDEX.md](API_INDEX.md) for a complete overview and search guide.

**Building a brain?** See [COMPONENT_CATALOG.md](COMPONENT_CATALOG.md)

**Implementing learning?** See [LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md)

**Training with curriculum?** See [DATASETS_REFERENCE.md](DATASETS_REFERENCE.md)

---

## Regenerating Documentation

To update these files after code changes:

```bash
python scripts/generate_api_docs.py
```

This will:
1. Scan the codebase for registered components (`@register_region`, `@register_pathway`)
2. Extract learning strategy factory functions (`create_*_strategy`)
3. Parse configuration dataclasses (`*Config`)
4. Generate fresh documentation with current timestamps

## What Gets Auto-Generated?

### Component Catalog
- Extracts from `@register_region` and `@register_pathway` decorators
- Includes: name, aliases, config class, source file, docstring
- Always matches ComponentRegistry state

### Learning Strategies API
- Extracts from `create_*_strategy` functions in `src/thalia/learning/`
- Includes: function signature, parameters, return type, docstring
- Always matches actual function signatures

### Configuration Reference
- Extracts from dataclasses ending in `Config`
- Includes: all fields with types and defaults
- Always matches actual config definitions

### Datasets Reference
- Extracts from `*Dataset` classes and `create_stage*` functions
- Includes: parameters, docstrings, source files
- Always matches available datasets for curriculum training

### Diagnostics Reference
- Extracts from `*Monitor` classes in `diagnostics/`
- Includes: key methods, docstrings, source files
- Always matches available monitoring tools

### Exceptions Reference
- Extracts from custom exception classes
- Includes: inheritance hierarchy, docstrings
- Always matches error handling API

### Module Exports Reference
- Extracts from `__all__` definitions in `__init__.py` files
- Includes: all public exports from each module
- Shows recommended import patterns

### Mixins Reference
- Extracts from mixin classes used by NeuralRegion
- Includes: NeuromodulatorMixin, GrowthMixin, ResettableMixin, DiagnosticsMixin
- Shows composition pattern and public methods

### Constants Reference
- Extracts from module-level constants (UPPERCASE names)
- Includes: biological time constants, thresholds, defaults
- **NEW**: Biological ranges (e.g., "3-7ms", "10-30ms")
- **NEW**: Scientific references from docstrings (Bi & Poo 1998, etc.)
- **NEW**: Enhanced tables with range column
- Organized by category (dopamine, acetylcholine, neuron parameters, learning rates)

### Neuron Factories Reference 🆕
- Extracts from `create_*_neurons()` factory functions
- Includes: Parameter tables, usage examples, cross-references
- Documents: pyramidal, relay, TRN, and cortical layer neurons
- Shows: Pre-configured neuron populations with biological parameters

### State Classes Reference
- Extracts from state dataclasses with STATE_VERSION
- Includes: All region and pathway state classes
- Documents: Field types, defaults, version migration patterns
- Shows: Checkpoint state structure

## Benefits

✅ **Always Synchronized** - Documentation never drifts from code
✅ **Zero Maintenance** - No manual updates needed
✅ **Complete Coverage** - Catches all registered components
✅ **Consistent Format** - Uniform structure across all entries
✅ **Cross-Referenced** - Connected documentation web with "See Also" sections ⭐
✅ **Scientifically Rigorous** - Biological ranges and citations included ⭐
✅ **Enhanced Discovery** - Neuron factories and preset configurations documented ⭐

## Integration

This auto-documentation system is referenced in:
- `docs/DOCUMENTATION_INDEX.md` - Main documentation index
- `.github/copilot-instructions.md` - AI assistant guidance
- Development workflow - Run before committing new components
