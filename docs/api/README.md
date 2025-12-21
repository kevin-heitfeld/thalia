# API Reference (Auto-Generated)

> **⚠️ Do not edit these files manually!**
> All documentation in this directory is auto-generated from the codebase.

## Contents

- **[COMPONENT_CATALOG.md](COMPONENT_CATALOG.md)** — All registered brain regions and pathways
- **[LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md)** — Learning strategy factory functions
- **[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)** — Configuration dataclasses
- **[DATASETS_REFERENCE.md](DATASETS_REFERENCE.md)** — Dataset classes and factory functions
- **[DIAGNOSTICS_REFERENCE.md](DIAGNOSTICS_REFERENCE.md)** — Diagnostic monitor classes
- **[EXCEPTIONS_REFERENCE.md](EXCEPTIONS_REFERENCE.md)** — Custom exception classes
- **[MODULE_EXPORTS.md](MODULE_EXPORTS.md)** — Public API exports from `__init__.py`
- **[MIXINS_REFERENCE.md](MIXINS_REFERENCE.md)** — Mixin classes providing NeuralRegion functionality
- **[CONSTANTS_REFERENCE.md](CONSTANTS_REFERENCE.md)** — All package-level constants *(auto-generated)*
- **[PROTOCOLS_REFERENCE.md](PROTOCOLS_REFERENCE.md)** — Protocol/interface definitions *(auto-generated)*
- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** — Code examples from docstrings *(auto-generated)*
- **[CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md)** — Checkpoint file structure and format *(auto-generated)*
- **[TYPE_ALIASES.md](TYPE_ALIASES.md)** — Type alias definitions *(auto-generated)*
- **[COMPONENT_RELATIONSHIPS.md](COMPONENT_RELATIONSHIPS.md)** — Component connections in preset architectures *(auto-generated)*
- **[ENUMERATIONS_REFERENCE.md](ENUMERATIONS_REFERENCE.md)** — All enumeration types *(auto-generated)*

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
- Organized by category (dopamine, acetylcholine, etc.)

## Benefits

✅ **Always Synchronized** - Documentation never drifts from code
✅ **Zero Maintenance** - No manual updates needed
✅ **Complete Coverage** - Catches all registered components
✅ **Consistent Format** - Uniform structure across all entries

## Integration

This auto-documentation system is referenced in:
- `docs/DOCUMENTATION_INDEX.md` - Main documentation index
- `.github/copilot-instructions.md` - AI assistant guidance
- Development workflow - Run before committing new components
