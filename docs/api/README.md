# API Reference (Auto-Generated)

> **⚠️ Do not edit these files manually!**
> All documentation in this directory is auto-generated from the codebase.

## Contents

- **[COMPONENT_CATALOG.md](COMPONENT_CATALOG.md)** — All registered brain regions and pathways
- **[LEARNING_STRATEGIES_API.md](LEARNING_STRATEGIES_API.md)** — Learning strategy factory functions
- **[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)** — Configuration dataclasses

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
