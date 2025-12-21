# Exceptions Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 19:26:47
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all custom exception classes in Thalia.

Total: 6 exception classes

## Exception Hierarchy

### `BiologicalPlausibilityError`

**Inherits from**: `ThaliaError`

**Source**: `thalia\core\errors.py`

**Description**: Operation violates biological plausibility constraints.

---

### `CheckpointError`

**Inherits from**: `ThaliaError`

**Source**: `thalia\core\errors.py`

**Description**: Error in checkpoint save/load operations.

---

### `ComponentError`

**Inherits from**: `ThaliaError`

**Source**: `thalia\core\errors.py`

**Description**: Error in brain component (region or pathway).

---

### `ConfigurationError`

**Inherits from**: `ThaliaError`

**Source**: `thalia\core\errors.py`

**Description**: Invalid configuration parameters.

---

### `IntegrationError`

**Inherits from**: `ThaliaError`

**Source**: `thalia\core\errors.py`

**Description**: Error in brain-wide coordination or integration.

---

### `ThaliaError`

**Inherits from**: `Exception`

**Source**: `thalia\core\errors.py`

**Description**: Base exception for all Thalia-specific errors.

---

