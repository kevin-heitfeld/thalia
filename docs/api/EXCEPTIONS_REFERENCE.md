# Exceptions Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-26 14:17:33
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all custom exception classes in Thalia.

Total: 6 exception classes

## 🎯 When to Use Each Exception

| Exception | Use Case |
|-----------|----------|
| `ConfigurationError` | Invalid configuration values or missing required config |
| `ComponentError` | Issues with component registration or initialization |
| `CheckpointError` | Problems loading or saving checkpoints |

## Exception Hierarchy

### [``BiologicalPlausibilityError``](../../src/thalia/core/errors.py#L98)

**Inherits from**: `ThaliaError`

**Source**: [`thalia/core/errors.py`](../../src/thalia/core/errors.py)

**Description**: Operation violates biological plausibility constraints.

---

### [``CheckpointError``](../../src/thalia/core/errors.py#L113)

**Inherits from**: `ThaliaError`

**Source**: [`thalia/core/errors.py`](../../src/thalia/core/errors.py)

**Description**: Error in checkpoint save/load operations.

---

### [``ComponentError``](../../src/thalia/core/errors.py#L68)

**Inherits from**: `ThaliaError`

**Source**: [`thalia/core/errors.py`](../../src/thalia/core/errors.py)

**Description**: Error in brain component (region or pathway).

---

### [``ConfigurationError``](../../src/thalia/core/errors.py#L87)

**Inherits from**: `ThaliaError`

**Source**: [`thalia/core/errors.py`](../../src/thalia/core/errors.py)

**Description**: Invalid configuration parameters.

---

### [``IntegrationError``](../../src/thalia/core/errors.py#L124)

**Inherits from**: `ThaliaError`

**Source**: [`thalia/core/errors.py`](../../src/thalia/core/errors.py)

**Description**: Error in brain-wide coordination or integration.

---

### [``ThaliaError``](../../src/thalia/core/errors.py#L51)

**Inherits from**: `Exception`

**Source**: [`thalia/core/errors.py`](../../src/thalia/core/errors.py)

**Description**: Base exception for all Thalia-specific errors.

---

