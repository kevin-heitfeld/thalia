# Protocols Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2026-01-26 15:21:38
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all Protocol classes defining interfaces for duck-typed components in Thalia.

Total: 1 protocols

## Protocol Classes

### [``BrainComponent``](../../src/thalia/core/protocols/component.py#L37)

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: [`thalia/core/protocols/component.py`](../../src/thalia/core/protocols/component.py)

**Description**: Unified protocol for all brain components (regions AND pathways).

**Required Methods**:

```python
def forward():
    ...

def reset_state():
    ...

def set_oscillator_phases(phases: Dict[str, float], signals: Dict[str, float] | None, theta_slot: int, coupled_amplitudes: Dict[str, float] | None):
    ...

def grow_output(n_new: int, initialization: str, sparsity: float):
    ...

def get_capacity_metrics():
    ...

def get_diagnostics():
    ...

def check_health():
    ...

def get_full_state():
    ...

def load_full_state(state: Dict[str, Any]):
    ...

def device():
    ...

```

---

