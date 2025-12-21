# Protocols Reference

> **Auto-generated documentation** - Do not edit manually!
> Last updated: 2025-12-21 21:45:51
> Generated from: `scripts/generate_api_docs.py`

This document catalogs all Protocol classes defining interfaces for duck-typed components in Thalia.

Total: 8 protocols

## Protocol Classes

### `BrainComponent`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\component.py`

**Description**: Unified protocol for all brain components (regions AND pathways).

**Required Methods**:

```python
def forward():
    ...

def reset_state():
    ...

def set_oscillator_phases(phases: Dict[str, float], signals: Dict[str, float] | None, theta_slot: int, coupled_amplitudes: Dict[str, float] | None):
    ...

def grow_input(n_new: int, initialization: str, sparsity: float):
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

```

---

### `Configurable`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\neural.py`

**Description**: Protocol for components initialized from configuration.

**Required Methods**:

```python
def from_thalia_config(cls, config: Any):
    ...

```

---

### `Diagnosable`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\neural.py`

**Description**: Protocol for components that provide diagnostic information.

**Required Methods**:

```python
def get_diagnostics():
    ...

```

---

### `Forwardable`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\neural.py`

**Description**: Protocol for components that process input through forward pass.

**Required Methods**:

```python
def forward(input_spikes: torch.Tensor):
    ...

```

---

### `Learnable`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\neural.py`

**Description**: Protocol for components with synaptic plasticity.

**Required Methods**:

```python
def learn(input_spikes: torch.Tensor, output_spikes: torch.Tensor):
    ...

```

---

### `NeuralComponentProtocol`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\neural.py`

**Description**: Full protocol for neural components (regions, pathways, populations).

---

### `Resettable`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\neural.py`

**Description**: Protocol for components that can reset their state.

**Required Methods**:

```python
def reset_state():
    ...

```

---

### `WeightContainer`

**Runtime Checkable**: ❌ No (static type checking only)

**Source**: `thalia\core\protocols\neural.py`

**Description**: Protocol for components that have learnable weights.

**Required Methods**:

```python
def get_weights():
    ...

def set_weights(weights: torch.Tensor):
    ...

```

---

