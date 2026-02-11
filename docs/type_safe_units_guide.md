# Guide: Type-Safe Unit Handling in Thalia

This document explains how to use Thalia's type system to prevent unit confusion
bugs (like passing currents where conductances are expected).

## The Problem We're Solving

In conductance-based neural models, we work with three related but DISTINCT quantities:

1. **Voltage** (V): Membrane potential, range typically [-1, 4] (normalized)
2. **Conductance** (g): Synaptic/membrane conductance, range [0, 5] (normalized by g_L)
3. **Current** (I): Membrane current, derived from I = g × (E - V)

**The bug**: Treating currents as conductances or vice versa.
- Example: Computing I = w × spikes, then passing I as conductance to neurons
- Result: Wrong dynamics, hyperactivity, oscillations

## The Solution: Type Aliases

We use Python's `NewType` for zero-cost type checking:

```python
from thalia.units import Conductance, ConductanceTensor, Voltage, VoltageTensor

def my_neuron_forward(
    g_exc: ConductanceTensor,  # Type checker enforces this is conductance
    g_inh: ConductanceTensor,
) -> tuple[torch.Tensor, VoltageTensor]:
    ...
```

Type checkers (mypy, pyright, pylance) will flag mismatches:

```python
# ❌ ERROR: Type mismatch
current = compute_current(weights, spikes)  # returns CurrentTensor
neurons.forward(current, None)  # expects ConductanceTensor!

# ✅ CORRECT: Use conductances directly
conductance = weights_as_conductances @ spikes.float()
neurons.forward(ConductanceTensor(conductance), None)
```

## Migration Guide

### Step 1: Import types

```python
from thalia.units import (
    Conductance, ConductanceTensor,
    Voltage, VoltageTensor,
    Current, CurrentTensor,
)
```

### Step 2: Annotate function signatures

**Before:**
```python
def forward(self, g_exc_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

**After:**
```python
def forward(
    self,
    g_exc_input: ConductanceTensor,  # Explicit: this is conductance
) -> tuple[torch.Tensor, VoltageTensor]:  # Explicit: returns voltage
    ...
```

### Step 3: Cast tensors when creating

```python
# When creating conductance tensors
g_exc = ConductanceTensor(weights @ spikes.float())

# When creating voltage tensors
v_mem = VoltageTensor(self.v_mem)
```

### Step 4: Handle region boundaries

At region boundaries, document units clearly:

```python
def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
    \"\"\"
    Args:
        region_inputs["sensory"]: Raw spikes (bool tensor)

    Returns:
        region_outputs["relay"]: Output spikes (bool tensor)

    Internal computation:
        1. Convert spikes to conductances: g = W @ spikes
        2. Pass conductances to neurons: neurons.forward(g_exc, g_inh)
        3. Return output spikes
    \"\"\"
    sensory_spikes = region_inputs.get("sensory")

    # Convert to conductance (weights represent conductances)
    g_exc = ConductanceTensor(self.synaptic_weights["sensory"] @ sensory_spikes.float())

    # Pass to neurons (type-checked!)
    relay_spikes, v_mem = self.relay_neurons.forward(g_exc, None)

    return {"relay": relay_spikes}
```

## Common Patterns

### Pattern 1: Synaptic weights AS conductances

**Correct:**
```python
# Weights represent conductances directly (e.g., 0.1 normalized conductance)
self.weights = nn.Parameter(torch.randn(n_out, n_in) * 0.1)

# Multiply by spikes to get total conductance
g_total = ConductanceTensor(self.weights @ input_spikes.float())
```

**Wrong:**
```python
# ❌ Don't compute "current-like" quantities then treat as conductance
current = self.weights @ input_spikes * gain * modulation * noise
neurons.forward(current, None)  # Type error: Current != Conductance
```

### Pattern 2: Gain modulation

**Correct (modulate conductances):**
```python
g_base = self.weights @ spikes.float()
g_modulated = ConductanceTensor(g_base * gain_factor)  # Scale conductance
neurons.forward(g_modulated, None)
```

**Wrong (mix current and conductance):**
```python
current = self.weights @ spikes * gain * ne_gain + noise  # Current-like
neurons.forward(current, None)  # ❌ Treating current as conductance!
```

### Pattern 3: Noise injection

**Correct (add to conductance):**
```python
g_exc = self.weights @ spikes.float()
noise = torch.randn_like(g_exc) * 0.01
g_total = ConductanceTensor(g_exc + noise)  # Small conductance noise
```

**Correct (add to voltage directly):**
```python
v_noisy = self.v_mem + torch.randn(self.n_neurons) * 0.05
```

**Wrong (massive current noise treated as conductance):**
```python
g_exc = self.weights @ spikes * gain
noise = torch.randn_like(g_exc) * 0.5  # ❌ Huge noise, wrong scale!
g_total = g_exc + noise  # Now conductances can be > 5.0!
```

## Type Checking Setup

### VS Code with Pylance

In `.vscode/settings.json`:
```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticSeverityOverrides": {
        "reportGeneralTypeIssues": "warning"
    }
}
```

### mypy

In `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Too strict for gradual migration
check_untyped_defs = true
```

Run: `mypy src/thalia`

## Runtime Checks (Optional)

For debugging, enable runtime validation:

```python
from thalia.units import is_conductance, is_voltage

def forward(self, g_exc: ConductanceTensor) -> VoltageTensor:
    # Debug mode: validate inputs
    if __debug__:
        assert is_conductance(g_exc), f"Invalid conductance: {g_exc.min()}, {g_exc.max()}"

    # ... compute ...

    # Debug mode: validate outputs
    if __debug__:
        assert is_voltage(v_mem), f"Invalid voltage: {v_mem.min()}, {v_mem.max()}"

    return VoltageTensor(v_mem)
```

## Benefits

1. **Catch bugs at development time**: Type checker flags mismatches before running code
2. **Self-documenting code**: Function signatures explicitly show expected units
3. **Zero runtime cost**: NewType is erased at runtime (no wrapper overhead)
4. **Gradual migration**: Can add types incrementally without breaking existing code
5. **IDE support**: Autocomplete and hover tooltips show unit information

## Limitations

- **No arithmetic checking**: Type checker won't verify `g * V` produces current
- **Requires discipline**: Must cast tensors explicitly with `ConductanceTensor(...)`
- **Not foolproof**: Can still bypass with `cast()` or `# type: ignore`

But it catches 90% of unit confusion bugs with minimal effort!

## References

- PEP 484: Type Hints (https://peps.python.org/pep-0484/)
- Python typing module: https://docs.python.org/3/library/typing.html
- Dimensional analysis in scientific computing: https://pint.readthedocs.io/
