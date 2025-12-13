# Contributing to Thalia

Thank you for your interest in contributing to Thalia! This guide will help you understand the codebase structure and development patterns.

## Table of Contents

- [Getting Started](#getting-started)
- [Codebase Architecture](#codebase-architecture)
- [Adding New Components](#adding-new-components)
- [Learning Rules](#learning-rules)
- [Testing Guidelines](#testing-guidelines)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/kevin-heitfeld/thalia.git
cd thalia

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

## Codebase Architecture

### Core Principles

1. **Biological Plausibility**: All components must use local learning rules (no backpropagation)
2. **Spike-Based Processing**: Use binary spikes (0 or 1), not firing rates
3. **Single Instance**: No batching - maintain continuous temporal dynamics
4. **Device Management**: Always specify `device=` at tensor creation (Pattern 1)

### Directory Structure

```
src/thalia/
├── core/                   # Brain system orchestration
│   ├── brain.py           # EventDrivenBrain (main system)
│   └── diagnostics_keys.py # Standard metric names
├── regions/               # Brain regions
│   ├── cortex/           # Layered cortex (L4→L2/3→L5)
│   ├── hippocampus/      # Trisynaptic circuit (DG→CA3→CA1)
│   ├── striatum/         # Reinforcement learning (D1/D2)
│   ├── prefrontal.py     # Working memory
│   ├── cerebellum.py     # Error-corrective learning
│   └── thalamus.py       # Relay and gating
├── pathways/             # Inter-region connections
│   ├── spiking_pathway.py # LIF neurons + STDP
│   └── sensory_pathways.py # Sensory encoding
├── learning/             # Learning rules and strategies
│   ├── rules/            # STDP, BCM, Hebbian, etc.
│   └── strategies/       # Strategy pattern implementations
├── neuromodulation/      # DA, ACh, NE systems
├── components/           # Reusable building blocks
│   ├── neurons/          # LIF, Conductance-LIF
│   └── synapses/         # STP, weight initialization
└── config/               # Configuration management
```

## Adding New Components

### Adding a New Brain Region

1. **Create the region file** in `src/thalia/regions/`:

```python
"""
MyRegion - Brief description.

Biological basis:
- Function: What does this region do?
- Connectivity: Inputs from X, outputs to Y
- Learning rule: Local rule used (STDP, BCM, etc.)
"""

from thalia.regions.base import NeuralComponent
from thalia.core.base.component_config import NeuralComponentConfig

class MyRegion(NeuralComponent):
    """Your region implementation."""

    def __init__(self, config: NeuralComponentConfig):
        super().__init__(config)
        # Initialize neurons, weights, state

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Process one timestep.

        Args:
            input_spikes: Binary spike tensor [n_input]

        Returns:
            output_spikes: Binary spike tensor [n_output]
        """
        # 1. Update membrane potentials
        # 2. Generate spikes
        # 3. Apply learning if enabled
        return output_spikes

    def reset_state(self) -> None:
        """Reset dynamic state between episodes."""
        self.membrane = torch.zeros(self.n_output, device=self.device)
        # Reset other state variables
```

2. **Register the region**:

```python
from thalia.managers.component_registry import register_region

@register_region(
    "my_region",
    description="My custom brain region",
    version="1.0"
)
class MyRegion(NeuralComponent):
    ...
```

3. **Add tests** in `tests/unit/test_my_region.py`:

```python
def test_my_region_initialization():
    config = NeuralComponentConfig(n_input=100, n_output=50)
    region = MyRegion(config)
    assert region.n_input == 100
    assert region.n_output == 50

def test_my_region_forward():
    region = MyRegion(config)
    input_spikes = torch.zeros(100, dtype=torch.bool)
    input_spikes[:10] = True  # 10% active
    output = region.forward(input_spikes)
    assert output.shape == (50,)
    assert output.dtype == torch.bool
```

### Adding a New Learning Rule

1. **Create the learning strategy** in `src/thalia/learning/rules/`:

```python
from thalia.learning.rules.strategy_protocol import LearningStrategy

class MyLearningStrategy(LearningStrategy):
    """My custom learning rule."""

    def __init__(self, config):
        self.learning_rate = config.learning_rate

    def apply_learning(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Apply learning update.

        Must be LOCAL - no access to distant gradients!
        """
        # Compute weight update using only local information
        delta_w = self.learning_rate * torch.outer(post_spikes, pre_spikes)
        return weights + delta_w

    def reset_state(self) -> None:
        """Reset traces/eligibility."""
        pass  # If your rule has state
```

2. **Register the strategy**:

```python
from thalia.learning.rules.strategies import LearningStrategyRegistry

LearningStrategyRegistry.register(
    "my_learning",
    MyLearningStrategy,
    description="My custom learning rule"
)
```

3. **Use in a region**:

```python
from thalia.mixins.learning_mixin import LearningStrategyMixin

class MyRegion(NeuralComponent, LearningStrategyMixin):
    def __init__(self, config):
        super().__init__(config)
        self.learning_strategy = LearningStrategyRegistry.create(
            "my_learning",
            config=config
        )
```

## Testing Guidelines

### Unit Tests

- Test individual components in isolation
- Mock dependencies when needed
- Verify tensor shapes and dtypes
- Check biological constraints (spikes are binary, weights are positive, etc.)

```python
def test_region_output_is_binary():
    region = MyRegion(config)
    output = region.forward(input_spikes)
    assert output.dtype == torch.bool
    assert torch.all((output == 0) | (output == 1))
```

### Integration Tests

- Test component interactions
- Verify coordinated behaviors (theta rhythm, pathway delays, etc.)
- Check biological plausibility across system

```python
def test_cortex_hippocampus_pathway():
    cortex = LayeredCortex(cortex_config)
    pathway = SpikingPathway(pathway_config)

    cortex_spikes = cortex.forward(input_spikes)
    pathway_output = pathway.forward(cortex_spikes)

    # Verify spike propagation
    assert pathway_output.shape == (hippocampus_input_size,)
```

## Code Style

### General Guidelines

1. **Type hints**: All public methods must have type hints
2. **Docstrings**: Use Google-style docstrings
3. **Naming**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_CASE`
4. **Line length**: Max 120 characters
5. **Imports**: Organize as stdlib → third-party → local

### Biological Patterns

#### ✅ DO:

```python
# Use spike-based processing
spikes = (membrane > threshold).bool()

# Local learning rules
delta_w = torch.outer(post_activity, pre_activity)

# Device handling at creation
tensor = torch.zeros(size, device=device)

# Named constants
from thalia.regulation.learning_constants import LEARNING_RATE_STDP
```

#### ❌ DON'T:

```python
# Firing rates instead of spikes
firing_rate = membrane.sigmoid()

# Global error signals (backprop)
loss.backward()

# Device handling after creation
tensor = torch.zeros(size)
tensor = tensor.to(device)

# Magic numbers
learning_rate = 0.001
```

### Using Standard Constants

Import biological parameters from centralized modules:

```python
# Learning rates
from thalia.regulation.learning_constants import (
    LEARNING_RATE_STDP,
    LEARNING_RATE_BCM,
)

# Neuromodulator parameters
from thalia.neuromodulation.constants import (
    DA_PHASIC_DECAY_PER_MS,
    ACH_ENCODING_LEVEL,
)

# Neuron parameters
from thalia.components.neurons.neuron_constants import (
    V_THRESHOLD_STANDARD,
    TAU_MEM_STANDARD,
)

# Diagnostic keys
from thalia.core.diagnostics_keys import DiagnosticKeys as DK

def get_diagnostics(self):
    return {
        DK.FIRING_RATE: self.compute_rate(),
        DK.WEIGHT_MEAN: self.weights.mean().item(),
    }
```

## Configuration Class Naming Convention

### The `***Config` Pattern

**Rule**: All configuration dataclasses MUST use the `***Config` suffix.

**Purpose**: Distinguish configuration (immutable setup parameters) from state (dynamic runtime data).

```python
# ✅ CORRECT: Configuration dataclass
@dataclass
class HippocampusConfig(BrainComponentConfig):
    """Configuration for hippocampus region."""
    n_dg: int = 1000
    n_ca3: int = 500
    learning_rate: float = 0.01
    device: str = "cpu"

# ✅ CORRECT: State class (not configuration)
@dataclass
class HippocampusState:
    """Runtime state for hippocampus."""
    dg_spikes: torch.Tensor
    ca3_membrane: torch.Tensor
    encoding_mode: bool

# ❌ INCORRECT: Missing Config suffix
@dataclass
class Hippocampus:  # Should be HippocampusConfig
    n_dg: int = 1000
```

### When to Use Configuration vs State

**Configuration (`***Config`)**:
- Immutable parameters set at initialization
- Network architecture (layer sizes, connectivity)
- Learning hyperparameters (rates, time constants)
- Device placement
- Biological constants (tau_mem, v_threshold)

**State (separate class or attributes)**:
- Dynamic runtime variables that change every timestep
- Membrane potentials, spike history
- Weight matrices (learnable parameters)
- Neuromodulator levels
- Eligibility traces

### Example

```python
# Configuration: Set once at initialization
config = StriatumConfig(
    n_actions=10,
    learning_rate=0.001,
    use_td_lambda=True,
    device="cuda"
)

# State: Changes every timestep
striatum = Striatum(config)
striatum.forward(input_spikes)  # Updates internal state
striatum.state.dopamine = 0.8   # Runtime state variable
```

### Why This Matters

- **Clarity**: Immediate distinction between setup and runtime data
- **Checkpointing**: Configuration saved once, state saved per checkpoint
- **Type Safety**: IDEs can autocomplete `config.learning_rate` vs `state.membrane`
- **Documentation**: Clear which values are hyperparameters vs dynamics
- **Testing**: Easy to create configs for different test scenarios

## Checkpoints and Brain Growth

**⚠️ CRITICAL LIMITATION**: Checkpoints are tied to brain architecture dimensions.

### The Rule

- ✅ **Can**: Save and load checkpoints with same architecture
- ✅ **Can**: Load checkpoint, then grow brain
- ❌ **Cannot**: Load old checkpoint into grown brain (dimension mismatch)
- ❌ **Cannot**: Load grown checkpoint into smaller brain

### Why This Matters

Checkpoints store tensors with fixed dimensions:
```python
# Checkpoint contains:
weights: torch.Tensor        # [n_output, n_input]
membrane: torch.Tensor       # [n_output]
vote_accumulators: torch.Tensor  # [n_actions]
```

Growing the brain changes `n_output` and `n_actions`, making old checkpoints incompatible:

```python
# Session 1: Train with 5 actions
brain = Brain(config_with_n_actions=5)
train(brain)
save_checkpoint("checkpoint.pt")  # Saves tensors shaped for n_actions=5

# Session 2: Resume and grow
brain = Brain(config_with_n_actions=5)
brain.load_checkpoint("checkpoint.pt")  # ✅ OK: dimensions match
brain.striatum.add_neurons(n_new=2)     # Now n_actions=7
train(brain)
save_checkpoint("grown_checkpoint.pt")  # Saves tensors shaped for n_actions=7

# Session 3: Try to resume from original checkpoint
brain = Brain(config_with_n_actions=7)  # Using grown architecture
brain.load_checkpoint("checkpoint.pt")  # ❌ CRASH: 5 ≠ 7
```

### Workaround

If you need to resume from an old checkpoint after growth:

```python
# Option 1: Load with original architecture, then re-grow
brain = Brain(config_with_n_actions=5)  # Match checkpoint
brain.load_checkpoint("old.pt")
brain.striatum.add_neurons(n_new=2)     # Re-apply growth
train(brain)

# Option 2: Keep separate checkpoint lineages per architecture
save_checkpoint(f"checkpoint_{brain.striatum.n_actions}actions.pt")
```

### Best Practices

1. **Save before growing**: Always checkpoint before `add_neurons()`
2. **Document growth**: Log growth events with step numbers
3. **Unique filenames**: Include architecture info in checkpoint names
4. **Keep base checkpoint**: Preserve pre-growth checkpoint for rollback
5. **Test dimension compatibility**: Validate checkpoint dimensions before loading

### Future Work

See [`docs/design/checkpoint_growth_compatibility.md`](../docs/design/checkpoint_growth_compatibility.md) for:
- Detailed analysis of the issue
- Proposed solutions (auto-resizing, migration)
- Testing requirements
- Implementation roadmap

## Device Management Patterns

### Pattern 1: Specify Device at Creation (Preferred)

**Rule**: For new tensors created within Thalia, ALWAYS specify `device=` at creation.

```python
# ✅ CORRECT: Specify device at creation
def __init__(self, config):
    self.membrane = torch.zeros(config.n_neurons, device=config.device)
    self.spikes = torch.zeros(config.n_neurons, dtype=torch.bool, device=config.device)

# ❌ INCORRECT: Create on CPU then move
def __init__(self, config):
    self.membrane = torch.zeros(config.n_neurons).to(config.device)  # Wasteful
```

### Pattern 2: Use WeightInitializer for Weights

**Rule**: For weight matrices, ALWAYS use the `WeightInitializer` registry.

```python
from thalia.utils.weight_initializer import WeightInitializer

# ✅ CORRECT: Use WeightInitializer with device
weights = WeightInitializer.gaussian(
    n_output=100,
    n_input=50,
    mean=0.3,
    std=0.1,
    device=config.device  # Always pass device
)

weights = WeightInitializer.xavier(n_output, n_input, device=device)
weights = WeightInitializer.sparse_random(
    n_output, n_input,
    sparsity=0.2,
    device=device
)

# ❌ INCORRECT: Direct torch initialization
weights = torch.randn(100, 50) * 0.1  # No device, no biological meaning
```

### Pattern 3: Move External Data to Device

**Rule**: For data coming from outside (datasets, user input), move to device explicitly.

```python
def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
    # ✅ CORRECT: Move external input to our device
    input_spikes = input_spikes.to(self.device)

    # Now process on correct device
    output = torch.mv(self.weights, input_spikes.float())
    return output > 0.5
```

### Common Patterns Summary

| **Use Case** | **Pattern** | **Example** |
|---|---|---|
| State tensors | `torch.zeros(..., device=device)` | `membrane = torch.zeros(n, device=device)` |
| Weight matrices | `WeightInitializer.*(..., device=device)` | `w = WeightInitializer.gaussian(n, m, device=device)` |
| External data | `.to(self.device)` | `input = input.to(self.device)` |
| nn.Module | `.to(device)` after creation | `self.neurons = LIF(config).to(device)` |

### Why This Matters (ADR-007: PyTorch Consistency)

- **Performance**: Avoids CPU↔GPU transfers
- **Clarity**: Explicit device placement at creation point
- **Consistency**: All Thalia tensors follow same pattern
- **Debugging**: Easier to track device mismatches
- **Biological meaning**: WeightInitializer documents biological reasoning

## Pull Request Process

1. **Create a feature branch**: `git checkout -b feature/my-feature`
2. **Write tests first**: TDD approach when possible
3. **Implement feature**: Follow code style guidelines
4. **Run tests**: `pytest tests/`
5. **Check types**: `mypy src/thalia/` (if available)
6. **Update documentation**: Add docstrings and update README if needed
7. **Submit PR**: Include description of changes and rationale

### PR Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] New features have tests
- [ ] Docstrings added to public methods
- [ ] Type hints on all public methods
- [ ] Biological plausibility maintained (local learning, binary spikes)
- [ ] No breaking changes (or documented in PR)
- [ ] Updated `docs/` if adding major features

## Questions?

- Check `docs/` for detailed architecture documentation
- Review `docs/patterns/` for common patterns
- See `docs/decisions/` for architectural decision records (ADRs)
- Open an issue for questions or suggestions

## References

- **ADR-004**: Binary spikes (bool tensors)
- **ADR-005**: No batch dimension (single instance)
- **ADR-007**: PyTorch consistency (device handling)
- **ADR-011**: Large file justification (biological circuit integrity)

Thank you for contributing to Thalia! 🧠
