# Copilot Instructions for Thalia

This file provides context for AI assistants working with the Thalia codebase.

## Project Overview

**Thalia** is a biologically-accurate spiking neural network framework for building multi-modal, biologically-plausible ML models with LLM-level (or better) capabilities.

Implements multiple brain regions with specialized learning rules to achieve:
- Language understanding and generation
- Multi-modal sensory processing (vision, audio, text)
- Reinforcement learning and decision-making
- Episodic and working memory

**Architecture Philosophy**:
- **Not**: Traditional deep learning with backpropagation
- **Is**: Neuroscience-inspired spiking networks with local learning rules and neuromodulation
- **Goal**: Match or exceed LLM capabilities using biologically-plausible mechanisms

## Architecture Principles

### 1. Brain Regions are Specialized
Each region has its own learning rule:
- **Striatum**: Three-factor rule (eligibility × dopamine) for RL
- **Hippocampus**: One-shot Hebbian for episodic memory
- **Cortex**: Unsupervised Hebbian/BCM/STDP for features
- **Cerebellum**: Supervised error-corrective (delta rule)
- **Prefrontal**: Gated Hebbian for working memory

### 2. All Processing is Spike-Based
- Use binary spikes (0 or 1), not firing rates
- LIF/Conductance-LIF neurons with membrane dynamics
- Temporal dynamics matter (spike timing, delays, traces)

### 3. Neuromodulation is Key
- **Dopamine**: Gates learning in striatum and prefrontal
- **Acetylcholine**: Modulates encoding/retrieval in hippocampus
- **Norepinephrine**: Arousal and gain modulation
- Set via `region.set_dopamine(level)`, not passed every forward()

## Code Patterns

### State Management
```python
# Mutable timestep state → RegionState dataclass
self.state.spikes  # Current spikes
self.state.dopamine  # Current neuromodulator level

# Immutable/config → Direct attributes
self.weights  # Learnable parameters
self.config  # Configuration
self.neurons  # Neuron objects
```
**See**: `docs/patterns/state-management.md`

### Weight Initialization
```python
# Always use WeightInitializer registry
weights = WeightInitializer.gaussian(n_output, n_input, mean=0.3, std=0.1, device=device)
weights = WeightInitializer.xavier(n_output, n_input, device=device)
weights = WeightInitializer.sparse_random(n_output, n_input, sparsity=0.2, device=device)

# Never: torch.randn() or torch.rand() directly
```

### Device Management
```python
# Pattern 1 (preferred): Specify device at creation
tensor = torch.zeros(size, device=device)

# Pattern 2 (only for nn.Module): Move after creation
module.to(device)

# Pattern 2 (only for external data): Move to correct device
input_data = batch["input"].to(self.device)
```

### Imports
```python
# External users (notebooks, experiments)
from thalia import Brain, Striatum, WeightInitializer

# Internal development (library code)
from thalia.core.neuron import ConductanceLIF
from thalia.regions.striatum import Striatum
from thalia.learning.bcm import BCMRule
```

## Common Tasks

### Adding a New Region
1. Inherit from `BrainRegion` (+ mixins if needed)
2. Create `RegionConfig` dataclass
3. Implement `_initialize_weights()` using `WeightInitializer`
4. Implement `forward()` - spike-based processing
5. Implement `_get_learning_rule()` - return learning type
6. Add to `regions/__init__.py` exports
7. Register with `@register_region("name")`

### Adding Learning Functionality
- **Diagnostics**: Use `DiagnosticsMixin`
- **Action Selection**: Use `ActionSelectionMixin`
- **Learning Strategies**: Use `LearningStrategyMixin`

### Testing
```bash
# Run specific test file
pytest tests/unit/test_brain_regions.py

# Run with coverage
pytest --cov=src/thalia tests/

# Run integration tests
pytest tests/integration/
```

## Key Files

### Documentation
- `docs/patterns/state-management.md` - When to use RegionState vs attributes
- `docs/patterns/configuration.md` - Config hierarchy and parameters
- `docs/patterns/mixins.md` - Available mixins and their methods
- `docs/design/checkpoint_format.md` - Checkpoint format specification
- `docs/design/curriculum_strategy.md` - Training curriculum stages
- `docs/decisions/` - Architecture decision records (ADRs)
- `docs/CODEBASE_IMPROVEMENTS.md` - Completed improvements roadmap

### Core Components
- `src/thalia/core/neuron.py` - LIF and ConductanceLIF neurons
- `src/thalia/core/weight_init.py` - Weight initialization registry
- `src/thalia/regions/base.py` - BrainRegion abstract base
- `src/thalia/core/brain.py` - Full brain system

### Regions
- `src/thalia/regions/striatum/` - Reinforcement learning (3-factor rule)
- `src/thalia/regions/hippocampus/` - Episodic memory (trisynaptic circuit)
- `src/thalia/regions/cortex/` - Feature learning (layered cortex)
- `src/thalia/regions/prefrontal.py` - Working memory (gated STDP)
- `src/thalia/regions/cerebellum.py` - Supervised learning (error-corrective)

## Biological Accuracy Constraints

### DO:
- Use spike-based processing (binary spikes)
- Implement local learning rules (no backprop)
- Respect biological time constants (tau_mem ~10-30ms)
- Use neuromodulators for gating/modulation
- Maintain causality (no future information)

### DON'T:
- Use global error signals or backpropagation
- Accumulate firing rates instead of individual spikes
- Implement non-local learning rules
- Use negative firing rates
- Access future timesteps in current computation

## Performance Considerations

- Batch size is typically 1 (single trial processing)
- Use `device=device` at tensor creation (not `.to(device)`)
- Sparse connectivity where biologically appropriate
- Short-term plasticity adds overhead (enable when needed)

## Debugging Tips

### Check Health
```python
health = region.check_health()
if not health.is_healthy:
    print(health.issues)
```

### Monitor Spikes
```python
firing_rate = region.get_firing_rate(spikes)
if firing_rate > 0.9:  # Runaway excitation
    # Check inhibition, thresholds
if firing_rate < 0.01:  # Silence
    # Check input strength, weights
```

### Visualize Learning
```python
metrics = region.learn(...)
print(f"LTP: {metrics['ltp']}, LTD: {metrics['ltd']}")
```

## Questions to Ask

When uncertain about implementation:
1. **Is this biologically plausible?** (Check neuroscience literature)
2. **Is learning local?** (No neuron should access distant gradients)
3. **Are spikes binary?** (No analog firing rates in processing)
4. **Is device handling correct?** (Pattern 1 for new tensors)
5. **Does it match existing patterns?** (Check similar regions)

## References

- **State Management**: `docs/patterns/state-management.md`
- **Config System**: `docs/patterns/configuration.md`
- **Mixins**: `docs/patterns/mixins.md`
- **Improvements**: `docs/CODEBASE_IMPROVEMENTS.md`

---

**Last Updated**: December 7, 2025
