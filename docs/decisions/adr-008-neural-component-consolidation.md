# ADR-008: Neural Component Consolidation

**Status**: Implemented
**Date**: December 11, 2025
**Author**: Thalia Team

## Context

Previously, Thalia had two separate base classes for neural populations:
- `BrainRegion` (for named functional units like Cortex, Hippocampus)
- `BaseNeuralPathway` (for inter-region connections)

This created artificial distinctions and code duplication, despite both being fundamentally the same: **populations of neurons with weights, dynamics, and learning**.

## Decision

**Consolidate into single `NeuralComponent` base class.**

All neural populations (regions, pathways, components) now inherit from `NeuralComponent`, which provides:
- Weights (synaptic connections)
- Dynamics (membrane potentials, spikes, traces)
- Learning rules (STDP, BCM, three-factor, etc.)
- Neuromodulation (dopamine, acetylcholine, norepinephrine) via `NeuromodulatorMixin`
- Growth support (add_neurons, capacity metrics)
- Diagnostics and health monitoring
- Checkpointing

## Implementation

### Unified Base Class

```python
# src/thalia/regions/base.py
class NeuralComponent(nn.Module, NeuromodulatorMixin, ABC):
    """Abstract base for ALL neural components (regions, pathways, populations)."""
    pass

# Backward compatibility
BrainRegion = NeuralComponent
```

### Pathway Consolidation

```python
# src/thalia/core/pathway_protocol.py
from thalia.regions.base import NeuralComponent

# Backward compatibility: BaseNeuralPathway is now just an alias
BaseNeuralPathway = NeuralComponent
```

### Usage

```python
# Named functional populations (regions)
class LayeredCortex(NeuralComponent):
    """Cortical layers with Hebbian/BCM learning."""
    pass

class Striatum(NeuralComponent):
    """Reinforcement learning with three-factor rule."""
    pass

# Connection populations (pathways)
class SpikingComponent(NeuralComponent):
    """Inter-region spiking connections with STDP."""
    pass
```

## Rationale

### Biological Reality

In the brain, there's no fundamental distinction between "regions" and "pathways":
- **V1 cortex**: Population of neurons
- **Thalamocortical pathway**: Also a population of neurons
- **Distinction**: Organizational/functional, not architectural

Both have:
- Synaptic weights that adapt
- Membrane dynamics and spike generation
- Plasticity rules (STDP, Hebbian, etc.)
- Neuromodulator receptors

### Code Quality

**Before**:
- `BrainRegion`: 200+ lines of base implementation
- `BaseNeuralPathway`: 200+ lines of duplicate implementation
- Separate maintenance, potential drift
- Pathways manually implemented neuromodulator state

**After**:
- `NeuralComponent`: Single 200+ line implementation
- All components inherit same interface
- Pathways automatically get neuromodulator support
- No code duplication

### API Consistency

**Before**:
```python
cortex.set_neuromodulators(da, ne, ach)  # Works
pathway.set_neuromodulators(da, ne, ach)  # Had to manually implement
```

**After**:
```python
cortex.set_neuromodulators(da, ne, ach)  # Works
pathway.set_neuromodulators(da, ne, ach)  # Inherited - just works!
```

## Consequences

### Positive

1. ✅ **Unified codebase**: Single base class, single source of truth
2. ✅ **Automatic neuromodulation**: Pathways inherit from `NeuromodulatorMixin`
3. ✅ **Reduced duplication**: ~200 lines of duplicate code eliminated
4. ✅ **Consistent API**: Same methods work for regions and pathways
5. ✅ **Biological accuracy**: Reflects reality that all are neuron populations
6. ✅ **Easier maintenance**: Changes propagate to all components automatically

### Considerations

1. **Semantic clarity**: "Region" vs "pathway" distinction now in class names, not base class
   - Solution: Descriptive names (LayeredCortex vs SpikingComponent)
2. **Backward compatibility**: Need aliases for existing code
   - Solution: `BrainRegion = NeuralComponent`, `BaseNeuralPathway = NeuralComponent`
3. **Documentation**: Need to update references
   - Solution: This ADR + updated docstrings

## Migration Guide

### For Library Users (Minimal Changes)

```python
# Before (still works!)
from thalia.regions.base import BrainRegion
from thalia.core.pathway_protocol import BaseNeuralPathway

# After (recommended)
from thalia.regions.base import NeuralComponent

# Both work - aliases maintained for backward compatibility
```

### For Contrib utors (New Code)

```python
# Named functional unit
class MyRegion(NeuralComponent):
    """My specialized region."""
    pass

# Connection population
class MyPathway(NeuralComponent):
    """My specialized pathway."""
    pass
```

## Related Decisions

- **ADR-006**: Temporal coding (pathways encode spike timing)
- **ADR-007**: PyTorch consistency (all use forward())
- **Component Parity Pattern**: Regions and pathways must have same features

## References

- `src/thalia/regions/base.py` - NeuralComponent implementation
- `src/thalia/core/pathway_protocol.py` - BaseNeuralPathway alias
- `src/thalia/core/neuromodulator_mixin.py` - Shared neuromodulation
- `docs/patterns/component-parity.md` - Design pattern documentation

---

**Key Insight**: The distinction between "regions" and "pathways" is organizational (how we think about the system), not architectural (what they fundamentally are). Both are populations of neurons, so they should share a base class.
