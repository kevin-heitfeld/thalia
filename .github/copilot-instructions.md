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

## ⭐ CRITICAL: Component Parity Principle

**Pathways are just as important as regions!**

When implementing features:
1. Add to `BrainComponent` protocol first (`src/thalia/core/component_protocol.py`)
2. Implement for `BrainRegion` base class
3. Implement for `BaseNeuralPathway` base class
4. Write tests for BOTH regions AND pathways
5. Update documentation mentioning both

**Why this matters:**
- Pathways are active learning components, not just "glue code"
- They learn via STDP/BCM during forward passes
- They need growth when connected regions grow
- They can become pathological (silent, saturated)
- Forgetting pathways breaks curriculum learning

**See**: `docs/patterns/component-parity.md` for detailed guidance.

## Architecture Principles

### 1. Brain Regions AND Pathways are Specialized
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

## Key Files

### Documentation
- `docs/patterns/component-parity.md` - ⭐ **START HERE** - Regions and pathways parity
- `docs/patterns/state-management.md` - When to use RegionState vs attributes
- `docs/patterns/mixins.md` - Available mixins and their methods
- `docs/design/checkpoint_format.md` - Checkpoint format specification
- `docs/design/curriculum_strategy.md` - Training curriculum stages
- `docs/decisions/` - Architecture decision records (ADRs)

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

## Questions to Ask

When uncertain about implementation:
1. **Is this biologically plausible?** (Check neuroscience literature)
2. **Is learning local?** (No neuron should access distant gradients)
3. **Are spikes binary?** (No analog firing rates in processing)
4. **Is device handling correct?** (Pattern 1 for new tensors)
5. **Does it match existing patterns?** (Check similar regions)
6. **Did I implement for BOTH regions AND pathways?** (Check component-parity.md)

## References

- **Component Parity**: `docs/patterns/component-parity.md`
- **State Management**: `docs/patterns/state-management.md`
- **Mixins**: `docs/patterns/mixins.md`
