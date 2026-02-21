# Copilot Instructions for Thalia

This file provides context for AI assistants working with the Thalia codebase.

## ⚠️ CRITICAL: Tool Usage Restrictions

### NEVER Use Semantic Search

**DO NOT use the `semantic_search` tool - it returns STALE INFORMATION from deleted/moved files.**

The semantic search index is outdated and contains references to files that no longer exist. Using it will give you false information about the codebase.

**Instead, use:**
- `grep_search` - searches actual current file contents
- `file_search` - finds files by glob pattern
- `read_file` - reads actual file contents
- `list_dir` - shows actual directory structure

## Python Environment Setup

**Configure the Python environment before running any Python code, tests, or commands.**

Use the `configure_python_environment` tool at the start of every session or before running:
- Python scripts (training, examples, etc.)
- Tests (`pytest`, `runTests`)
- Python terminal commands
- Package installations

This ensures the correct environment is activated with all dependencies (pytest, torch, etc.) available. If you see import errors or missing packages that should be installed, you likely forgot this step.

## Project Overview

**Thalia** is a biologically-accurate spiking neural network framework for building multi-modal, biologically-plausible ML models with LLM-level (or better) capabilities.

**Architecture Philosophy**:
- **Is**: Neuroscience-inspired spiking networks with local learning rules and neuromodulation
- **Is not**: Traditional deep learning with backpropagation
- **Goal**: Match or exceed LLM capabilities using biologically-plausible mechanisms

**Key Components**:
- **NeuralRegion**: Base class (nn.Module + mixins) for brain regions
    - **Learning Strategies**: Pluggable learning rules (STDP, BCM, Hebbian, Three-factor, etc.)
    - **Synaptic Weights**: Stored at target dendrites in `region._synaptic_weights` dict
- **AxonalTract**: Pure spike routing with delays (NO weights)

## Biological Accuracy Constraints

### DO:
- Use spike-based processing (binary spikes)
- Implement local learning rules (no backprop)
- Respect biological time constants (tau_mem ~10-30ms)
- Use neuromodulators for gating/modulation
- Maintain causality (no future information)
- **Use conductance inputs**: All neurons expect `g_exc`, `g_inh` (conductances, NOT currents)

### DON'T:
- Use global error signals or backpropagation
- Accumulate firing rates instead of individual spikes
- Implement non-local learning rules
- Use negative firing rates
- Access future timesteps in current computation
- **Pass currents as conductances**: Cannot convert I → g without knowing voltage!

## Type Safety: Units and Dimensional Analysis

**CRITICAL**: Thalia uses type aliases to prevent unit confusion bugs.

From `thalia.units`:
- `ConductanceTensor`: Synaptic/membrane conductances (≥ 0, normalized by g_L)
- `VoltageTensor`: Membrane potentials (typically [-1, 4] normalized)
- `CurrentTensor`: Membrane currents (derived: I = g × (E - V))

**Common bug**: Treating currents as conductances:
```python
# ❌ WRONG: Computing current-like quantities
current = weights @ spikes * gain * modulation + noise
neurons.forward(current, None)  # Type error!

# ✅ CORRECT: Use conductances directly
g_exc = ConductanceTensor(weights @ spikes.float())
neurons.forward(g_exc, None)
```

See [docs/type_safe_units_guide.md](../docs/type_safe_units_guide.md) for full details.
