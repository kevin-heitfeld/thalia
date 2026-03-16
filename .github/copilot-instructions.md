# Copilot Instructions for Thalia

This file provides context for AI assistants working with the Thalia codebase.

## ⚠️ CRITICAL: Tool Usage Restrictions

### NEVER Use Semantic Search

**DO NOT use the `semantic_search` tool - it returns STALE INFORMATION from deleted/moved files.**

The semantic search index is outdated and contains references to files that no longer exist. Using it will give you false information about the codebase.

## Development Philosophy

This codebase is developed by a single developer with full control. There are no external users and no backwards compatibility constraints. Refactorings should be made **aggressively** to keep the codebase clean:

- **Remove code, don't deprecate it.** If something is superseded or no longer needed, delete it outright.
- **Rename and restructure freely.** Update all call sites rather than adding compatibility shims.
- **Prefer clean design over migration paths.** There is no one else to break.
- **Extensive refactoring is encouraged.** You are allowed to edit **every** file in this project, no matter how large the changes. Do not shy away from sweeping structural changes when they improve the design.
- **Fix root causes, not symptoms.** Long-term architectural solutions are always preferred over quick patches that paper over deeper issues. If a fix requires touching 50 files, do it.

## Python Environment Setup

**Configure the Python environment before running any Python code, tests, or commands.**

Use the `configure_python_environment` tool at the start of every session or before running:
- Python scripts (training, examples, etc.)
- Tests (`pytest`, `runTests`)
- Python terminal commands
- Package installations

This ensures the correct environment is activated with all dependencies (pytest, torch, etc.) available. If you see import errors or missing packages that should be installed, you likely forgot this step.

## Project Overview

**Thalia** is a spiking neural network framework that occupies the middle ground between current ML architectures (transformers, deep learning) and full biophysical simulation (Blue Brain Project). It is built on the thesis that this middle ground — biologically principled but computationally tractable — is the correct path forward from today's ML.

**What Thalia is**:
- A bridge between ML and neuroscience: more biologically grounded than transformers, more scalable and learnable than biophysical simulations
- Spike-based processing with local learning rules (STDP, Hebbian, neuromodulated plasticity) — no backpropagation
- A unified brain with persistent state, neuromodulation, and recurrent attractor dynamics

**How it differs from current ML** (transformers, DNNs):
- Spikes instead of continuous activations
- Local learning rules instead of global backpropagation
- Causal processing (no future information leakage)
- Persistent neural state (membrane voltage, synaptic traces) instead of stateless layers
- Neuromodulators (dopamine, acetylcholine) actively gate learning and attention
- No batch dimension — one brain, one unified state

**How it differs from biophysical simulation** (Blue Brain Project):
- Simplified neuron models (conductance-based LIF, not compartmental Hodgkin-Huxley)
- Focus on learning and plasticity, not ultra-accurate ion channel kinetics
- Designed to scale to millions of neurons, not thousands
- Prioritizes network-level computation over morphological fidelity

**Goal**: Demonstrate that biologically principled spiking networks with local learning rules can achieve genuine cognition — learning, memory, and thought — without backpropagation or global loss functions.

## Biological Accuracy Constraints

### DO:
- Use spike-based processing (binary spikes)
- Implement local learning rules (no backprop)
- Maintain causality (no future information)
- Respect biological time constants (tau_mem ~10-30ms)
- Use neuromodulators for gating/modulation
- Use conductance inputs: All neurons expect `g_ampa_input`, `g_nmda_input`, `g_gaba_a_input`, `g_gaba_b_input` (conductances, NOT currents)

### DON'T:
- Accumulate firing rates instead of individual spikes
- Use global error signals or backpropagation
- Implement non-local learning rules
- Access future timesteps in current computation
- Pass currents as conductances: Cannot convert I → g without knowing voltage!
