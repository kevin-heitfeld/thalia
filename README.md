# THALIA

> **Th**inking **A**rchitecture via **L**earning **I**ntegrated **A**ttractors

*"She who flourishes"* — Greek Muse of spontaneous thought

A framework for building genuinely thinking spiking neural networks that generate thoughts through recurrent dynamics, form concept attractors, and achieve spontaneous cognition.

## Vision

Create an SNN-based architecture where **thinking is not input→output processing, but the network talking to itself** — recurrent dynamics that generate, test, and evolve thoughts spontaneously.

## Core Principles

1. **Thoughts as Dynamical Attractors** — Stable patterns of neural activity representing concepts
2. **Temporal Dynamics** — Spike timing matters, not just rates
3. **Hierarchical Time Constants** — Fast sensory, slow abstract layers
4. **Self-Referential Processing** — Output feeds back as input
5. **Embodied Grounding** — Concepts emerge from sensorimotor patterns

**Key Features:**
- **Spiking Neurons**: ConductanceLIF neurons (ONLY neuron model) with conductance-based dynamics
- **Flexible Construction**: DynamicBrain built with `BrainBuilder` or from configuration
- **Circuit Modeling**: Laminar cortex (L4→L2/3→L5, L6a/L6b corticothalamic), trisynaptic hippocampus (DG→CA3→CA1), D1/D2 striatal pathways
- **Clock-Driven Execution**: Regular timesteps (1ms) with biologically accurate axonal delays (1-20ms)
- **Axonal Delays**: Realistic conduction delays (1-20ms) via `CircularDelayBuffer` in `AxonalProjection`
- **Synapses at Dendrites**: Weights stored at target regions (`synaptic_weights` dict)
- **Neuromodulation**: Dopamine (reward), acetylcholine (encoding/retrieval), norepinephrine (arousal)
- **Learning Strategies**: Pluggable learning rules (STDP, BCM, Hebbian, three-factor, error-corrective)
- **Learning Rules**: STDP, BCM, Hebbian, three-factor (dopamine-modulated) at target synapses
- **Temporal Coordination**: Theta (8Hz), alpha (10Hz), gamma (40Hz) oscillations with cross-frequency coupling

## Documentation

- [Architecture Decisions](docs/decisions/) — ADRs documenting key technical choices
- [Curriculum Strategy](docs/curriculum_strategy.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
