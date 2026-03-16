# THALIA

> **Th**inking **A**rchitecture via **L**earning **I**ntegrated **A**ttractors

*"She who flourishes"* — Greek Muse of spontaneous thought

A spiking neural network framework occupying the middle ground between current ML architectures (transformers, deep learning) and full biophysical simulation (Blue Brain Project). Built on the thesis that this middle ground — biologically principled but computationally tractable — is the correct path forward from today's ML.

## Why the Middle Ground?

Current ML networks (transformers, DNNs) are powerful but fundamentally disconnected from how brains compute — they rely on global backpropagation, continuous activations, and stateless layers. Biophysical simulations (Blue Brain Project) are faithful to biology but too detailed to scale or learn. Thalia bridges this gap:

- **More biologically grounded than transformers**: spike-based processing, local learning rules (STDP, Hebbian, neuromodulated plasticity), persistent neural state, neuromodulation (dopamine, acetylcholine), no backpropagation
- **More scalable and learnable than biophysical simulations**: simplified neuron models (conductance-based LIF, not compartmental Hodgkin-Huxley), designed to scale to large networks, focused on network-level computation over morphological fidelity

## Key Design Principles

- **Spike-based**: Binary spikes, not continuous activations or firing rates
- **Local learning only**: STDP, Hebbian, three-factor rules — no global error signals
- **Causal**: No future information leakage
- **Unified brain**: One persistent state with neuromodulation — no batch dimension
- **Recurrent attractor dynamics**: Thoughts emerge from network dynamics, not feedforward computation

## Documentation

- [Architecture Decisions](docs/decisions/) — ADRs documenting key technical choices
- [Curriculum Strategy](docs/curriculum_strategy.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
