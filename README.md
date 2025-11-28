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

## Installation

```bash
# Clone the repository
git clone https://github.com/username/thalia.git
cd thalia

# Install in development mode
pip install -e ".[dev]"

# Or with experiment dependencies
pip install -e ".[all]"
```

## Quick Start

```python
from thalia.core import LIFNeuron, SNNLayer
from thalia.learning import STDP

# Create a layer of 100 LIF neurons
layer = SNNLayer(n_neurons=100, neuron_type=LIFNeuron)

# Simulate for 1000ms
spikes = layer.simulate(duration=1000, dt=1.0)

# Visualize
from thalia.visualization import plot_raster
plot_raster(spikes)
```

## Project Status

🚧 **Pre-Alpha** — Core infrastructure under development

See [docs/PLANNING.md](docs/PLANNING.md) for the full roadmap.

## Documentation

- [Planning & Roadmap](docs/PLANNING.md)
- [Architecture Decisions](docs/DECISIONS.md)
- [Design Documents](docs/design/)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## License

MIT License — see [LICENSE](LICENSE) for details.
