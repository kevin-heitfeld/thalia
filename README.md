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

✅ **v0.9.2** — All 9 phases complete, 387 tests passing, 6 experiments validated

| Component | Status | Description |
|-----------|--------|-------------|
| Core SNN | ✅ | LIF neurons, synapses, layers |
| Learning | ✅ | STDP, homeostatic, reward-modulated |
| Attractors | ✅ | Pattern storage, recall, transitions |
| Memory | ✅ | Working memory with gating |
| Hierarchy | ✅ | Multi-timescale processing |
| Daydream | ✅ | Spontaneous thought generation |
| World Model | ✅ | Predictive processing |
| Inner Speech | ✅ | Verbal reasoning |
| Metacognition | ✅ | Self-monitoring, confidence |

### Experiments

```bash
# Run all experiments
python experiments/scripts/exp1_basic_lif.py      # Neuron dynamics
python experiments/scripts/exp2_stdp_learning.py  # STDP learning
python experiments/scripts/exp3_attractors.py     # Pattern recall
python experiments/scripts/exp4_mnist_snn.py      # Real MNIST (87%!)
python experiments/scripts/exp5_spontaneous_thought.py  # Free association
python experiments/scripts/exp6_sequence_learning.py    # Language patterns (83%!)
```

### Applications

```bash
# Interactive visualization demo
python examples/interactive_demo.py --text

# Chatbot using spiking neurons
python examples/chatbot.py --demo
```

See [docs/PLANNING.md](docs/PLANNING.md) for the full roadmap and [docs/CHANGELOG.md](docs/CHANGELOG.md) for version history.

## Documentation

- [Planning & Roadmap](docs/PLANNING.md)
- [Architecture Decisions](docs/DECISIONS.md)
- [Design Documents](docs/design/)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## License

MIT License — see [LICENSE](LICENSE) for details.
