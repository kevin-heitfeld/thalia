# Experiments Directory

This directory contains experimental notebooks and scripts for the Thalia spiking neural network project.

## Structure

- `notebooks/` - Jupyter notebooks for exploration and analysis
- `scripts/` - Standalone experiment scripts
- `results/` - Generated figures and output data

## Experiments

### Experiment 1: Basic LIF (`exp1_basic_lif.py`)
Basic Leaky Integrate-and-Fire neuron validation.

### Experiment 2: STDP Learning (`exp2_stdp_learning.py`) ✅ **COMPLETE**
Validates that a spiking neural network can learn temporal patterns using biologically realistic mechanisms.

**Key Results:**
- 9/10 feedforward accuracy from random initialization
- 5/5 recurrent chain accuracy (sequence prediction)
- Pure Hebbian coincidence detection (no STDP timing windows for feedforward)
- Predictive coding for recurrent connections

See `scripts/exp2_README.md` for detailed documentation.

### Experiment 3: Attractors (`exp3_attractors.py`)
Attractor dynamics in recurrent networks.

### Experiment 4: MNIST SNN (`exp4_mnist_snn.py`)
MNIST classification with spiking neural networks.

### Experiment 5: Spontaneous Thought (`exp5_spontaneous_thought.py`)
Exploring spontaneous activity patterns.

### Experiment 6: Sequence Learning (`exp6_sequence_learning.py`)
Learning sequential patterns and replay.
