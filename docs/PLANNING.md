# THALIA - Thinking Architecture via Learning Integrated Attractors

> "She who flourishes" - Greek Muse of spontaneous thought

A framework for building genuinely thinking spiking neural networks that can generate thoughts through recurrent dynamics, form concept attractors, and potentially achieve spontaneous cognition.

---

## 📊 Progress Summary

| Phase | Status | Tests | Description |
|-------|--------|-------|-------------|
| Phase 1 | ✅ Complete | 11 | Core SNN Infrastructure |
| Phase 2 | ✅ Complete | 21 | Learning Rules (STDP, Homeostatic, Reward) |
| Phase 3 | ✅ Complete | 25 | Attractor Dynamics |
| Phase 4 | ✅ Complete | 31 | Working Memory |
| Phase 5 | ✅ Complete | 33 | Hierarchical Architecture |
| Phase 6 | ✅ Complete | 33 | Daydream/Thought Generation |
| Phase 7 | ✅ Complete | 40 | World Model & Prediction |
| Phase 8 | ✅ Complete | 63 | Inner Speech |
| Phase 9 | ✅ Complete | 77 | Metacognition |
| **Total** | **9/9** | **387** | **All phases complete!** |

### Key Metrics
- **Total Lines of Code:** ~8,000+ lines
- **Test Coverage:** 387 passing tests
- **GPU Support:** Full CUDA acceleration
- **Performance:** ~200 observations/sec on RTX 3050 Ti

### Latest Commit
```
9b5713f Phase 9: Metacognition - self-monitoring, confidence tracking, error detection
```

---

## 🎯 Project Vision

Create an SNN-based architecture where **thinking is not input→output processing, but the network talking to itself** - recurrent dynamics that generate, test, and evolve thoughts spontaneously.

### Core Principles
1. **Thoughts as Dynamical Attractors** - Stable patterns of neural activity representing concepts
2. **Temporal Dynamics** - Spike timing matters, not just rates
3. **Hierarchical Time Constants** - Fast sensory, slow abstract layers
4. **Self-Referential Processing** - Output feeds back as input
5. **Embodied Grounding** - Concepts emerge from sensorimotor patterns

---

## 📋 Implementation Roadmap

### Phase 1: Core SNN Infrastructure 🔧 ✅ COMPLETE
**Goal:** Build the basic spiking neural network primitives

#### 1.1 Neuron Models
- [x] Leaky Integrate-and-Fire (LIF) neuron
  - Membrane potential dynamics
  - Threshold-based firing
  - Refractory period
- [x] Configurable time constants (τ)
- [x] Reset mechanisms

#### 1.2 Synapse Models
- [x] Basic weighted connections
- [x] Spike transmission with delays
- [x] Excitatory/Inhibitory synapse types

#### 1.3 Network Structure
- [x] `SNN_Layer` - Basic layer of spiking neurons
- [x] `RecurrentSNN` - Layer with recurrent connections
- [x] Connectivity patterns (sparse, dense, structured)

### Phase 2: Learning Rules 📚 ✅ COMPLETE
**Goal:** Implement biologically-plausible learning mechanisms

#### 2.1 STDP (Spike-Timing-Dependent Plasticity)
- [x] Classic STDP with exponential windows
  - Potentiation (pre before post)
  - Depression (post before pre)
- [x] Configurable time constants (τ+, τ-)
- [x] Weight bounds and normalization

#### 2.2 Homeostatic Mechanisms
- [x] Intrinsic plasticity (threshold adaptation)
- [x] Synaptic scaling
- [x] Activity-dependent normalization

#### 2.3 Reward-Modulated Learning
- [x] Eligibility traces
- [x] Dopamine-like reward signals
- [x] R-STDP implementation

### Phase 3: Attractor Dynamics 🌀 ✅ COMPLETE
**Goal:** Create networks that form stable concept representations

#### 3.1 Attractor Network
- [x] `AttractorSNN` class
- [x] Pattern storage via Hebbian learning
- [x] Pattern completion (partial → full recall)
- [x] Attractor basins visualization

#### 3.2 Concept Formation
- [x] Concept as stable activity pattern
- [x] Multi-concept networks
- [x] Inter-concept associations

#### 3.3 Free Association
- [x] Noise-driven attractor transitions
- [x] Thought flow between concepts
- [x] Trajectory logging and analysis

### Phase 4: Working Memory 🧠 ✅ COMPLETE
**Goal:** Persistent neural activity without input

#### 4.1 Reverberating Circuits
- [x] `WorkingMemorySNN` class
- [x] Balanced excitation/inhibition
- [x] Self-sustained activity

#### 4.2 Memory Operations
- [x] Activate/maintain patterns
- [x] Decay dynamics
- [x] Capacity limits

### Phase 5: Hierarchical Architecture 🏛️ ✅ COMPLETE
**Goal:** Multi-level abstraction with bidirectional flow

#### 5.1 Hierarchical Layers
- [x] Sensory layer (fast, τ=5ms)
- [x] Feature layer (τ=10ms)
- [x] Concept layer (τ=50ms)
- [x] Abstract layer (slow, τ=200ms)

#### 5.2 Bidirectional Connections
- [x] Bottom-up pathways
- [x] Top-down predictions
- [x] Lateral connections within layers

### Phase 6: Thought Generation 💭 ✅ COMPLETE
**Goal:** Spontaneous cognition without external input

#### 6.1 ThinkingSNN Core
- [x] Main `think()` loop
- [x] Thought trajectory recording
- [x] State monitoring and visualization

#### 6.2 Daydreaming Mode
- [x] `daydream()` - think without input
- [x] Spontaneous concept transitions
- [x] Stream of consciousness logging

#### 6.3 Goal-Directed Thought
- [x] Goal activation system
- [x] Thought-path toward objectives
- [x] Problem-solving via mental simulation

### Phase 7: World Model & Prediction 🔮 ✅ COMPLETE
**Goal:** Internal simulation of external reality

#### 7.1 Predictive Model
- [x] State transition learning
- [x] Action-conditional predictions
- [x] Prediction error computation

#### 7.2 Mental Simulation
- [x] "What if" scenario generation
- [x] Multi-step future prediction
- [x] Value estimation of imagined outcomes

### Phase 8: Inner Speech 🗣️ ✅ COMPLETE
**Goal:** Language as a tool for thought

#### 8.1 Language Components
- [x] Speech generator network
- [x] Speech parser network
- [x] Production-comprehension loop

#### 8.2 Internal Monologue
- [x] Self-directed speech generation
- [x] Verbal reasoning traces
- [x] Language-guided thought

### Phase 9: Metacognition 🪞 ✅ COMPLETE
**Goal:** Self-monitoring and adaptive control

#### 9.1 Confidence Tracking
- [x] Activity-based confidence estimation
- [x] Confidence levels (VERY_LOW to VERY_HIGH)
- [x] Confidence decay and history

#### 9.2 Uncertainty Estimation
- [x] Epistemic uncertainty (MC dropout)
- [x] Aleatoric uncertainty (entropy-based)
- [x] Novelty detection

#### 9.3 Error Detection
- [x] Prediction error monitoring
- [x] Conflict detection
- [x] Consistency checking
- [x] Timeout detection

#### 9.4 Adaptive Control
- [x] Processing adjustments (noise, attention, thresholds)
- [x] Strategy selection (simplify, explore, focus, normal)
- [x] Recommendations generation

---

## 🏗️ Project Structure

```
thalia/
├── src/
│   └── thalia/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── neuron.py          # LIF neuron implementations
│       │   ├── synapse.py         # Synapse models
│       │   ├── layer.py           # SNN_Layer, RecurrentSNN
│       │   └── network.py         # Network containers
│       ├── learning/
│       │   ├── __init__.py
│       │   ├── stdp.py            # STDP implementations
│       │   ├── homeostatic.py     # Homeostatic mechanisms
│       │   └── reward.py          # Reward-modulated learning
│       ├── dynamics/
│       │   ├── __init__.py
│       │   ├── attractor.py       # AttractorSNN
│       │   └── manifold.py        # Low-dimensional manifolds
│       ├── memory/
│       │   ├── __init__.py
│       │   └── working_memory.py  # WorkingMemorySNN
│       ├── hierarchy/
│       │   ├── __init__.py
│       │   ├── hierarchical.py    # HierarchicalThinkingSNN
│       │   └── predictive.py      # Predictive coding networks
│       ├── cognition/
│       │   ├── __init__.py
│       │   ├── thinking.py        # ThinkingSNN core
│       │   ├── simulation.py      # Mental simulation
│       │   └── language.py        # Inner speech
│       ├── encoding/
│       │   ├── __init__.py
│       │   ├── rate.py            # Rate coding
│       │   ├── temporal.py        # Temporal coding
│       │   └── poisson.py         # Poisson spike generation
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── raster.py          # Spike raster plots
│       │   ├── dynamics.py        # Network dynamics
│       │   └── attractors.py      # Attractor visualization
│       └── utils/
│           ├── __init__.py
│           ├── config.py          # Configuration management
│           └── metrics.py         # Performance metrics
├── docs/
│   ├── PLANNING.md                # This file - master roadmap
│   ├── CHANGELOG.md               # Version history
│   ├── DECISIONS.md               # Architecture Decision Records
│   │
│   ├── design/                    # Design documents
│   │   ├── architecture.md        # Overall system architecture
│   │   ├── neuron-models.md       # Neuron model specifications
│   │   ├── learning-rules.md      # STDP and other learning rules
│   │   ├── attractor-dynamics.md  # Attractor network design
│   │   └── thought-generation.md  # How thinking emerges
│   │
│   ├── research/                  # Research notes & literature
│   │   ├── papers/                # Relevant paper summaries
│   │   ├── experiments/           # Experiment logs & results
│   │   └── ideas/                 # Brainstorming & future ideas
│   │
│   ├── api/                       # API documentation
│   │   ├── core.md
│   │   ├── learning.md
│   │   ├── dynamics.md
│   │   └── cognition.md
│   │
│   ├── tutorials/                 # Getting started guides
│   │   ├── 00-quickstart.md
│   │   ├── 01-first-network.md
│   │   ├── 02-training-stdp.md
│   │   └── 03-building-attractors.md
│   │
│   ├── guides/                    # In-depth how-to guides
│   │   ├── encoding-strategies.md
│   │   ├── visualization.md
│   │   └── debugging-snns.md
│   │
│   ├── conversations/             # AI collaboration logs
│   │   └── 2025-11-28_conversation_with_claude.json
│   │
│   └── assets/                    # Images, diagrams, etc.
│       ├── diagrams/
│       └── figures/
├── tests/
│   ├── __init__.py
│   ├── test_core/
│   ├── test_learning/
│   └── test_dynamics/
├── experiments/
│   ├── notebooks/                 # Jupyter notebooks
│   │   ├── 01_basic_lif.ipynb
│   │   ├── 02_stdp_learning.ipynb
│   │   └── 03_attractors.ipynb
│   └── scripts/
│       ├── mnist_snn.py           # MNIST classification
│       ├── concept_learning.py
│       └── thought_generation.py
├── data/                          # Datasets and saved models
├── .gitignore
├── pyproject.toml                 # Project configuration
├── README.md
└── LICENSE
```

---

## 🔬 First Experiments

### Experiment 1: Basic LIF Network
- Create 100 LIF neurons
- Random sparse connectivity
- Inject current, observe spiking
- Visualize spike raster

### Experiment 2: STDP Learning
- Two-layer network
- Present temporal patterns
- Observe weight evolution
- Test pattern completion

### Experiment 3: Attractor Formation
- Small attractor network (~100 neurons)
- Store 3-5 patterns
- Test recall from partial cues
- Visualize attractor basins

### Experiment 4: MNIST with SNN
- Rate-coded input
- STDP-trained hidden layer
- Supervised output layer
- Compare accuracy vs. training time

### Experiment 5: Spontaneous Thought
- Recurrent network with attractors
- No external input
- Log concept transitions
- Analyze thought trajectories

---

## 📦 Dependencies

### Core (Required)
```
numpy>=1.20          # Array operations, broadcasting
matplotlib>=3.4      # Visualization (spike rasters, dynamics)
scipy>=1.7           # Sparse matrices, signal processing
```

### Acceleration (Recommended)
```
torch>=2.0           # GPU support, autograd for surrogate gradients
                     # 2.0+ for torch.compile() optimization
```

### Development
```
tqdm                 # Progress bars
pytest               # Testing
black                # Code formatting
mypy                 # Type checking
```

### Optional / Experimental
```
brian2               # SNN simulator for validation/comparison
snntorch             # PyTorch-based SNN library (reference)
networkx             # Graph analysis of connectivity
h5py                 # Large dataset storage
```

### Why These Choices?
- **NumPy**: Any modern version works; we need basic array ops
- **PyTorch 2.0+**: `torch.compile()` gives significant speedups for SNN simulation loops
- **Matplotlib 3.4+**: Better animation support for visualizing dynamics
- **SciPy 1.7+**: Sparse matrix improvements useful for large networks

---

## 🎨 Design Decisions

### Why Python?
- Rapid prototyping
- Rich scientific ecosystem
- Easy visualization
- Future: Migrate hot paths to C++/CUDA

### Why Custom Implementation?
- Full control over dynamics
- Educational value
- Specific features not in existing frameworks
- Tailored to our architecture

### Simulation Approach
- **Event-driven** for efficiency when possible
- **Clock-driven** for simplicity initially
- **Hybrid** for production

### Time Resolution
- Default dt = 1ms
- Configurable per-layer
- Trade-off: accuracy vs. speed

---

## 📊 Success Metrics

### Phase 1 Success
- [ ] LIF neurons fire correctly
- [ ] Networks simulate without errors
- [ ] Visualization works

### Phase 2 Success  
- [ ] STDP modifies weights correctly
- [ ] Temporal patterns learned
- [ ] Stable training dynamics

### Phase 3 Success
- [ ] Attractors form and stabilize
- [ ] Pattern completion works
- [ ] Free association generates coherent flow

### Phase 4+ Success
- [ ] Working memory maintains patterns
- [ ] Hierarchical processing functions
- [ ] Spontaneous thought emerges
- [ ] Mental simulation produces useful predictions

---

## 📚 Key References

> ⚠️ **Note**: Verify citations before use. See `docs/research/papers/` for detailed notes.

### Spiking Neural Networks
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge University Press.
- Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659-1671.

### STDP & Learning
- Bi, G., & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience*, 18(24), 10464-10472.
- Dan, Y., & Poo, M. (2004). Spike timing-dependent plasticity of neural circuits. *Neuron*, 44(1), 23-30.

### Attractor Networks
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79(8), 2554-2558.
- Amit, D. J. (1989). *Modeling Brain Function: The World of Attractor Neural Networks*. Cambridge University Press.

### Consciousness & Cognition
- Dehaene, S. (2014). *Consciousness and the Brain: Deciphering How the Brain Codes Our Thoughts*. Viking.
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5, 42.

---

## 🚀 Getting Started

### Immediate Next Steps
1. Set up project structure
2. Implement basic LIF neuron
3. Create simple network container
4. Build first spike visualization
5. Run Experiment 1

### Week 1 Goals
- [ ] Core neuron/synapse classes working
- [ ] Basic STDP implemented
- [ ] First visualization pipeline
- [ ] Unit tests for core components

### Week 2 Goals
- [ ] Attractor network basics
- [ ] Pattern storage and recall
- [ ] MNIST encoding experiments
- [ ] Documentation started

---

## 💡 Open Questions

1. ~~**Simulation Backend**: Pure NumPy vs. PyTorch for GPU?~~
   **Decision**: PyTorch with GPU. Performance is critical for large-scale simulations.
2. ~~**Neuron Precision**: Float32 sufficient or need Float64?~~
   **Decision**: Float32 is sufficient. Consider quantization (int8/int16) for future scaling.
3. **Scaling Strategy**: How to handle millions of neurons?
4. **Validation**: How do we know it's "thinking" vs. random?
5. **Hardware**: Target neuromorphic chips later (Loihi, SpiNNaker)?

---

## 🤝 Contributing Guidelines

(To be developed as project grows)

- Code style: Black + isort
- Type hints encouraged
- Docstrings for all public APIs
- Tests for new features
- Experiments documented in notebooks

---

*Last updated: 2025-11-28*
*Status: Planning Phase*
