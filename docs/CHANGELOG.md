# Changelog

All notable changes to THALIA are documented in this file.

## [0.9.1] - 2025-11-28 🧪 Experiments Complete!

All 5 validation experiments are now passing, demonstrating THALIA's core capabilities.

### Experiments Added
**Commit:** `dba7fd9`

| Experiment | Description | Results |
|------------|-------------|---------|
| **Exp 1: Basic LIF** | LIF neuron dynamics validation | ✅ All neuron behavior correct |
| **Exp 2: STDP Learning** | Spike-timing plasticity | ✅ Weights learned, patterns stable |
| **Exp 3: Attractors** | Pattern storage & recall | ✅ 100% recall at 50% corruption |
| **Exp 4: MNIST SNN** | Classification benchmark | ✅ 43% accuracy (4x above chance) |
| **Exp 5: Spontaneous Thought** | Free association | ✅ 13 transitions, 67% associated |

#### Files
- `experiments/scripts/exp1_basic_lif.py`
- `experiments/scripts/exp2_stdp_learning.py`
- `experiments/scripts/exp3_attractors.py`
- `experiments/scripts/exp4_mnist_snn.py`
- `experiments/scripts/exp5_spontaneous_thought.py`
- `experiments/results/*.png` - Visualization outputs

---

## [0.9.0] - 2025-11-28 🎉 All Core Phases Complete!

THALIA has achieved all planned functionality for spontaneous thinking through spiking neural networks.

### Summary
- **9 phases completed**
- **387 tests passing**
- **~8,000+ lines of code**
- **Full GPU acceleration**

---

### Phase 9: Metacognition 🪞
**Commit:** `9b5713f`

Self-monitoring and adaptive control - the system can now "think about its thinking."

#### Added
- `src/thalia/metacognition/metacognition.py` (~1,170 lines)
  - `ConfidenceTracker` - tracks confidence from activity consistency
  - `UncertaintyEstimator` - MC dropout for epistemic, entropy for aleatoric
  - `ErrorDetector` - prediction errors, conflicts, consistency, timeouts
  - `CognitiveMonitor` - integrates all monitoring components
  - `MetacognitiveController` - generates processing adjustments
  - `MetacognitiveNetwork` - complete SNN-based metacognitive system
- 77 tests, ~205 observations/sec on GPU

---

### Phase 8: Inner Speech 🗣️
**Commit:** `eb9757a`

Language as a tool for thought - internal dialogue and verbal reasoning.

#### Added
- `src/thalia/speech/inner_speech.py` (~1,100 lines)
  - `TokenVocabulary` - manages token embeddings
  - `InnerVoice` - generates internal speech
  - `InnerDialogue` - manages self-dialogue between voices
  - `DialogueManager` - orchestrates multi-turn conversations
  - `ReasoningChain` - tracks reasoning with premises and conclusions
  - `InnerSpeechNetwork` - complete verbal reasoning system
- 63 tests, ~194 utterances/sec on GPU

---

### Phase 7: World Model & Prediction 🔮
**Commit:** `3736a66`

Internal simulation of external reality with predictive processing.

#### Added
- `src/thalia/world/world_model.py` (~900 lines)
  - `StateTransitionModel` - learns environment dynamics
  - `ActionModel` - predicts action outcomes
  - `PredictionErrorTracker` - monitors prediction accuracy
  - `BeliefState` - probabilistic world beliefs
  - `WorldModelSNN` - complete predictive processing system
- 40 tests

---

### Phase 6: Daydream Mode 💭
**Commit:** `92eef5c`

Spontaneous cognition without external input.

#### Added
- `src/thalia/cognition/daydream.py` (~750 lines)
  - `DaydreamMode` enum: FREE, THEMED, DREAM, GOAL
  - `DaydreamEngine` - generates spontaneous thought streams
  - Novelty-seeking transitions, recency effects
- 33 tests

---

### Phase 5b: Hierarchical Architecture 🏛️
**Commit:** `9a772c1`

Multi-level temporal processing with bidirectional connections.

#### Added
- `src/thalia/hierarchy/hierarchical.py` (~600 lines)
  - Multi-layer with time constants: sensory (5ms) → abstract (200ms)
  - Bottom-up feature extraction, top-down predictions
- 33 tests

---

### Phase 5a: ThinkingSNN Integration 🧠
**Commit:** `9c9dae9`

Integrated cognitive architecture combining all components.

#### Added
- `src/thalia/cognition/thinking.py` (~500 lines)
  - `ThinkingSNN` - main cognitive architecture
  - `think()` method for deliberate cognition
- 23 tests

---

### Phase 4: Working Memory 🧠
**Commit:** `ace55b8`

Persistent neural activity without input.

#### Added
- `src/thalia/memory/working_memory.py` (~450 lines)
  - Multi-slot working memory with gated access
  - Self-sustained activity, decay dynamics
- 31 tests

---

### Phase 3: Attractor Dynamics 🌀
**Commit:** `69d0bb3`

Stable concept representations through attractor networks.

#### Added
- `src/thalia/dynamics/attractor.py` (~400 lines)
  - Hopfield-like attractor network
  - Pattern storage, completion, transitions
- `src/thalia/dynamics/manifold.py` (~350 lines)
- 48 tests (attractor + manifold)

---

### Phase 2: Learning Rules 📚
**Commit:** `2d1489d`

Biologically-plausible learning mechanisms.

#### Added
- `src/thalia/learning/stdp.py` - Classic STDP
- `src/thalia/learning/homeostatic.py` - Intrinsic plasticity, synaptic scaling
- `src/thalia/learning/reward.py` - R-STDP, eligibility traces
- 21 tests

---

### Phase 1: Core SNN Infrastructure 🔧
**Commit:** `2d1489d`

Basic spiking neural network primitives.

#### Added
- `src/thalia/core/neuron.py` - LIF neurons with configurable τ
- `src/thalia/core/layer.py` - SNN layers with recurrent connections
- `src/thalia/encoding/` - Rate, temporal, Poisson coding
- 18 tests

---

## [0.1.0] - 2025-11-28

### Initial Commit
**Commit:** `713c284`

- Project scaffolding and structure
- `pyproject.toml` configuration
- Documentation framework

---

## Architecture

```
thalia/
├── src/thalia/
│   ├── core/           # LIF neurons, layers
│   ├── learning/       # STDP, homeostatic, reward
│   ├── dynamics/       # Attractors, manifolds
│   ├── memory/         # Working memory
│   ├── hierarchy/      # Hierarchical processing
│   ├── cognition/      # ThinkingSNN, daydream
│   ├── world/          # World model, prediction
│   ├── speech/         # Inner speech, dialogue
│   ├── metacognition/  # Self-monitoring
│   └── encoding/       # Spike encoding
├── tests/              # 387 tests
├── examples/           # Demo scripts
└── docs/               # Documentation
```

## Performance (RTX 3050 Ti)

| Component | Throughput |
|-----------|------------|
| Inner Speech | ~194 utterances/sec |
| Metacognition | ~205 observations/sec |
| Full ThinkingSNN | ~100 think cycles/sec |
