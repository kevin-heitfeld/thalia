# Thalia Brain Regions Experiment Plan

This document outlines a systematic experimental program to validate and demonstrate
the brain regions implemented in Thalia. The experiments progress from single-region
validation to multi-region integration and emergent cognition.

## Overview

The Thalia library implements five biologically-inspired brain regions:

| Region | Learning Rule | Biological Function |
|--------|---------------|---------------------|
| **Cortex** | Hebbian + BCM | Unsupervised feature extraction |
| **Cerebellum** | Delta rule (error-corrective) | Supervised motor/nonlinear learning |
| **Striatum** | Three-factor RL | Reward-based action selection |
| **Hippocampus** | One-shot Hebbian | Episodic memory, pattern completion |
| **Prefrontal** | Gated working memory | Cognitive control, context maintenance |

## Experimental Phases

### Phase 1: Single Region Validation ✅ COMPLETE

Validate each region independently with canonical tasks.

| Experiment | Region | Task | Success Criterion | Status |
|------------|--------|------|-------------------|--------|
| exp1_cortex_predictive | Cortex | Predictive coding on MNIST | >50% pred error reduction, NMI>0.1 | ✅ PASSED |
| exp2_cerebellum_timing | Cerebellum | Interval timing | <20% timing error, multi-delay learning | ✅ PASSED |
| exp3_striatum_bandit | Striatum | Multi-armed bandit | Converges to optimal arm 90%+ | ✅ PASSED |
| exp4_hippocampus_memory | Hippocampus | Hetero-associative memory | 80%+ F1 recall | ✅ PASSED |
| exp5_prefrontal_delay | Prefrontal | Delayed match-to-sample | 85%+ accuracy with delay=20 | ✅ PASSED |

### Phase 2: Two-Region Compositions

Test interactions between pairs of regions.

| Experiment | Regions | Task | Success Criterion |
|------------|---------|------|-------------------|
| exp6_cortex_striatum | Cortex + Striatum | Visual categorization with reward | Learns category-reward mappings |
| exp7_hippocampus_prefrontal | Hippocampus + Prefrontal | Context-dependent memory recall | Context switches memory retrieval |
| exp8_cerebellum_cortex | Cerebellum + Cortex | Supervised feature refinement | Cortex features improve with error signal |
| exp9_striatum_prefrontal | Striatum + Prefrontal | Rule-based action selection | Follows abstract rules, not just reward |

### Phase 3: Multi-Region Systems

Integrate 3+ regions for complex cognitive tasks.

| Experiment | Regions | Task | Success Criterion |
|------------|---------|------|-------------------|
| exp10_working_memory_rl | Prefrontal + Striatum + Hippocampus | Memory-guided decision making | Uses past context to optimize choices |
| exp11_motor_learning | Cortex + Cerebellum + Striatum | Sequence learning with timing | Learns timed motor sequences |
| exp12_cognitive_control | All 5 regions | Task switching | Adapts behavior to changing rules |

### Phase 4: Emergent Cognition

Explore emergent properties from full integration.

| Experiment | Description | Emergent Property |
|------------|-------------|-------------------|
| exp13_spontaneous_replay | Memory consolidation during idle | Hippocampal replay without input |
| exp14_mental_simulation | Planning via internal simulation | Prefrontal "imagines" action outcomes |
| exp15_metacognition | Self-monitoring and adaptation | System recognizes own uncertainty |

---

## Phase 1: Detailed Specifications

### Experiment 1: Cortex Predictive Coding ✅

**Objective**: Demonstrate that Cortex learns useful feature representations via
predictive coding - minimizing prediction error between top-down predictions and
bottom-up input.

**Setup**:
- Input: MNIST digits (7x7 downsampled = 49 features)
- Architecture: 49 input → 64 cortex neurons
- Learning: Predictive coding with inference steps
- Duration: 2000 training samples, 10 epochs

**Analysis**:
1. Prediction error reduction over training
2. Clustering quality (NMI, ARI) of learned representations
3. t-SNE visualization of cortex representations
4. Comparison with PCA and random projection baselines

**Success Metrics** (2/3 required):
- Prediction error improves >50%
- kNN accuracy within 10% of raw pixels
- NMI > 0.1

**Result**: PASSED (86% pred error reduction, NMI=0.285)

---

### Experiment 2: Cerebellum Interval Timing ✅

**Objective**: Validate that Cerebellum can learn precise timing - a core
cerebellar function analogous to eyeblink conditioning.

**Setup**:
- Input: Temporal basis functions (64 neurons with different peak times)
- Task: Produce output at specific delays (18, 22, 38 timesteps)
- Architecture: 64 input → 1 Purkinje cell
- Learning: Delta rule with climbing fiber error signals

**Analysis**:
1. Timing error (% deviation from target)
2. Accuracy (responses within ±5 timestep window)
3. Learning curves for multiple target delays
4. Before/after training comparison

**Success Metrics** (2/3 required):
- Best timing error < 20%
- All delays achieve ≥50% accuracy
- Training improves timing for ≥2/3 delays

**Result**: PASSED (3/3 criteria - 5.3% timing error, 100% accuracy)

---

### Experiment 3: Striatum Multi-Armed Bandit ✅

**Objective**: Validate reward-modulated learning for action selection.

**Setup**:
- Environment: 4-armed bandit with reward probabilities [0.2, 0.4, 0.6, 0.8]
- Architecture: Context → Striatum (4 actions)
- Learning: Three-factor rule (eligibility × dopamine)
- Episodes: 1000 trials

**Analysis**:
1. Learning curve (optimal action rate over time)
2. Action probability evolution
3. Comparison with ε-greedy baseline
4. Adaptation to reward distribution changes

**Success Metrics**:
- Converges to optimal arm 90%+ of trials

**Result**: PASSED (99% optimal action rate)

---

### Experiment 4: Hippocampus Hetero-Associative Memory ✅

**Objective**: Demonstrate one-shot episodic memory formation and pattern completion.

**Setup**:
- Input: 20 pattern pairs (64-bit cue → 64-bit target)
- Architecture: 64 cue → weights → 64 target (hetero-associative)
- Learning: One-shot Hebbian (single exposure)
- Test: Present cue, measure target recall via weight-based readout

**Analysis**:
1. Recall F1 score for stored patterns
2. Memory capacity (F1 vs number of patterns)
3. Pattern completion with partial cues (75%, 50%, 25%)
4. Interference between patterns

**Success Metrics**:
- Average F1 ≥ 80%

**Result**: PASSED (95% F1 recall)

---

### Experiment 5: Prefrontal Delayed Match-to-Sample ✅

**Objective**: Validate working memory maintenance across delay periods.

**Setup**:
- Task: Present sample → delay (5-50 timesteps) → match/non-match
- Architecture: Sample → PFC (gated WM) → decision
- Challenge: Information must persist without external input

**Analysis**:
1. Accuracy vs delay length
2. WM state visualization during delay
3. Effect of distractors during delay
4. Gating dynamics

**Success Metrics**:
- 85%+ accuracy with 20-step delay

**Result**: PASSED (94% accuracy at delay=20)

---

## Implementation Notes

### File Structure

```
experiments/scripts/
├── regions/             # New brain region experiments
│   ├── __init__.py
│   ├── exp_utils.py     # Shared utilities
│   ├── phase1/          # Single region experiments ✅
│   │   ├── exp1_cortex_predictive.py
│   │   ├── exp2_cerebellum_timing.py
│   │   ├── exp3_striatum_bandit.py
│   │   ├── exp4_hippocampus_memory.py
│   │   └── exp5_prefrontal_delay.py
│   ├── phase2/          # Two-region compositions
│   ├── phase3/          # Multi-region systems
│   └── phase4/          # Emergent cognition
```

### Running Experiments

Each experiment can be run standalone:

```bash
python -m experiments.scripts.regions.phase1.exp1_cortex_clustering
```

Or via the experiment runner:

```bash
python -m experiments.run --experiment exp1 --config experiments/configs/exp1.yaml
```

### Common Utilities

All experiments share:
- `exp_utils.py`: Data loading, metrics, visualization
- Standard logging format
- Results saved to `experiments/results/regions/`
- Automatic hyperparameter tracking

---

## Timeline

| Week | Phase | Experiments |
|------|-------|-------------|
| 1 | Phase 1 | exp1-exp5 (single region validation) |
| 2 | Phase 2 | exp6-exp9 (two-region compositions) |
| 3 | Phase 3 | exp10-exp12 (multi-region systems) |
| 4 | Phase 4 | exp13-exp15 (emergent cognition) |

---

## References

1. BCM Theory: Bienenstock, Cooper, Munro (1982)
2. Cerebellar Learning: Marr-Albus-Ito models
3. Striatal RL: Schultz et al., dopamine and reward prediction
4. Hippocampal Memory: Hopfield networks, pattern separation
5. Prefrontal WM: O'Reilly, Frank - PBWM model
