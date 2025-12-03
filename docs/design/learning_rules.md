# Learning Rules

## Overview

Thalia implements several biologically-inspired learning rules. The key distinction is between:

1. **Rate-coded learning** - Uses analog firing rates as proxies for activity
2. **Spike-based learning** - Uses actual discrete spike events and timing

For true spiking neural networks, spike-based rules are essential. Rate-coded rules are faster to simulate but lose temporal precision.

---

## Spike-Timing-Dependent Plasticity (STDP)

### Mechanism

Weight change depends on relative spike timing:

$$\Delta w = \begin{cases} A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \text{ (post after pre)} \\ -A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \text{ (pre after post)} \end{cases}$$

Where $\Delta t = t_{post} - t_{pre}$

### Implementation

Uses eligibility traces for efficient batch computation:
- Pre-synaptic trace: updated on pre spikes, decays exponentially
- Post-synaptic trace: updated on post spikes, decays exponentially

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| τ_+ | 20 ms | LTP time constant |
| τ_- | 20 ms | LTD time constant |
| A_+ | 0.01 | LTP amplitude |
| A_- | 0.0105 | LTD amplitude (slight depression bias) |

---

## Predictive STDP (PREDICTIVE_STDP)

### Overview

A three-factor learning rule that combines STDP with prediction error modulation. This is Thalia's first truly spike-based learning rule that works with discrete spike events.

**Key insight**: Learning should only occur when predictions are wrong. If the network correctly predicts its inputs, no weight change is needed.

### Algorithm

```
For each timestep:
  1. Update eligibility traces from spike correlations
  2. Compute prediction error (actual vs predicted input)
  3. Apply STDP modulated by prediction error magnitude
  4. Normalize weights to prevent homogenization
```

### Mathematical Formulation

**Eligibility Traces:**
$$\text{pre\_trace}_i(t) = \lambda \cdot \text{pre\_trace}_i(t-1) + \text{pre\_spike}_i(t)$$
$$\text{post\_trace}_j(t) = \lambda \cdot \text{post\_trace}_j(t-1) + \text{post\_spike}_j(t)$$

Where $\lambda = 1 - \frac{dt}{\tau_{STDP}}$

**STDP Update:**
$$\text{LTP}_{ji} = \text{post\_spike}_j \cdot \text{pre\_trace}_i$$
$$\text{LTD}_{ji} = \text{post\_trace}_j \cdot \text{pre\_spike}_i$$

**Soft Bounds (prevents saturation):**
$$w_{norm} = \frac{w - w_{min}}{w_{max} - w_{min}}$$
$$\text{LTP}_{soft} = \text{LTP} \cdot (1 - w_{norm})$$
$$\text{LTD}_{soft} = \text{LTD} \cdot w_{norm}$$

**Competitive Anti-Hebbian (forces specialization):**
$$\text{anti\_hebbian}_{ji} = (1 - \text{post\_spike}_j) \cdot \text{pre\_spike}_i \cdot w_{norm}$$

**Combined STDP:**
$$\Delta w_{STDP} = \eta \cdot (\text{LTP}_{soft} - \alpha \cdot \text{LTD}_{soft} - \beta \cdot \text{anti\_hebbian})$$

**Prediction Error Modulation:**
$$\text{pred\_error} = \text{MSE}(\text{predicted\_input}, \text{actual\_input})$$
$$\text{modulation} = m_{min} + (1 - m_{min}) \cdot \sigma(k \cdot (\text{pred\_error} - \theta))$$
$$\Delta w = \Delta w_{STDP} \cdot \text{modulation}$$

**Weight Normalization (per neuron):**
$$w_{ji} \leftarrow w_{ji} \cdot \frac{w_{target}}{\sum_i w_{ji} + \epsilon}$$

### Key Components

| Component | Purpose |
|-----------|---------|
| STDP eligibility | Captures spike timing correlations |
| Soft bounds | Prevents weight saturation |
| Anti-Hebbian | Forces neurons to specialize |
| Prediction error | Third factor - gates when learning occurs |
| Weight normalization | Prevents all neurons learning same thing |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hebbian_lr | 0.005 | Base learning rate |
| pred_error_modulation | 0.5 | How strongly error gates learning |
| pred_error_tau_ms | 20.0 | Smoothing time constant for error |
| pred_error_min_mod | 0.2 | Minimum learning even at low error |
| heterosynaptic_ratio | 0.3 | LTD relative to LTP |
| stdp_tau_plus_ms | 20.0 | Eligibility trace time constant |

### Usage

```python
from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.base import LearningRule

config = CortexConfig(
    n_input=64,
    n_output=32,
    learning_rule=LearningRule.PREDICTIVE_STDP,
    hebbian_lr=0.005,
    pred_error_modulation=0.5,
    kwta_k=8,  # Recommended for competition
)
cortex = Cortex(config)

# Training loop
for input_spikes in spike_data:
    output_spikes = cortex.forward(input_spikes)
    metrics = cortex.learn(input_spikes, output_spikes)
```

### Performance

On a 4-pattern classification task with Poisson-encoded inputs:
- **Accuracy**: 35-55% (above 25% chance)
- **Prediction error reduction**: ~45%
- **Weight stability**: Normalized, not saturating

### When to Use

- True spiking networks where spike timing matters
- Unsupervised feature learning
- Competitive learning / winner-take-all networks
- When prediction error is a meaningful signal

---

## Homeostatic Plasticity

### BCM Rule (Bienenstock-Cooper-Munro)

Implements sliding threshold for LTP/LTD transition:

$$\Delta w \propto y(y - \theta_m)$$

Where $\theta_m$ is the modification threshold that slides based on recent activity.

### Synaptic Scaling

TODO: Multiplicative scaling to maintain target firing rates.

### Intrinsic Plasticity

TODO: Adjustment of neuron excitability.

---

## Reward-Modulated Learning

### Overview

Three-factor rules where weight changes require:
1. Pre-synaptic activity (eligibility trace)
2. Post-synaptic activity
3. Reward/dopamine signal (modulation)

### THREE_FACTOR (Rate-Coded)

Classic three-factor learning with rate-coded eligibility:

$$\Delta w = \eta \times \text{eligibility} \times \text{dopamine}$$

Where eligibility is computed from pre-post activity correlation (outer product).

### REWARD_MODULATED_STDP (Spike-Based)

Combines STDP with dopamine modulation for spike-based reinforcement learning:

1. **STDP eligibility traces** - Computed from spike timing, not just correlation
2. **Dopamine as third factor** - Gates whether eligibility converts to weight change
3. **Soft bounds** - Prevents weight saturation
4. **Competitive anti-Hebbian** - Non-spiking neurons weaken to active inputs

**Key difference from THREE_FACTOR**: STDP eligibility captures the DIRECTION of learning (LTP vs LTD based on timing), while dopamine modulates the MAGNITUDE.

**Algorithm:**
```
For each timestep:
  1. Update STDP traces (pre_trace, post_trace)
  2. Compute LTP = post_spike × pre_trace
  3. Compute LTD = post_trace × pre_spike
  4. Apply soft bounds to LTP and LTD
  5. Accumulate into eligibility (long timescale: 100-1000ms)

When reward arrives:
  6. Compute dopamine signal from reward
  7. Apply: Δw = eligibility × dopamine
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| stdp_lr | 0.02 | STDP learning rate |
| stdp_tau_ms | 20.0 | STDP trace time constant |
| eligibility_tau_ms | 500.0 | Eligibility persistence |
| dopamine_burst | 1.0 | Reward signal magnitude |
| dopamine_dip | -0.5 | Punishment signal magnitude |

**Usage:**
```python
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.base import LearningRule

config = StriatumConfig(
    n_input=64,
    n_output=4,
    learning_rule=LearningRule.REWARD_MODULATED_STDP,
    stdp_lr=0.02,
    eligibility_tau_ms=500.0,
)
striatum = Striatum(config)

# Training loop
for input_spikes in spike_data:
    output_spikes = striatum.forward(input_spikes)
    # Build eligibility (no reward yet)
    striatum.learn(input_spikes, output_spikes)

# When reward is known
striatum.learn(last_input, last_output, reward=1.0)
```

### Known Challenges

**Exploration Problem**: RL from scratch with random initialization faces the fundamental exploration-exploitation tradeoff. With symmetric patterns, positive and negative rewards can cancel out, preventing learning. Solutions include:
- Epsilon-greedy exploration
- Optimistic initialization
- Asymmetric learning rates (DA burst > |DA dip|)
- Temperature annealing

This is not specific to spiking networks - it affects all RL systems.
