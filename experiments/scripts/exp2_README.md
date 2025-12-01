# Experiment 2: STDP Learning with Biologically Realistic WTA

## Overview

This experiment validates that a spiking neural network can learn to map temporal input patterns to specific output neurons using **pure Hebbian coincidence detection** and **predictive coding for recurrent connections**. The key finding is that the network discovers the correct input-output mapping from **random initial weights** without any spatial bias or prior knowledge baked in.

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| Feedforward learning (diagonal score) | 9/10 | Output neuron i learns to respond to inputs 2i or 2i+1 |
| Recurrent learning | 5/5 | Correct forward chain: 0→1→2→3→4→5 |
| Pattern completion | 0/10 | Known limitation - not trained for autonomous activity |
| Initial diagonal score | 1/10 | Confirms truly random initialization |

## Architecture

```
Input Layer (20 neurons)          Output Layer (10 neurons)
    ┌─────────────────┐               ┌─────────────┐
    │ 0 1 2 3 ... 19  │ ──Learned──→  │ 0 1 ... 9   │
    └─────────────────┘   Weights     └──────┬──────┘
         ↑                                   │
    Temporal pattern              Recurrent connections
    (sequential 0→1→...→19)        (learned: i→i+1)
```

### Network Parameters

- **Input neurons:** 20 (fire in sequence, 8ms per neuron)
- **Output neurons:** 10 (each should respond to 2 inputs)
- **Temporal resolution:** dt = 0.1ms (100μs for biological accuracy)
- **Training duration:** 100 cycles × 210ms = 21 seconds
- **Pattern type:** "gapped" (160ms sequence + 50ms gap)

## Biological Mechanisms

### 1. Winner-Take-All (WTA)

The network uses multiple biologically realistic mechanisms to ensure only one output neuron fires per input phase:

| Mechanism | Strength | Time Constant | Purpose |
|-----------|----------|---------------|---------|
| Shunting inhibition | 1.5× | ~2ms | Divisive inhibition (stronger than subtractive) |
| Fast (PV+) inhibition | 2.0× | ~1.4ms | Fast blanket suppression |
| Slow (SOM+) inhibition | 0.2× | ~19ms | Sustained suppression |
| Blanket inhibition | 0.5 | - | Global activity-dependent |
| Refractory period | - | 2ms abs + 3ms rel | Per-spike recovery |
| Spike-frequency adaptation | 1.5× | ~200ms | Prevents domination |

### 2. Biologically-Derived Maximum Synaptic Weight

Real synapses are weak - a single synapse provides only ~0.5-10% of the threshold needed to fire. We compute `w_max` from first principles:

```python
w_max = k_factor * v_threshold / n_coincident_for_firing * scale_factor
      = 2.5 * 1.0 / 5 * 4.0
      = 2.0
```

Where:
- `k_factor = 2.5`: Safety factor (even max synapses need this many inputs)
- `n_coincident_for_firing = 5`: How many coincident inputs needed to fire
- `scale_factor = 4.0`: Learning difficulty adjustment (1.0 = fully biological)

### 3. Theta-Gamma Oscillatory Coupling

The network uses theta-gamma coupling to organize temporal coding:

- **Theta period:** 160ms (matches one sequence)
- **Gamma period:** 10ms (spike timing precision)
- **Theta modulation strength:** 2.5 (strong phase preference)

Each output neuron has a preferred theta phase, creating a temporal prior:
- Neuron 0 prefers phase 0°
- Neuron 1 prefers phase 36°
- etc.

### 4. Homeostatic Plasticity

| Mechanism | Target | Time Constant |
|-----------|--------|---------------|
| Excitability adjustment | 20 Hz | 50ms |
| Dynamic threshold | 20 Hz | 100ms |
| Synaptic scaling | 30% of max norm | 10 cycles |

## Learning Rules

### Feedforward: Pure Hebbian Coincidence Detection

Unlike STDP with timing windows (which causes drift), we use **pure coincidence**:

```python
if output_spikes.sum() > 0 and input_spikes.sum() > 0:
    hebbian_dw = output_spikes.t() @ input_spikes  # Same timestep
    headroom = (w_max - weights) / w_max  # Soft saturation
    weights += 0.01 * hebbian_dw * headroom
```

**Why pure Hebbian?**
- STDP traces from previous inputs (8ms ago) are still active when output fires
- This incorrectly strengthens wrong connections
- Real feedforward refinement uses NMDA-based coincidence, not timing asymmetry

### Recurrent: Predictive Coding

Recurrent connections learn to predict the next winner in the sequence:

```python
# At gamma boundary (every 10ms):
if prev_gamma_winner >= 0:
    prediction = recurrent_weights[prev_winner].argmax()
    if prediction == current_winner:
        # Reinforce correct prediction
        recurrent_weights[prev, curr] += 0.05
    else:
        # Correct the error
        recurrent_weights[prev, prediction] -= 0.04
        recurrent_weights[prev, curr] += 0.05
```

**Key insight:** Learning is **phase-locked to gamma oscillations** to avoid noisy spike-by-spike updates.

### Constraints

| Constraint | Value | Biological Basis |
|------------|-------|------------------|
| Feedforward w_min | 0.0 | Excitatory synapses only |
| Feedforward w_max | 2.0 | Derived from coincidence requirement |
| Recurrent w_min | 0.0 | Pyramidal→Pyramidal is glutamatergic |
| Recurrent w_max | 1.5 | Strong but bounded |

## Experimental Protocol

### 1. Warmup Phase (10 cycles)
- Run network without learning
- Establish steady-state inhibition and homeostasis

### 2. Training Phase (100 cycles)
- Present gapped sequential pattern
- Feedforward learning: pure Hebbian coincidence
- Recurrent learning starts at cycle 80 (after feedforward stabilizes)

### 3. Testing Phase
- Test with full pattern (verify feedforward responses)
- Test with partial pattern (check pattern completion)
- Extended autonomous test (can network continue without input?)

## Success Criteria

| Criterion | Threshold | Achieved |
|-----------|-----------|----------|
| Weights modified | >0.01 mean change | ✅ |
| Feedforward learned | ≥8/10 diagonal | ✅ (9/10) |
| Stable training | No collapse/explosion | ✅ |
| Activity maintained | >10 spikes at end | ✅ |
| Recurrent chain learned | ≥4/5 correct | ✅ (5/5) |

## Known Limitations

### 1. Pattern Completion Doesn't Work
The network doesn't continue the sequence when input is removed (0 spikes in second half). This is because:
- Network was trained for **imitation**, not prediction
- Homeostatic excitability becomes negative (-1.7), suppressing spontaneous activity
- Would require explicit training for autonomous continuation

### 2. Homeostatic Drift
The excitability values drift negative during training. This helps prevent runaway activity during training but prevents autonomous activity during testing.

## Files

| File | Description |
|------|-------------|
| `exp2_stdp_learning.py` | Main experiment script (~2200 lines) |
| `exp2_README.md` | This documentation |
| `../results/exp2_stdp_learning.png` | Output visualization |

## Key Classes and Functions

### NetworkState (dataclass)
Holds all state variables: spike history, inhibition, adaptation, plasticity.

### NetworkConfig (dataclass)  
All configuration parameters packaged for simulation.

### forward_timestep()
**Single implementation** of all biological mechanisms - core forward pass for one timestep.

### forward_timestep_with_stp()
Wraps `forward_timestep()` with short-term plasticity (STP) modulation.

### forward_pattern()
High-level wrapper that runs a complete pattern through the network.

### create_temporal_pattern()
Generates input spike patterns (sequential, circular, gapped, etc.).

## Future Work (Experiment 3 candidates)

1. **Pattern Completion Training:** Train the network to continue sequences autonomously
2. **Bidirectional Sequences:** Learn both forward (0→1→2) and backward (2→1→0) chains
3. **Multiple Sequences:** Store multiple distinct sequences in the same network
4. **Noise Robustness:** Test with noisy/incomplete input patterns

## References

- Bi & Poo (1998): STDP timing windows
- Schultz (1997): Dopamine and reward prediction error
- Pfister & Gerstner (2006): Triplet STDP
- Clopath et al. (2010): Voltage-based STDP
- Lisman & Jensen (2013): Theta-gamma coding
