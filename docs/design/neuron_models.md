# Neuron Models

**Status**: 🟢 Current
**Last Updated**: December 7, 2025

This document describes the neuron models implemented in Thalia and planned future extensions.

## Current Models

### Leaky Integrate-and-Fire (LIF)

A simplified current-based model available for comparison and educational purposes.

#### Mathematical Description

The membrane potential $V(t)$ evolves according to:

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)$$

Where:
- $\tau_m$ = membrane time constant (20 ms default)
- $V_{rest}$ = resting potential (-65 mV)
- $R$ = membrane resistance
- $I(t)$ = input current

#### Spike Generation

When $V \geq V_{thresh}$:
1. Emit spike (binary output = 1)
2. Reset: $V \leftarrow V_{reset}$
3. Enter absolute refractory period (optional)

#### Default Parameters

| Parameter | Value | Unit | Biological Range |
|-----------|-------|------|------------------|
| τ_mem | 20 | ms | 10-30 ms |
| V_thresh | -50 | mV | -50 to -40 mV |
| V_reset | -70 | mV | -70 to -60 mV |
| V_rest | -65 | mV | -70 to -60 mV |
| τ_ref | 2 | ms | 1-5 ms |

#### Implementation

See `src/thalia/core/neuron.py` - `LIF` class

---

### Conductance-Based LIF (ConductanceLIF)

**Primary neuron model used in Thalia.** Extends LIF with voltage-dependent synaptic currents and reversal potentials for biological realism and natural stability.

#### Mathematical Description

The membrane equation becomes:

$$C_m \frac{dV}{dt} = g_L(E_L - V) + g_E(E_E - V) + g_I(E_I - V) + g_{adapt}(E_{adapt} - V)$$

Where:
- $g_L$ = leak conductance
- $g_E(t)$ = excitatory synaptic conductance (time-varying)
- $g_I(t)$ = inhibitory synaptic conductance (time-varying)
- $g_{adapt}(t)$ = adaptation conductance (optional, spike-triggered)
- $E_L, E_E, E_I, E_{adapt}$ = reversal potentials for each current

#### Synaptic Conductances

Conductances decay exponentially between synaptic events:

$$\tau_E \frac{dg_E}{dt} = -g_E + \text{(excitatory input)}$$

$$\tau_I \frac{dg_I}{dt} = -g_I + \text{(inhibitory input)}$$

#### Key Advantages

1. **Natural Saturation**: Currents automatically saturate when $V$ approaches reversal potentials
2. **Shunting Inhibition**: Inhibition is divisive (multiplicative effect), not just subtractive
3. **Voltage-Dependent**: Current magnitude depends on driving force $(E - V)$
4. **No Artificial Clamping**: No need for `v_min`/`v_max` bounds

#### Default Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| $C_m$ | 1.0 | a.u. | Membrane capacitance |
| $g_L$ | 0.05 | a.u. | Leak conductance (τ_m ≈ 20ms) |
| $E_L$ | 0.0 | norm | Resting potential |
| $E_E$ | 3.0 | norm | Excitatory reversal (above threshold) |
| $E_I$ | -0.5 | norm | Inhibitory reversal (hyperpolarizing) |
| $\tau_E$ | 5.0 | ms | Excitatory time constant (AMPA-like) |
| $\tau_I$ | 10.0 | ms | Inhibitory time constant (GABA-like) |
| $v_{threshold}$ | 1.0 | norm | Spike threshold |
| $v_{reset}$ | 0.0 | norm | Reset potential |

#### Spike-Frequency Adaptation (Optional)

Enable by setting `adapt_increment > 0`:

$$g_{adapt}(t+1) = g_{adapt}(t) \cdot e^{-dt/\tau_{adapt}} + \delta_{spike} \cdot \text{adapt_increment}$$

This implements a slow potassium current that hyperpolarizes the neuron after spiking, causing firing rate adaptation.

#### Implementation

See `src/thalia/core/neuron.py` - `ConductanceLIF` class

---

## Future Models

The following neuron models are candidates for future implementation based on specific research needs.

### Adaptive Exponential Integrate-and-Fire (AdEx)

**Status**: Planned
**Priority**: Medium

#### Description

Extends LIF with:
1. **Exponential spike initiation**: Captures the rapid depolarization before a spike
2. **Spike-triggered adaptation**: Models calcium-activated potassium currents

#### Equation

$$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V - V_T}{\Delta_T}} - w + I$$

$$\tau_w \frac{dw}{dt} = a(V - E_L) - w$$

Where:
- $\Delta_T$ = spike slope factor (~2 mV)
- $w$ = adaptation variable
- $a$ = subthreshold adaptation coupling

#### Use Cases

- Modeling cortical regular-spiking and fast-spiking interneurons
- Burst firing patterns
- Spike-frequency adaptation with more biological realism than LIF

#### References

- Brette & Gerstner (2005) - "Adaptive Exponential Integrate-and-Fire Model"

---

### Izhikevich Model

**Status**: Planned
**Priority**: Medium

#### Description

A two-variable model that reproduces diverse firing patterns with computational efficiency comparable to LIF.

#### Equations

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$

$$\frac{du}{dt} = a(bv - u)$$

If $v \geq 30$ mV, then: $v \leftarrow c$, $u \leftarrow u + d$

#### Use Cases

- Reproducing diverse cortical neuron types (RS, IB, CH, FS, LTS)
- Modeling thalamic neurons
- Studying bifurcations and dynamical regimes

#### References

- Izhikevich (2003) - "Simple Model of Spiking Neurons"

---

### Hodgkin-Huxley (HH)

**Status**: Deferred
**Priority**: Low (computational cost)

#### Description

The classic biophysical model with voltage-gated sodium and potassium channels.

#### Equations

$$C_m \frac{dV}{dt} = -g_{Na}m^3h(V - E_{Na}) - g_K n^4(V - E_K) - g_L(V - E_L) + I$$

Plus gating variable dynamics for $m, h, n$.

#### Use Cases

- High-fidelity neuron models for detailed biophysical studies
- Modeling ion channel dynamics explicitly
- Drug effects on specific channels

#### Why Deferred

- **Computational cost**: 10-100x slower than LIF
- **Complexity**: Requires careful parameter tuning
- **Overkill**: Most cognitive modeling doesn't need this level of detail

#### When to Implement

- Studying specific ion channel effects (e.g., channelopathies)
- Modeling dendritic computation requiring voltage-gated channels
- High-fidelity single-neuron studies

#### References

- Hodgkin & Huxley (1952) - "A Quantitative Description of Membrane Current"

---

## Model Selection Guide

| Model | Complexity | Speed | Biological Detail | Use Case |
|-------|-----------|-------|-------------------|----------|
| **LIF** | Low | Fast | Minimal | Default, rapid prototyping |
| **ConductanceLIF** | Medium | Fast | Good | Shunting inhibition, realistic dynamics |
| **AdEx** | Medium | Medium | Good | Adaptation, burst firing |
| **Izhikevich** | Medium | Medium | Good | Diverse firing patterns |
| **HH** | High | Slow | Excellent | Detailed biophysics only |

### Current Recommendation

- **Use ConductanceLIF** (default) for:
  - All production code and experiments
  - Realistic synaptic dynamics
  - Proper shunting inhibition
  - Natural E/I balance
- **Use LIF** only for:
  - Quick prototyping and debugging
  - Performance-critical scenarios where biological detail is less important
- **Future models** will be added based on experimental needs

---

## Implementation Notes

### Neuron State Management

All neuron models maintain state in a `NeuronState` dataclass:
- `v`: Membrane potential
- `spikes`: Binary spike output
- `refractory`: Refractory timer (if applicable)
- Model-specific state (e.g., `g_E`, `g_I` for ConductanceLIF)

See `docs/patterns/state-management.md` for details.

### Device Handling

Create neurons with explicit device specification:

```python
from thalia.components.neurons import LIF, ConductanceLIF

# Pattern 1: Device at creation
neuron = LIF(n_neurons=100, device=device)

# Pattern 2: Move after creation
neuron = LIF(n_neurons=100).to(device)
```

### Time Constants

All time constants (τ_mem, τ_E, τ_I, etc.) are in **milliseconds**. The simulation timestep `dt` is typically 0.1-1.0 ms.

---

**See Also**:
- `src/thalia/core/neuron.py` — Neuron implementations
- `docs/patterns/state-management.md` — State handling patterns
- `docs/decisions/adr-003-clock-driven.md` — Why fixed timestep simulation
