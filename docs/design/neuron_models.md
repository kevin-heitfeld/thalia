# Neuron Models

## Leaky Integrate-and-Fire (LIF)

The primary neuron model used in THALIA.

### Mathematical Description

The membrane potential $V(t)$ evolves according to:

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R \cdot I(t)$$

Where:
- $\tau_m$ = membrane time constant (20 ms default)
- $V_{rest}$ = resting potential (-65 mV)
- $R$ = membrane resistance
- $I(t)$ = input current

### Spike Generation

When $V \geq V_{thresh}$:
1. Emit spike
2. Reset: $V \leftarrow V_{reset}$
3. Optional: enter refractory period

### Default Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| τ_mem | 20 | ms |
| V_thresh | -50 | mV |
| V_reset | -70 | mV |
| V_rest | -65 | mV |

## Future Models

- Adaptive Exponential IF (AdEx)
- Izhikevich
- Hodgkin-Huxley (for detailed biophysics)
