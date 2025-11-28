# Learning Rules

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

## Homeostatic Plasticity

TODO: Synaptic scaling, intrinsic plasticity

## Reward-Modulated Learning

TODO: Three-factor rules, dopamine modulation
