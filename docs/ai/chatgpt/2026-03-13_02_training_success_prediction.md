Two diagnostics are particularly predictive of whether a large spiking network will **train successfully** rather than drifting into saturation or silence. They are widely used in reinforcement-learning SNNs and large cortical simulations.

---

# 1. Branching ratio (network stability)

The most important early diagnostic is the **branching ratio**.

It measures how many spikes one spike produces in the next timestep.

[
\sigma = \frac{E[N_{t+1}]}{E[N_t]}
]

Where (N_t) is the number of spikes in a time bin.

### Interpretation

| branching ratio | regime                      |
| --------------- | --------------------------- |
| σ < 0.9         | subcritical (activity dies) |
| 0.9–1.1         | critical / optimal          |

> 1.2 | supercritical (runaway activity) |

Healthy cortex sits extremely close to:

[
\sigma \approx 1
]

This produces **scale-free avalanches and maximal information propagation**.

### Why this predicts training success

Learning rules such as:

* Spike-Timing-Dependent Plasticity
* Reward-Modulated STDP

require activity that is:

* not dying out
* not exploding

If σ is wrong, plasticity either:

* never triggers, or
* destabilizes the network.

### Implementation

Compute spike counts in 1–5 ms bins:

```python
N_t = spike_counts[:-1]
N_t1 = spike_counts[1:]

sigma = np.mean(N_t1) / np.mean(N_t)
```

Better estimate:

```python
sigma = np.sum(N_t1 * N_t) / np.sum(N_t**2)
```

---

# 2. Excitation–inhibition lag correlation

Healthy cortical networks maintain **tight timing balance** between excitation and inhibition.

Diagnostic:

Compute cross-correlation of population conductances:

[
C(\tau) = corr(g_E(t), g_I(t+\tau))
]

Expected pattern:

| lag    | correlation     |
| ------ | --------------- |
| 0–2 ms | strong positive |

> 10 ms | near zero |

Interpretation:

* excitation slightly **leads inhibition**
* inhibition quickly **stabilizes activity**

This produces the **asynchronous irregular regime**.

If you observe:

| pattern                        | meaning             |
| ------------------------------ | ------------------- |
| E and I perfectly synchronized | oscillatory locking |
| I leads E                      | network suppression |
| E dominates                    | runaway excitation  |

---

# 3. What a healthy large-scale SNN shows

When all diagnostics are combined:

| metric                | healthy value |
| --------------------- | ------------- |
| branching ratio       | 0.95–1.05     |
| ISI CV                | ~1            |
| pairwise correlation  | 0.01–0.05     |
| E/I conductance ratio | 3–6           |
| E→I lag               | 1–3 ms        |
| population spectrum   | broadband     |

If a network satisfies those simultaneously, it usually trains stably.

---

# 4. The subtle diagnostic almost nobody checks

Large simulations such as those used by the Blue Brain Project also monitor **effective synaptic gain**:

[
G = \frac{\partial r_{out}}{\partial r_{in}}
]

If:

| gain  | behaviour |
| ----- | --------- |
| G < 1 | stable    |
| G ≈ 1 | critical  |
| G > 1 | unstable  |

Learning works best when:

[
G \approx 0.9
]

This keeps the network sensitive but stable.

---

# 5. Practical workflow neuroscientists use

For a new SNN brain simulation:

1. run **5–10 seconds spontaneous activity**

2. compute

   * firing rate distributions
   * ISI CV
   * branching ratio
   * E/I lag correlation
   * population spectrum

3. only then enable plasticity.

Skipping this step is why many SNN projects appear unstable during learning.
